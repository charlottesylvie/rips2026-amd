"""
minplus_algebra.py

Core min-plus (tropical) semiring primitives.

In the (min, +) semiring:
    "addition"       a (+) b  :=  min(a, b)      identity = +inf
    "multiplication"  a (x) b  :=  a + b          identity = 0

Under this algebra, the single-source shortest path (SSSP) relaxation
step of Bellman-Ford:

    d[v] = min( d[v], min_{(u,v) in E} ( d[u] + w(u,v) ) )

is *exactly* a min-plus matrix-vector product:

    d_new = A (x) d_old        (tropical mat-vec multiply)

where A is the (sparse) weighted adjacency matrix with A[v, u] = w(u, v)
(edge from u to v) and A[v, u] = +inf where no edge exists.

This is the same trick used in GraphBLAS / linear-algebraic graph
algorithms (Kepner & Gilbert) and is precisely what we want to map onto
GPU sparse matrix kernels, since "tropical SpMV" has the same sparsity
pattern / memory access behavior as ordinary SpMV -- only the
multiply-accumulate operators change (+ -> min, * -> +).
"""

import numpy as np
import scipy.sparse as sp

INF = np.float64(np.inf)


def tropical_spmv_dense_row(A: sp.csr_matrix, d: np.ndarray) -> np.ndarray:
    """
    Reference (slow, obviously-correct) tropical SpMV: for each row v,
        d_new[v] = min over u in N(v) of (d[u] + A[v, u])
    Implemented with explicit Python loops over the CSR structure --
    intentionally naive, used only to cross-check the vectorized version.
    """
    n = A.shape[0]
    d_new = np.full(n, INF, dtype=np.float64)
    indptr, indices, data = A.indptr, A.indices, A.data
    for v in range(n):
        start, end = indptr[v], indptr[v + 1]
        best = INF
        for k in range(start, end):
            u = indices[k]
            w = data[k]
            cand = d[u] + w
            if cand < best:
                best = cand
        d_new[v] = best
    return d_new


def tropical_spmv(A: sp.csr_matrix, d: np.ndarray) -> np.ndarray:
    """
    Vectorized tropical SpMV using CSR structure directly.

    For each row v: d_new[v] = min_{k in row v} ( d[indices[k]] + data[k] )

    This avoids Python-level inner loops by using np.minimum.reduceat
    over the flattened (gathered) candidate-value array, which is the
    CPU analogue of what a GPU kernel does per-row (each "row" maps to
    one thread / one warp-cooperative reduction).
    """
    n = A.shape[0]
    indptr, indices, data = A.indptr, A.indices, A.data

    if len(indices) == 0:
        return np.full(n, INF, dtype=np.float64)

    # Gather: candidate[k] = d[u] + w  for every nonzero entry k = (v,u)
    candidates = d[indices] + data

    # Rows with at least one nonzero get reduced via reduceat; empty rows
    # (isolated nodes, or nodes with no in-neighbors in this orientation)
    # are left at the +inf default and never touched.
    row_has_entries = indptr[1:] > indptr[:-1]
    n_rows = len(row_has_entries)

    d_new = np.full(n_rows, INF, dtype=np.float64)
    if not np.any(row_has_entries):
        return d_new

    # CORRECTNESS NOTE: np.minimum.reduceat requires STRICTLY meaningful
    # (non-duplicate) boundaries to reduce a full segment -- if two
    # boundaries are equal, reduceat degenerates to returning a single
    # element for that slot instead of reducing the intended range. This
    # bites exactly when an empty row sits adjacent to a non-empty one
    # (their boundaries can collide). The robust fix: call reduceat ONLY
    # on the boundaries of rows that actually have entries (these are
    # guaranteed strictly increasing, since each such row has >=1 nonzero
    # contributing a unique starting index), then scatter the reduced
    # values back into the rows that have entries. Empty rows are never
    # touched and remain at their INF initial value.
    real_starts = indptr[:-1][row_has_entries]
    reduced = np.minimum.reduceat(candidates, real_starts)
    d_new[row_has_entries] = reduced
    return d_new


def tropical_matrix_from_edges(n_nodes, edges, orientation="incoming"):
    """
    Build a CSR matrix A from an edge list, ready for tropical_spmv.

    edges: iterable of (u, v, w) meaning a directed edge u -> v with
           nonnegative weight w (delay / congestion cost in RRG terms).

    orientation:
      "incoming"  -> A[v, u] = w   (row v lists predecessors; this is what
                     tropical_spmv expects, since d_new[v] = min_u d[u]+w)
      This is the natural layout for relaxation: row = node being updated,
      columns = its in-neighbors.

    Returns a scipy.sparse.csr_matrix of shape (n_nodes, n_nodes), with
    missing entries implicitly +inf (CSR just stores explicit edges; the
    tropical_spmv function treats non-stored entries as +inf, NOT as 0 --
    this is the key difference from ordinary linear-algebra SpMV).
    """
    # IMPORTANT: scipy's COO -> CSR conversion (and coo_matrix construction
    # itself, once duplicates are merged on a later .tocsr()/.sum_duplicates())
    # SUMS duplicate (row, col) entries. That is correct for ordinary linear
    # algebra but WRONG for the min-plus semiring, where a parallel edge
    # should contribute its MINIMUM weight, not the sum of both weights.
    # We therefore dedupe on the raw (row, col, weight) triplets BEFORE
    # constructing any scipy sparse matrix at all.
    raw_rows, raw_cols, raw_vals = [], [], []
    for (u, v, w) in edges:
        if orientation == "incoming":
            raw_rows.append(v)
            raw_cols.append(u)
        else:
            raw_rows.append(u)
            raw_cols.append(v)
        raw_vals.append(w)

    best = {}
    for r, c, v in zip(raw_rows, raw_cols, raw_vals):
        key = (r, c)
        if key not in best or v < best[key]:
            best[key] = v

    if best:
        rows, cols, vals = zip(*[(r, c, v) for (r, c), v in best.items()])
    else:
        rows, cols, vals = [], [], []

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return A