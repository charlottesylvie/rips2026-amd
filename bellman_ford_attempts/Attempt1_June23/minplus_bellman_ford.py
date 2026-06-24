"""
minplus_bellman_ford.py

Bellman-Ford single-source shortest path (SSSP) expressed as repeated
tropical (min-plus) sparse matrix-vector relaxation.

Classical Bellman-Ford:
    for i in range(n-1):
        for each edge (u, v, w):
            if d[u] + w < d[v]:
                d[v] = d[u] + w

Min-plus reformulation (this file):
    for i in range(n-1):
        d = minimum( d, A (x) d )      # one tropical SpMV per iteration

This is exactly the "Bellman-Ford via min-plus matrix powers" idea from
Kepner & Gilbert's "Graph Algorithms in the Language of Linear Algebra"
and matches the relaxation structure the AMD project description asks
to map onto GPU sparse kernels. Each outer iteration parallelizes over
ALL nodes / edges at once (it's a single SpMV) rather than processing
edges sequentially -- this is exactly the parallelism profile that
makes the problem GPU-friendly.

Includes:
  - sssp_min_plus_bellman_ford(): the GPU-shaped algorithm (dense/sparse
    matrix-vector relaxation), run on CPU with NumPy/SciPy for now.
  - sssp_classical_bellman_ford(): textbook reference implementation,
    used ONLY to check correctness against the tropical version.
  - negative cycle detection (relevant for general graphs; FPGA RRGs
    have nonnegative weights, but it's good practice / good science to
    keep the check).
"""

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from minplus_algebra import INF, tropical_spmv, tropical_matrix_from_edges


@dataclass
class SSSPResult:
    dist: np.ndarray          # shortest distance from source to each node
    pred: np.ndarray          # predecessor array, for path reconstruction
    iterations_used: int      # number of relaxation rounds actually run
    converged: bool           # True if it stabilized before the n-1 cap
    has_negative_cycle: bool


def sssp_min_plus_bellman_ford(A: sp.csr_matrix, source: int,
                                max_iters: int | None = None,
                                track_pred: bool = True) -> SSSPResult:
    """
    Min-plus Bellman-Ford SSSP.

    A: CSR adjacency matrix where A[v, u] = w(u -> v)  (row = destination,
       "incoming" orientation -- see tropical_algebra.tropical_matrix_from_edges).
    source: source node index.
    max_iters: defaults to n-1 (the standard Bellman-Ford bound, since
       a shortest simple path visits at most n-1 edges).

    Returns SSSPResult. Detects negative cycles via one extra relaxation
    round past convergence (standard technique).
    """
    n = A.shape[0]
    if max_iters is None:
        max_iters = n - 1

    d = np.full(n, INF, dtype=np.float64)
    d[source] = 0.0
    pred = np.full(n, -1, dtype=np.int64)

    converged = False
    it_used = 0
    for it in range(max_iters):
        it_used = it + 1
        d_relaxed = tropical_spmv(A, d)          # <-- the GPU-shaped kernel call
        d_new = np.minimum(d, d_relaxed)

        if track_pred:
            _update_predecessors(A, d, d_new, pred)

        if np.array_equal(d_new, d) or np.all(
            np.isclose(d_new, d, equal_nan=False) | (np.isinf(d_new) & np.isinf(d))
        ):
            d = d_new
            converged = True
            break
        d = d_new

    # One more relaxation round to check for negative cycles.
    has_negative_cycle = False
    d_check = tropical_spmv(A, d)
    d_check = np.minimum(d, d_check)
    if not np.all(
        np.isclose(d_check, d, equal_nan=False) | (np.isinf(d_check) & np.isinf(d))
    ):
        has_negative_cycle = True

    return SSSPResult(dist=d, pred=pred, iterations_used=it_used,
                       converged=converged, has_negative_cycle=has_negative_cycle) 


def _update_predecessors(A: sp.csr_matrix, d_old: np.ndarray,
                          d_new: np.ndarray, pred: np.ndarray) -> None:
    """
    Recompute predecessor pointers for any node whose distance improved
    this round. This is a small, separate post-pass (not part of the
    tropical SpMV itself) -- on a real GPU kernel you'd fuse an argmin
    alongside the min, but keeping it separate here keeps the core
    algebra function clean and easy to verify.
    """
    n = A.shape[0]
    indptr, indices, data = A.indptr, A.indices, A.data
    improved = np.flatnonzero(~np.isclose(d_new, d_old) & ~(np.isinf(d_new) & np.isinf(d_old)))
    for v in improved:
        start, end = indptr[v], indptr[v + 1]
        if start == end:
            continue
        us = indices[start:end]
        ws = data[start:end]
        cand = d_old[us] + ws
        best_local = np.argmin(cand)
        if np.isclose(cand[best_local], d_new[v]):
            pred[v] = us[best_local]


def reconstruct_path(pred: np.ndarray, source: int, target: int) -> list[int] | None:
    """Walk predecessor pointers back from target to source."""
    if pred[target] == -1 and target != source:
        return None
    path = [target]
    cur = target
    visited = {target}
    while cur != source:
        cur = pred[cur]
        if cur == -1 or cur in visited:
            return None  # disconnected or corrupted pred chain
        visited.add(cur)
        path.append(cur)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Reference implementation, used ONLY for cross-checking correctness.
# ---------------------------------------------------------------------------

def sssp_classical_bellman_ford(n_nodes: int, edges, source: int) -> SSSPResult:
    """
    Textbook edge-list Bellman-Ford (O(V*E)), no algebra tricks.
    edges: list of (u, v, w) directed edges u -> v with weight w.
    """
    d = np.full(n_nodes, INF, dtype=np.float64)
    pred = np.full(n_nodes, -1, dtype=np.int64)
    d[source] = 0.0

    it_used = 0
    converged = False
    for it in range(n_nodes - 1):
        it_used = it + 1
        changed = False
        for (u, v, w) in edges:
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                pred[v] = u
                changed = True
        if not changed:
            converged = True
            break

    has_negative_cycle = False
    for (u, v, w) in edges:
        if d[u] + w < d[v]:
            has_negative_cycle = True
            break

    return SSSPResult(dist=d, pred=pred, iterations_used=it_used,
                       converged=converged, has_negative_cycle=has_negative_cycle)