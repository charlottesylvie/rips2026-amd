"""
test_minplus_bellman_ford.py

Test plan:
  1. Unit tests on the tropical SpMV primitive itself (small hand-checked
     cases: does min-plus matvec do what the math says it should?).
  2. Correctness tests: min-plus Bellman-Ford vs. classical edge-list
     Bellman-Ford on the SAME random graphs, asserting identical
     distances (this is the most important test -- it's the one that
     proves the "reformulate as linear algebra" step didn't break
     anything).
  3. Cross-check against scipy.sparse.csgraph.bellman_ford (an
     independent, well-tested reference) as a second opinion.
  4. A worked, hand-drawn example resembling an FPGA routing resource
     graph (RRG): a small grid of "wires" with directed, weighted
     interconnect edges, so you can sanity check path reconstruction
     against something you can also check on paper.
  5. Edge cases: disconnected nodes, single node, negative-cycle
     detection, multi-edge (parallel edge, keep-min) handling.
  6. A scaling smoke test (does it run, and run sparsely, on a larger
     sparse random graph -- e.g. 5,000 nodes -- similar in spirit to a
     small FPGA RRG fragment).

Run with: python3 -m pytest test_minplus_bellman_ford.py -v
       or: python3 test_minplus_bellman_ford.py   (runs everything directly)
"""

import numpy as np
import scipy.sparse as sp
import random

from bellman_ford_attempts.Attempt1_June23.minplus_algebra import (
    INF, tropical_spmv, tropical_spmv_dense_row, tropical_matrix_from_edges
)
from bellman_ford_attempts.Attempt1_June23.minplus_bellman_ford import (
    sssp_min_plus_bellman_ford, sssp_classical_bellman_ford, reconstruct_path
)


# ---------------------------------------------------------------------------
# 1. Tropical SpMV primitive sanity checks
# ---------------------------------------------------------------------------

def test_tropical_spmv_basic():
    # Graph: 0 -> 1 (w=2), 0 -> 2 (w=5), 1 -> 2 (w=1)
    edges = [(0, 1, 2.0), (0, 2, 5.0), (1, 2, 1.0)]
    A = tropical_matrix_from_edges(3, edges)  # incoming orientation
    d = np.array([0.0, INF, INF])

    d1 = tropical_spmv(A, d)
    # node 1 reachable from 0 via weight 2 -> d1[1] should be 2
    # node 2 reachable from 0 via weight 5 -> d1[2] should be 5 (1 not yet updated)
    assert d1[0] == INF  # no self-loop / no incoming edge to 0
    assert d1[1] == 2.0
    assert d1[2] == 5.0
    print("test_tropical_spmv_basic: PASS")


def test_tropical_spmv_matches_naive_loop_version():
    rng = np.random.default_rng(42)
    n = 60
    density = 0.08
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < density:
                edges.append((u, v, float(rng.integers(1, 20))))
    A = tropical_matrix_from_edges(n, edges)
    d = rng.uniform(0, 50, size=n)
    d[rng.integers(0, n)] = 0.0  # force at least one finite-ish path source

    out_fast = tropical_spmv(A, d)
    out_naive = tropical_spmv_dense_row(A, d)
    assert np.array_equal(np.isinf(out_fast), np.isinf(out_naive))
    finite_mask = ~np.isinf(out_fast)
    assert np.allclose(out_fast[finite_mask], out_naive[finite_mask])
    print("test_tropical_spmv_matches_naive_loop_version: PASS")


def test_parallel_edges_keep_minimum_weight():
    # Two parallel edges 0 -> 1 with weights 9 and 3; min-plus matrix
    # construction must keep the MIN (3), not sum them (which plain
    # COO->CSR conversion would do by default).
    edges = [(0, 1, 9.0), (0, 1, 3.0)]
    A = tropical_matrix_from_edges(2, edges)
    d = np.array([0.0, INF])
    out = tropical_spmv(A, d)
    assert out[1] == 3.0, f"expected 3.0 (min of parallel edges), got {out[1]}"
    print("test_parallel_edges_keep_minimum_weight: PASS")


# ---------------------------------------------------------------------------
# 2. Min-plus Bellman-Ford vs classical Bellman-Ford on random graphs
# ---------------------------------------------------------------------------

def _random_dag_or_graph(n, n_edges, seed, allow_cycles=True, nonneg=True):
    rng = random.Random(seed)
    edges = []
    seen = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 20:
        attempts += 1
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v or (u, v) in seen:
            continue
        if not allow_cycles and v <= u:
            continue
        w = rng.uniform(0.5, 20.0) if nonneg else rng.uniform(-5.0, 20.0)
        edges.append((u, v, w))
        seen.add((u, v))
    return edges


def test_matches_classical_bellman_ford_random_graphs():
    n_trials = 15
    for trial in range(n_trials):
        n = random.Random(trial).randint(10, 80)
        n_edges = int(n * random.Random(trial + 1).uniform(2, 6))
        edges = _random_dag_or_graph(n, n_edges, seed=trial, allow_cycles=True, nonneg=True)
        source = 0

        A = tropical_matrix_from_edges(n, edges)
        result_tropical = sssp_min_plus_bellman_ford(A, source)
        result_classical = sssp_classical_bellman_ford(n, edges, source)

        finite_t = ~np.isinf(result_tropical.dist)
        finite_c = ~np.isinf(result_classical.dist)
        assert np.array_equal(finite_t, finite_c), (
            f"[trial {trial}] reachability mismatch between tropical and classical BF"
        )
        assert np.allclose(result_tropical.dist[finite_t], result_classical.dist[finite_c]), (
            f"[trial {trial}] distance mismatch:\n"
            f"tropical: {result_tropical.dist}\nclassical: {result_classical.dist}"
        )
    print(f"test_matches_classical_bellman_ford_random_graphs: PASS ({n_trials} random graphs)")


# ---------------------------------------------------------------------------
# 3. Cross-check against scipy's independent Bellman-Ford implementation
# ---------------------------------------------------------------------------

def test_matches_scipy_bellman_ford_reference():
    from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford

    n = 40
    edges = _random_dag_or_graph(n, n * 4, seed=7, allow_cycles=True, nonneg=True)
    source = 0

    A = tropical_matrix_from_edges(n, edges)
    result = sssp_min_plus_bellman_ford(A, source)

    # Build scipy's expected "outgoing" weighted adjacency (row=u, col=v).
    rows = [u for (u, v, w) in edges]
    cols = [v for (u, v, w) in edges]
    vals = [w for (u, v, w) in edges]
    A_scipy = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    dist_scipy = scipy_bellman_ford(A_scipy, directed=True, indices=source)

    finite_mine = ~np.isinf(result.dist)
    finite_scipy = ~np.isinf(dist_scipy)
    assert np.array_equal(finite_mine, finite_scipy), "reachability mismatch vs scipy reference"
    assert np.allclose(result.dist[finite_mine], dist_scipy[finite_scipy]), (
        f"distance mismatch vs scipy:\nmine:  {result.dist}\nscipy: {dist_scipy}"
    )
    print("test_matches_scipy_bellman_ford_reference: PASS")


# ---------------------------------------------------------------------------
# 4. Worked FPGA-routing-resource-graph-style example with path reconstruction
# ---------------------------------------------------------------------------

def test_small_rrg_style_example_and_path_reconstruction():
    """
    A small hand-built graph in the spirit of the abstract figure in the
    project description: nodes 0..9 represent wire segments / pins,
    directed weighted edges represent programmable interconnect with a
    delay cost. We hand-verify the shortest path from node 0 to node 9.

    Graph (mirroring the kind of structure in the project figure):
        0 -> 1 (1.5)
        0 -> 4 (0.7)
        1 -> 7 (0.3)
        1 -> 5 (0.4)
        4 -> 5 (0.3)
        4 -> 6 (1.1)
        5 -> 6 (1.0)
        5 -> 8 (1.2)
        6 -> 9 (2.1)
        7 -> 6 (2.1)
        8 -> 9 (2.5)

    By hand: shortest path 0 -> 9
        0->4->5->6->9 = 0.7+0.3+1.0+2.1 = 4.1
        0->4->6->9    = 0.7+1.1+2.1     = 3.9   <-- best so far
        0->1->5->6->9 = 1.5+0.4+1.0+2.1 = 5.0
        0->4->5->8->9 = 0.7+0.3+1.2+2.5 = 4.7
    So expected shortest distance to node 9 is 3.9 via 0 -> 4 -> 6 -> 9.
    """
    edges = [
        (0, 1, 1.5), (0, 4, 0.7),
        (1, 7, 0.3), (1, 5, 0.4),
        (4, 5, 0.3), (4, 6, 1.1),
        (5, 6, 1.0), (5, 8, 1.2),
        (6, 9, 2.1),
        (7, 6, 2.1),
        (8, 9, 2.5),
    ]
    n = 10
    A = tropical_matrix_from_edges(n, edges)
    result = sssp_min_plus_bellman_ford(A, source=0)

    assert np.isclose(result.dist[9], 3.9), f"expected 3.9, got {result.dist[9]}"

    path = reconstruct_path(result.pred, source=0, target=9)
    assert path == [0, 4, 6, 9], f"expected [0,4,6,9], got {path}"
    print(f"test_small_rrg_style_example_and_path_reconstruction: PASS "
          f"(dist={result.dist[9]}, path={path})")


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_disconnected_node_stays_infinite():
    edges = [(0, 1, 1.0)]
    A = tropical_matrix_from_edges(3, edges)  # node 2 has no edges at all
    result = sssp_min_plus_bellman_ford(A, source=0)
    assert np.isinf(result.dist[2])
    print("test_disconnected_node_stays_infinite: PASS")


def test_single_node_graph():
    A = tropical_matrix_from_edges(1, [])
    result = sssp_min_plus_bellman_ford(A, source=0)
    assert result.dist[0] == 0.0
    print("test_single_node_graph: PASS")


def test_negative_cycle_detected():
    # 0 -> 1 (1), 1 -> 2 (-5), 2 -> 1 (1)  => cycle 1->2->1 has weight -4
    edges = [(0, 1, 1.0), (1, 2, -5.0), (2, 1, 1.0)]
    A = tropical_matrix_from_edges(3, edges)
    result = sssp_min_plus_bellman_ford(A, source=0, max_iters=3)
    assert result.has_negative_cycle, "expected negative cycle to be detected"
    print("test_negative_cycle_detected: PASS")


def test_zero_weight_self_consistent():
    # A direct edge should never be beaten by a longer detour when
    # the direct edge is already optimal.
    edges = [(0, 1, 0.0), (0, 2, 5.0), (1, 2, 5.0)]
    A = tropical_matrix_from_edges(3, edges)
    result = sssp_min_plus_bellman_ford(A, source=0)
    assert result.dist[1] == 0.0
    assert result.dist[2] == 5.0  # both 0->2 directly and 0->1->2 tie at 5
    print("test_zero_weight_self_consistent: PASS")


# ---------------------------------------------------------------------------
# 6. Scaling smoke test (sparse, larger graph)
# ---------------------------------------------------------------------------

def test_scaling_smoke_test_5000_nodes():
    import time
    n = 5000
    rng = np.random.default_rng(0)
    avg_degree = 6
    n_edges = n * avg_degree

    us = rng.integers(0, n, size=n_edges)
    vs = rng.integers(0, n, size=n_edges)
    ws = rng.uniform(0.1, 10.0, size=n_edges)
    mask = us != vs
    edges = list(zip(us[mask].tolist(), vs[mask].tolist(), ws[mask].tolist()))

    A = tropical_matrix_from_edges(n, edges)

    t0 = time.time()
    result = sssp_min_plus_bellman_ford(A, source=0, max_iters=40)
    t1 = time.time()

    n_reachable = int(np.sum(~np.isinf(result.dist)))
    print(f"test_scaling_smoke_test_5000_nodes: PASS "
          f"(n={n}, edges={len(edges)}, time={t1 - t0:.3f}s, "
          f"reachable={n_reachable}/{n}, iters_used={result.iterations_used})")
    assert n_reachable > n * 0.5  # sanity: most of a random sparse graph should be reachable


# ---------------------------------------------------------------------------
# Runner (so this also works without pytest installed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_tropical_spmv_basic,
        test_tropical_spmv_matches_naive_loop_version,
        test_parallel_edges_keep_minimum_weight,
        test_matches_classical_bellman_ford_random_graphs,
        test_matches_scipy_bellman_ford_reference,
        test_small_rrg_style_example_and_path_reconstruction,
        test_disconnected_node_stays_infinite,
        test_single_node_graph,
        test_negative_cycle_detected,
        test_zero_weight_self_consistent,
        test_scaling_smoke_test_5000_nodes,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__}: FAIL -- {e}")
    print()
    if failed == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
    else:
        print(f"{failed} / {len(tests)} TESTS FAILED")