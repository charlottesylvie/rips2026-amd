"""
demo.py

Minimal end-to-end usage example: build a graph, run min-plus
Bellman-Ford, inspect distances, and reconstruct a shortest path.

Run with: python3 demo.py
"""

from minplus_algebra import tropical_matrix_from_edges
from minplus_bellman_ford import sssp_min_plus_bellman_ford, reconstruct_path

# A small toy "routing resource graph": nodes are wires/pins, edges are
# programmable interconnect with a delay/congestion weight.
edges = [
    (0, 1, 1.5), (0, 4, 0.7),
    (1, 7, 0.3), (1, 5, 0.4),
    (4, 5, 0.3), (4, 6, 1.1),
    (5, 6, 1.0), (5, 8, 1.2),
    (6, 9, 2.1),
    (7, 6, 2.1),
    (8, 9, 2.5),
]
n_nodes = 10
source = 0

A = tropical_matrix_from_edges(n_nodes, edges)
result = sssp_min_plus_bellman_ford(A, source)

print(f"Distances from node {source}:")
for v in range(n_nodes):
    d = result.dist[v]
    print(f"  node {v}: {'unreachable' if d == float('inf') else f'{d:.2f}'}")

print(f"\nConverged in {result.iterations_used} relaxation rounds "
      f"(each round = 1 tropical sparse matrix-vector multiply)")
print(f"Negative cycle detected: {result.has_negative_cycle}")

target = 9
path = reconstruct_path(result.pred, source, target)
print(f"\nShortest path {source} -> {target}: {path}  (cost {result.dist[target]:.2f})")