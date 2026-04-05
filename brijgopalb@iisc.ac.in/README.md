# Property-Based Tests for Floyd-Warshall (E0 251o Project)

**Team member:** Brijgopal Bharadwaj (`brijgopalb@iisc.ac.in`)

## Algorithm Under Test

**Floyd-Warshall all-pairs shortest paths** from
`networkx.algorithms.shortest_paths.dense`:

| Function | Returns |
|---|---|
| `nx.floyd_warshall(G)` | `dict[dict]` of distances |
| `nx.floyd_warshall_predecessor_and_distance(G)` | `(predecessors, distances)` dicts |
| `nx.floyd_warshall_numpy(G, nodelist)` | `numpy.ndarray` distance matrix |
| `nx.reconstruct_path(source, target, pred)` | shortest-path node list |

## File Overview

| File | Purpose |
|---|---|
| `test_floyd_warshall.py` | 15 property-based tests with detailed docstrings |
| `graph_strategies.py` | Reusable Hypothesis graph-generation strategies |
| `requirements.txt` | Python dependencies |

## Running the Tests

```bash
pip install -r requirements.txt
pytest test_floyd_warshall.py -v
```

## Properties Tested

| # | Property | Type |
|---|----------|------|
| 1 | Zero self-distance | Invariant |
| 2 | Triangle inequality | Invariant |
| 3 | Symmetry (undirected) | Invariant |
| 4 | Path weight = reported distance | Postcondition |
| 5 | Subpath optimality | Invariant |
| 6 | `floyd_warshall` vs `floyd_warshall_predecessor_and_distance` | Cross-impl |
| 7 | `floyd_warshall` vs `floyd_warshall_numpy` | Cross-impl |
| 8 | Weight scaling | Metamorphic |
| 9 | Edge addition monotonicity | Metamorphic |
| 10 | Subgraph distance lower bound | Metamorphic |
| 11 | Graph reversal transposes distances | Metamorphic |
| 12 | Empty graph distances | Boundary |
| 13 | Disconnected components | Boundary |
| 14 | Negative cycle detection | Boundary |
| 15 | Idempotence / determinism | Determinism |

## Graph Generator Library (`graph_strategies.py`)

The companion module provides reusable Hypothesis strategies for generating
diverse NetworkX graphs.  It is designed to be imported by any test suite:

**Topology strategies** (unweighted structure):
`random_graph_topology`, `complete_graph_topology`, `path_graph_topology`,
`cycle_graph_topology`, `star_graph_topology`, `tree_graph_topology`,
`empty_graph_topology`, `disconnected_graph_topology`, `dag_topology`

**Composite strategies** (topology + weights):
`positive_weighted_digraph`, `nonneg_weighted_digraph`,
`undirected_nonneg_graph`, `dag_with_weights`, `negative_cycle_digraph`,
`disconnected_weighted_digraph`, `positive_weighted_undirected`,
`mixed_topology_weighted_digraph`
