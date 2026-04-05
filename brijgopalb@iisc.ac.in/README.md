# Property-Based Testing: Floyd-Warshall All-Pairs Shortest Paths

**Course:** E0 251o Data Structures & Graph Analytics (2026)
**Team member:** Brijgopal Bharadwaj (`brijgopalb@iisc.ac.in`)

---

## Algorithm Under Test

The **Floyd-Warshall** family from
[`networkx.algorithms.shortest_paths.dense`](https://github.com/networkx/networkx/blob/main/networkx/algorithms/shortest_paths/dense.py)
computes all-pairs shortest-path distances in O(V^3) time.

| Function | Returns | Purpose |
|---|---|---|
| `nx.floyd_warshall(G)` | `dict[dict]` of distances | Convenience wrapper |
| `nx.floyd_warshall_predecessor_and_distance(G)` | `(predecessors, distances)` dicts | Full output with path reconstruction data |
| `nx.floyd_warshall_numpy(G, nodelist)` | `numpy.ndarray` distance matrix | NumPy-based implementation |
| `nx.reconstruct_path(source, target, pred)` | shortest-path node list | Extracts path from predecessor dict |

### Why Floyd-Warshall?

Floyd-Warshall has rich mathematical structure that lends itself to
property-based testing: triangle inequality, subpath optimality (Bellman's
principle), symmetry in undirected graphs, transpose relationships under
edge reversal, and linear scaling under weight multiplication.  It also
has three distinct implementations in NetworkX (dict, dict+predecessors,
numpy) whose outputs can be cross-validated against each other.

---

## Project Structure

```
brijgopalb@iisc.ac.in/
├── graph_strategies.py      # Reusable Hypothesis graph generation library
├── test_floyd_warshall.py   # 15 property-based tests with detailed docstrings
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes __pycache__, .hypothesis/, .pytest_cache/
└── README.md                # This file
```

---

## Running the Tests

```bash
pip install -r requirements.txt
pytest test_floyd_warshall.py -v
```

Tested against **NetworkX >= 3.4**, **Hypothesis >= 6.0**, **NumPy >= 1.24**,
**Python 3.12**.

---

## Graph Generation Library (`graph_strategies.py`)

### Design Decision: Functions over Classes

Hypothesis strategies are first-class objects that compose natively via
`draw()`, `st.one_of()`, and `st.flatmap()`.  Wrapping them in a class
would add indirection without improving composability.  The Hypothesis
documentation, `hypothesis-networkx`, and NetworkX's own test suite all
use the functional `@st.composite` pattern.

### Architecture: Three Composable Layers

```
Layer 1: Topology strategies     (unweighted structure)
Layer 2: Modifier helpers         (weights, self-loops, isolated nodes)
Layer 3: graph_builder()          (composes layer 1 + layer 2 in one call)
```

**Layer 1 -- Topology strategies** produce unweighted graph structure.
Every topology strategy has a uniform signature
`(draw, min_nodes, max_nodes, directed)` so any can be plugged into
`graph_builder`.

| Strategy | Structure |
|---|---|
| `random_graph_topology` | Erdos-Renyi G(n,p) |
| `complete_graph_topology` | Complete graph / digraph |
| `path_graph_topology` | Linear path 0 -> 1 -> ... -> n-1 |
| `cycle_graph_topology` | Single cycle on n nodes |
| `star_graph_topology` | Hub connected to n-1 leaves |
| `tree_graph_topology` | Random labeled tree (Prufer-based) |
| `empty_graph_topology` | n nodes, zero edges |
| `disconnected_graph_topology` | Two disjoint cliques |
| `dag_topology` | Random DAG (edges go low-index to high-index) |
| `bipartite_graph_topology` | Two-partition random bipartite graph |

**Layer 2 -- Modifier helpers** mutate a graph in-place:

| Helper | Effect |
|---|---|
| `_assign_weights` | Independent random integer weight per edge |
| `_assign_uniform_weight` | Same random weight for every edge |
| `_add_self_loops` | Positive-weight self-loops on random node subset |
| `_add_isolated_nodes` | Append 1-3 degree-0 nodes |

**Layer 3 -- `graph_builder()`** is the single composable entry-point:

```python
from graph_strategies import graph_builder, cycle_graph_topology

# Mixed topologies, positive weights (default)
@given(G=graph_builder())

# Specific topology
@given(G=graph_builder(topology=cycle_graph_topology))

# Undirected, non-negative weights
@given(G=graph_builder(directed=False, min_weight=0))

# With structural edge-cases
@given(G=graph_builder(self_loops=True, isolated_nodes=True, uniform_weight=True))
```

**Specialized strategies** that have bespoke construction logic and cannot
be expressed through the generic builder:

| Strategy | Why standalone |
|---|---|
| `dag_with_weights` | Requires DAG edge-ordering invariant (i < j) with negative weights |
| `negative_cycle_digraph` | Constructs a Hamiltonian cycle and forces its total weight negative |

### Edge-Case Coverage Audit

An empirical audit (100 samples per strategy) confirmed the following
coverage:

| Feature | Generated? | Why it matters |
|---|---|---|
| Self-loops | Yes (`self_loops=True`) | Positive self-loop must not change dist(v,v)=0 |
| Isolated nodes | Yes (`isolated_nodes=True`) | Must produce dist(iso, v) = inf for v != iso |
| Uniform weights | Yes (`uniform_weight=True`) | Tests BFS-equivalence degenerate case |
| Zero-weight edges | Yes (`min_weight=0`) | Boundary for non-negative weight assumptions |
| Negative edges (no negative cycle) | Yes (`dag_with_weights`) | DAG guarantees safety |
| Negative cycles | Yes (`negative_cycle_digraph`) | Must produce dist(v,v) < 0 on cycle nodes |
| Disconnected components | Yes (`disconnected_graph_topology`) | Cross-component dist must be inf |
| Empty graph | Yes (`empty_graph_topology`) | Degenerate boundary with 0 edges |
| Bipartite structure | Yes (`bipartite_graph_topology`) | Restricted connectivity patterns |

---

## Properties Tested (15 tests)

### Invariant Properties (Tests 1-5)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 1 | `test_zero_self_distance` | dist(v, v) == 0 for all v (no negative cycles) | `graph_builder()` (mixed topologies) |
| 2 | `test_triangle_inequality` | dist(u, w) <= dist(u, v) + dist(v, w) | `dag_with_weights()` |
| 3 | `test_symmetry_undirected` | dist(u, v) == dist(v, u) in undirected graphs | `graph_builder(directed=False)` |
| 4 | `test_path_weight_equals_distance` | reconstruct_path weight matches reported dist | `graph_builder(topology=random_graph_topology)` |
| 5 | `test_subpath_optimality` | Every sub-path of a shortest path is optimal (Bellman) | `graph_builder(topology=random_graph_topology)` |

### Cross-Implementation Consistency (Tests 6-7)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 6 | `test_fw_dict_vs_pred_dist` | `floyd_warshall` == `floyd_warshall_predecessor_and_distance` | `dag_with_weights()` |
| 7 | `test_fw_dict_vs_numpy` | `floyd_warshall` dict == `floyd_warshall_numpy` matrix | `graph_builder(topology=random_graph_topology)` |

### Metamorphic Properties (Tests 8-11)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 8 | `test_weight_scaling` | Scaling weights by k scales distances by k | `graph_builder(topology=random_graph_topology)` |
| 9 | `test_edge_addition_monotonicity` | Adding a non-negative edge can only decrease distances | `graph_builder(min_weight=0)` |
| 10 | `test_subgraph_distance_lower_bound` | dist_G(u,v) <= dist_H(u,v) for subgraph H of G | `graph_builder(topology=random_graph_topology)` |
| 11 | `test_graph_reversal_transposes_distances` | dist_G(u,v) == dist_{G^R}(v,u) | `dag_with_weights()` |

### Boundary / Edge-Case Properties (Tests 12-14)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 12 | `test_empty_graph_distances` | Zero edges: diagonal=0, off-diagonal=inf | `empty_graph_topology()` |
| 13 | `test_disconnected_components` | Cross-component distances are infinite | `graph_builder(topology=disconnected_graph_topology)` |
| 14 | `test_negative_cycle_detection` | Negative cycle produces dist(u,u) < 0 | `negative_cycle_digraph()` |

### Idempotence / Determinism (Test 15)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 15 | `test_idempotence` | Calling FW twice on same graph gives identical output | `graph_builder(topology=random_graph_topology)` |

---

## Key Design Decisions

### 1. Functional strategies over class-based wrapper

Hypothesis strategies compose natively as functions.  A class would add
indirection without improving composability.  Instead, `graph_builder()`
serves as the single flexible entry-point that any topology, weight
scheme, and modifier can be plugged into.

### 2. Three-layer architecture for graph generation

Separating topology (structure), modifiers (weights/self-loops/isolates),
and the builder (composition) means each layer is independently testable
and reusable.  Adding a new topology or modifier doesn't require changing
existing code.

### 3. DAG-based testing for negative weights

Floyd-Warshall supports negative edge weights provided no negative cycles
exist.  Rather than generating random graphs and filtering for acyclicity
(which wastes test budget), `dag_topology` guarantees acyclicity by
construction: edges only go from lower-index to higher-index nodes.

### 4. Adapted to installed NetworkX behaviour

The installed NetworkX (>= 3.4) does **not** raise `NetworkXUnbounded` for
negative cycles -- instead, `floyd_warshall` returns negative diagonal
entries `dist[v][v] < 0`.  Test 14 checks this actual behaviour rather
than the documented-but-unimplemented exception.  The `floyd_warshall_tree`
function exists only in the unreleased `main` branch and is excluded.

### 5. Cross-implementation consistency as a testing strategy

Tests 6-7 exploit the fact that NetworkX provides three independent
implementations of Floyd-Warshall (dict, dict+predecessors, numpy).
Comparing their outputs is a powerful differential testing technique that
can catch bugs that no single-implementation property test would find.

---

## Documentation Standard

Every test includes a detailed docstring covering:

1. **What property** is being tested (formal statement)
2. **Mathematical basis** -- the theorem or reasoning behind it
3. **Test strategy** -- what graphs are generated and why
4. **Assumptions / preconditions** -- required for the property to hold
5. **Why failure matters** -- what kind of bug a violation would reveal

This follows the documentation structure specified in the project rubric
(E0-251o-Project.md, Section 3).
