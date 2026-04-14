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

---

## Example Graphs

Concrete worked examples for key test properties.  Each example shows a
small specific graph, the exact Floyd-Warshall output, and the test that
verifies the property.

> **Color key (all diagrams)**
> - 🔵 Blue — general directed graph nodes
> - 🟢 Green — source node / Component A
> - 🟡 Amber — sink / target / Component B
> - 🟣 Violet — reversed graph nodes
> - 🔴 Red — negative-cycle nodes

---

### Example 1 — Zero Self-Distance and Triangle Inequality (Tests 1, 2)

The 2-hop path 1 → 2 → 3 costs 5 + 1 = 6 but the direct edge 1 → 3 costs 2,
so Floyd-Warshall must choose the direct edge.

```mermaid
flowchart LR
    classDef blue fill:#3b82f6,stroke:#1d4ed8,color:#fff
    n0((0)) -->|3| n1((1))
    n1 -->|5| n2((2))
    n1 -->|2| n3((3))
    n2 -->|1| n3
    class n0,n1,n2,n3 blue
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=3)
G.add_edge(1, 2, weight=5)
G.add_edge(1, 3, weight=2)
G.add_edge(2, 3, weight=1)   # 2-hop: 1→2→3 costs 6 — direct 1→3 wins at 2

dist = nx.floyd_warshall(G)
# dist[0][0] = 0             ← Test 1: zero self-distance
# dist[1][3] = 2             ← direct edge beats 2-hop path
# dist[0][3] = 5             ← 0→1(3) + 1→3(2)
#
# Triangle inequality check (Test 2):
#   dist[0][3]  ≤  dist[0][1] + dist[1][3]
#       5       ≤      3      +     2       ✓
```

**Why this matters for Test 2:** `dag_with_weights` generates DAGs with
mixed-sign weights.  A triangle-inequality violation would mean the
algorithm reported a path from 0 to 3 longer than the sum of its two
sub-legs — a logical impossibility for any correct shortest-path algorithm.

---

### Example 2 — Path Reconstruction Consistency (Test 4)

The predecessor dictionary (`pred`) and the distance dictionary (`dist`)
are computed together.  The weight of any reconstructed path must exactly
match the reported distance.

```mermaid
flowchart LR
    classDef src fill:#10b981,stroke:#059669,color:#fff
    classDef mid fill:#3b82f6,stroke:#1d4ed8,color:#fff
    classDef snk fill:#f59e0b,stroke:#d97706,color:#fff
    n0((0)) -->|4| n1((1))
    n1 -->|1| n2((2))
    n2 -->|7| n3((3))
    class n0 src
    class n1,n2 mid
    class n3 snk
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=4)
G.add_edge(1, 2, weight=1)
G.add_edge(2, 3, weight=7)

pred, dist = nx.floyd_warshall_predecessor_and_distance(G)
path = nx.reconstruct_path(0, 3, pred)   # → [0, 1, 2, 3]

# Sum of edge weights along the reconstructed path:
path_weight = 4 + 1 + 7   # = 12
# dist[0][3]  =  12  ✓  (Test 4 asserts path_weight == dist[0][3])
```

**Why this matters:** `pred` and `dist` could theoretically be
built by code paths that diverge.  A silent off-by-one in either would
make the reconstructed path claim a different cost than the reported
distance, breaking any consumer of `reconstruct_path`.

---

### Example 3 — Symmetry on Undirected Graphs (Test 3)

In an undirected graph every edge is bidirectional, so
`dist(u, v)` must always equal `dist(v, u)`.

```mermaid
flowchart LR
    classDef violet fill:#8b5cf6,stroke:#6d28d9,color:#fff
    n0((0)) ---|3| n1((1))
    n1 ---|5| n2((2))
    n1 ---|4| n3((3))
    n2 ---|2| n3
    class n0,n1,n2,n3 violet
```

```python
G = nx.Graph()
G.add_edge(0, 1, weight=3)
G.add_edge(1, 2, weight=5)
G.add_edge(1, 3, weight=4)
G.add_edge(2, 3, weight=2)

dist = nx.floyd_warshall(G)
# Shortest 0 → 2:  0─1(3) + 1─2(5) = 8   (not via 3: 3+4+2=9)
# dist[0][2] = 8,  dist[2][0] = 8   ← symmetry holds (Test 3) ✓
# dist[0][3] = 7,  dist[3][0] = 7   ← symmetry holds           ✓
```

**Why this matters:** If Floyd-Warshall initialises only one direction when
converting `nx.Graph` edges to its internal distance table, the symmetry
would break — distances would be finite in one direction and infinite in
the other on the same undirected edge.

---

### Example 4 — Weight Scaling is Linear (Test 10)

Multiplying every edge weight by *k* must multiply every finite distance by
*k*, because the **same** path remains optimal and its cost scales linearly.

```mermaid
flowchart LR
    subgraph orig["Original  ·  k = 1  ·  dist[0][2] = 4"]
        a0((0)) -->|2| a1((1))
        a1 -->|5| a2((2))
        a0 -->|3| a3((3))
        a3 -->|1| a2
    end
    subgraph scld["Scaled  ·  k = 3  ·  dist[0][2] = 12 = 3 × 4"]
        b0((0)) -->|6| b1((1))
        b1 -->|15| b2((2))
        b0 -->|9| b3((3))
        b3 -->|3| b2
    end
    classDef blue  fill:#3b82f6,stroke:#1d4ed8,color:#fff
    classDef green fill:#10b981,stroke:#059669,color:#fff
    class a0,a1,a2,a3 blue
    class b0,b1,b2,b3 green
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=2); G.add_edge(1, 2, weight=5)
G.add_edge(0, 3, weight=3); G.add_edge(3, 2, weight=1)
# Shortest 0→2: via node 3  →  3 + 1 = 4  (beats 2 + 5 = 7)

k = 3
G_scaled = G.copy()
for u, v in G_scaled.edges():
    G_scaled[u][v]["weight"] *= k
# Shortest 0→2 in scaled: via node 3  →  9 + 3 = 12 = 3 × 4  ✓

# Test 10 asserts: dist_scaled[u][v] == k * dist_orig[u][v]  for all finite pairs
```

---

### Example 5 — Graph Reversal Transposes the Distance Matrix (Test 13)

Flipping every edge direction creates a bijection between u→v paths in G
and v→u paths in G^R, so `dist_G(u,v) == dist_GR(v,u)`.

```mermaid
flowchart LR
    subgraph GG["Graph G"]
        g0((0)) -->|2| g1((1))
        g1 -->|3| g2((2))
        g1 -->|1| g3((3))
    end
    subgraph GR["Reversed G^R"]
        r2((2)) -->|3| r1((1))
        r3((3)) -->|1| r1
        r1 -->|2| r0((0))
    end
    classDef blue   fill:#3b82f6,stroke:#1d4ed8,color:#fff
    classDef violet fill:#8b5cf6,stroke:#6d28d9,color:#fff
    class g0,g1,g2,g3 blue
    class r0,r1,r2,r3 violet
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=2)
G.add_edge(1, 2, weight=3)
G.add_edge(1, 3, weight=1)

dist_G  = nx.floyd_warshall(G)
dist_GR = nx.floyd_warshall(G.reverse())

# dist_G[0][2]  = 5    dist_GR[2][0]  = 5  ← transpose equality (Test 13) ✓
# dist_G[0][3]  = 3    dist_GR[3][0]  = 3  ✓
# dist_G[2][0]  = inf  dist_GR[0][2]  = inf  ✓ (no return path in either direction)
```

---

### Example 6 — Infinite Cross-Component Distances (Test 17)

With no edges between the two components, any cross-component distance
must be infinite — Floyd-Warshall should never fabricate a path.

```mermaid
flowchart LR
    subgraph A["Component A"]
        a0((0)) -->|5| a1((1))
    end
    subgraph B["Component B"]
        b2((2)) -->|4| b3((3))
    end
    classDef green fill:#10b981,stroke:#059669,color:#fff
    classDef amber fill:#f59e0b,stroke:#d97706,color:#fff
    class a0,a1 green
    class b2,b3 amber
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=5)
G.add_edge(2, 3, weight=4)
# No edges cross the component boundary

dist = nx.floyd_warshall(G)
# dist[0][2] = inf  ← no cross-component path (Test 17) ✓
# dist[1][3] = inf  ✓
# dist[0][1] = 5    ← within-component distance unaffected ✓
```

---

### Example 7 — Negative Cycle Produces Negative Self-Distance (Test 18)

The three-node cycle has total weight 2 + 1 − 6 = **−3**.
Floyd-Warshall computes shortest *walks*, so traversing the cycle once
drives `dist(u, u)` below zero for every node on it.

```mermaid
flowchart LR
    classDef red fill:#ef4444,stroke:#b91c1c,color:#fff
    n0((0)) -->|"+2"| n1((1))
    n1 -->|"+1"| n2((2))
    n2 -->|"−6"| n0
    class n0,n1,n2 red
```

```python
G = nx.DiGraph()
G.add_edge(0, 1, weight=2)
G.add_edge(1, 2, weight=1)
G.add_edge(2, 0, weight=-6)   # cycle total: 2 + 1 − 6 = −3

dist = nx.floyd_warshall(G)
# dist[0][0] = -3,  dist[1][1] = -3,  dist[2][2] = -3
# ← all cycle nodes get a negative self-distance (Test 18) ✓
```

**What Test 18 detects:** A negative diagonal entry is the observable
signal that a negative cycle exists.  Test 18 uses `negative_cycle_digraph`
(which forces a Hamiltonian cycle with negative total weight) and asserts
that at least one `dist[v][v] < 0`.  This is distinct from the *bug
discovery* test, which documents that Floyd-Warshall does **not** raise
an exception the way Bellman-Ford does.

---

## Project Structure

```
brijgopalb@iisc.ac.in/
├── test_floyd_warshall.py   # Single self-contained file: strategies + 20 property tests + bug discovery
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes __pycache__, .hypothesis/, .pytest_cache/
└── README.md                # This file
```

The project follows the rubric requirement of a **single Python file**
containing all imports, graph-generation strategies, helper functions,
and property-based tests with detailed docstrings.

---

## Running the Tests

```bash
pip install -r requirements.txt
pytest test_floyd_warshall.py -v
```

To view Hypothesis statistics (showing `event()` and `target()` output):

```bash
pytest test_floyd_warshall.py -v --hypothesis-show-statistics
```

Tested against **NetworkX 3.6.1**, **Hypothesis >= 6.0**, **NumPy >= 1.24**,
**Python 3.12**.

---

## Graph Generation Library

### Design Decision: Functions over Classes

Hypothesis strategies are first-class objects that compose natively via
`draw()`, `st.one_of()`, and `st.flatmap()`.  Wrapping them in a class
would add indirection without improving composability.  The Hypothesis
documentation, `hypothesis-networkx`, and NetworkX's own test suite all
use the functional `@st.composite` pattern.

### Architecture: Three Composable Layers

```mermaid
flowchart LR
    classDef topo    fill:#1a1a2e,stroke:#e94560,color:#eaeaea,rx:6
    classDef mod     fill:#16213e,stroke:#0f3460,color:#eaeaea,rx:6
    classDef builder fill:#0f3460,stroke:#e94560,color:#ffffff,font-weight:bold
    classDef output  fill:#e94560,stroke:#c73652,color:#ffffff,font-weight:bold,rx:20

    subgraph L1["🗺️  Layer 1 · Topology  (unweighted structure)"]
        direction TB
        t1("🎲 random_graph")
        t2("🔗 complete_graph")
        t3("➡️  path_graph")
        t4("🔄 cycle_graph")
        t5("⭐ star_graph")
        t6("🌲 tree_graph")
        t7("◯  empty_graph")
        t8("✂️  disconnected")
        t9("⬇️  dag")
        t10("⬡  bipartite")
    end

    subgraph L2["🔧  Layer 2 · Modifiers  (mutate in-place)"]
        direction TB
        m1("⚖️  assign_weights")
        m2("🪄 uniform_weight")
        m3("🔁 add_self_loops")
        m4("🏝️  add_isolated_nodes")
    end

    subgraph L3["⚙️  Layer 3 · Composition"]
        gb["graph_builder()"]
    end

    L1 -- "topology=" --> gb
    L2 -- "flags" --> gb
    gb -- "yields" --> out(["📦 nx.Graph / DiGraph"])

    class t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 topo
    class m1,m2,m3,m4 mod
    class gb builder
    class out output
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
| `bipartite_graph_topology` | Two-partition random bipartite graph — used exclusively by Test 24 (parity property); intentionally excluded from `ALL_TOPOLOGIES` |

**Layer 2 -- Modifier helpers** mutate a graph in-place:

| Helper | Effect |
|---|---|
| `_assign_weights` | Independent random integer weight per edge |
| `_assign_uniform_weight` | Same random weight for every edge |
| `_add_self_loops` | Positive-weight self-loops on random node subset |
| `_add_isolated_nodes` | Append 1-3 degree-0 nodes |

**Layer 3 -- `graph_builder()`** is the single composable entry-point:

```python
from test_floyd_warshall import graph_builder, cycle_graph_topology

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

## Properties Tested (20 tests + 1 bug discovery)

```mermaid
mindmap
  root((20 Property Tests))
    Invariant 1–5
      test_zero_self_distance
      test_triangle_inequality
      test_symmetry_undirected
      test_path_weight_equals_distance
      test_subpath_optimality
    Cross-Implementation 6–7
      test_fw_dict_vs_pred_dist
      test_fw_dict_vs_numpy
    Cross-Algorithm 8–9
      test_fw_vs_dijkstra
      test_fw_vs_bellman_ford
    Metamorphic 10–14
      test_weight_scaling
      test_edge_addition_monotonicity
      test_subgraph_distance_lower_bound
      test_graph_reversal_transposes_distances
      test_node_addition_invariance
    Boundary 15–18
      test_single_node_self_distance
      test_empty_graph_distances
      test_disconnected_components
      test_negative_cycle_detection
    Postcondition 19
      test_complete_graph_uniform_weight
    Idempotence 20
      test_idempotence
```

### Invariant Properties (Tests 1-5)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 1 | `test_zero_self_distance` | dist(v, v) == 0 for all v (no negative cycles) | `graph_builder()` + `@example` with self-loop graph |
| 2 | `test_triangle_inequality` | dist(u, w) <= dist(u, v) + dist(v, w) | `dag_with_weights()` + `target()` |
| 3 | `test_symmetry_undirected` | dist(u, v) == dist(v, u) in undirected graphs | `graph_builder(directed=False)` |
| 4 | `test_path_weight_equals_distance` | reconstruct_path weight matches reported dist | `graph_builder(topology=random_graph_topology)` |
| 5 | `test_subpath_optimality` | Every sub-path of a shortest path is optimal (Bellman) | `graph_builder(topology=random_graph_topology)` |

### Cross-Implementation Consistency (Tests 6-7)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 6 | `test_fw_dict_vs_pred_dist` | `floyd_warshall` == `floyd_warshall_predecessor_and_distance` | `dag_with_weights()` |
| 7 | `test_fw_dict_vs_numpy` | `floyd_warshall` dict == `floyd_warshall_numpy` matrix | `graph_builder(topology=random_graph_topology)` |

### Cross-Algorithm Validation (Tests 8-9)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 8 | `test_fw_vs_dijkstra` | FW distances match Dijkstra (non-negative weights) | `graph_builder(topology=random_graph_topology)` + `event()` |
| 9 | `test_fw_vs_bellman_ford` | FW distances match Bellman-Ford (negative weights, no neg cycles) | `dag_with_weights()` |

### Metamorphic Properties (Tests 10-14)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 10 | `test_weight_scaling` | Scaling weights by k scales distances by k | `graph_builder(topology=random_graph_topology)` |
| 11 | `test_edge_addition_monotonicity` | Adding a non-negative edge can only decrease distances | `graph_builder(min_weight=0)` + `data.draw()` |
| 12 | `test_subgraph_distance_lower_bound` | dist_G(u,v) <= dist_H(u,v) for subgraph H of G | `graph_builder(topology=random_graph_topology)` + `data.draw()` |
| 13 | `test_graph_reversal_transposes_distances` | dist_G(u,v) == dist_{G^R}(v,u) | `dag_with_weights()` |
| 14 | `test_node_addition_invariance` | Adding an isolated node preserves all existing distances | `graph_builder(topology=random_graph_topology)` |

### Boundary / Edge-Case Properties (Tests 15-18)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 15 | `test_single_node_self_distance` | Single node: dist(v,v) = 0 | Parametric on `node_id` |
| 16 | `test_empty_graph_distances` | Zero edges: diagonal=0, off-diagonal=inf | `empty_graph_topology()` |
| 17 | `test_disconnected_components` | Cross-component distances are infinite | `graph_builder(topology=disconnected_graph_topology)` |
| 18 | `test_negative_cycle_detection` | Negative cycle produces dist(u,u) < 0 | `negative_cycle_digraph()` |

### Postcondition Properties (Tests 19–23)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 19 | `test_complete_graph_uniform_weight` | Complete digraph, uniform weight w: dist(u,v) = w for all u≠v | Parametric on `n`, `w` |
| 21 | `test_path_graph_exact_distances` | Directed path: dist(i,j) = prefix-sum of weights; dist(i,j)=inf for i>j | Parametric on `weights` list |
| 22 | `test_star_graph_exact_distances` | Directed star: dist(hub,leaf)=spoke weight; all other non-self distances = inf | Parametric on `spoke_weights` list |
| 23 | `test_single_edge_exact_distances` | Single edge (u,v,w): dist(u,v)=w; dist(v,u)=inf; all others = inf | Parametric on `u`, `v`, `w` |
| 24 | `test_bipartite_parity_of_distances` | Connected bipartite + unit weights: dist(u,v) even iff same partition | `bipartite_graph_topology(directed=False)` |

### Idempotence / Determinism (Test 20)

| # | Test | Property | Generator |
|---|------|----------|-----------|
| 20 | `test_idempotence` | Calling FW twice on same graph gives identical output | `graph_builder(topology=random_graph_topology)` |

### Bug Discovery

| Test | Finding |
|------|---------|
| `test_negative_cycle_silent_failure` | Floyd-Warshall silently returns invalid results on negative cycles, unlike Bellman-Ford and Johnson which raise `NetworkXUnbounded` |

---

## Bug Discovery: Silent Failure on Negative Cycles

### Summary

Floyd-Warshall silently returns invalid distance values when the input
graph contains a negative-weight cycle.  In contrast, NetworkX's other
shortest-path algorithms (`single_source_bellman_ford`,
`johnson`) raise `NetworkXUnbounded` on the same input.

```mermaid
sequenceDiagram
    participant U as User Code
    participant FW as floyd_warshall
    participant BF as single_source_bellman_ford
    participant J  as johnson

    Note over U: Graph G with negative cycle<br/>0→1(+1), 1→2(+1), 2→0(−5)

    U->>FW: floyd_warshall(G)
    FW-->>U: ⚠️ Returns invalid distances silently<br/>{0:{0:−3, 1:−2, …}, …}

    U->>BF: single_source_bellman_ford_path_length(G, 0)
    BF-->>U: ✅ Raises NetworkXUnbounded

    U->>J: johnson(G)
    J-->>U: ✅ Raises NetworkXUnbounded
```

### Minimal Reproducer

```python
import networkx as nx

G = nx.DiGraph()
G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, -5)])

# Bellman-Ford correctly detects the negative cycle:
nx.single_source_bellman_ford_path_length(G, 0)
# → raises NetworkXUnbounded("Negative cycle detected.")

# Johnson correctly detects the negative cycle:
nx.johnson(G)
# → raises NetworkXUnbounded("Negative cycle detected.")

# Floyd-Warshall silently returns meaningless distances:
dist = nx.floyd_warshall(G)
# → {0: {0: -3, 1: -2, 2: -1}, 1: {0: -4, 1: -3, 2: -2}, 2: {0: -8, 1: -7, 2: -6}}
# No exception, no warning.
```

### Root Cause

The absence of a negative-cycle check is visible at a precise location in
the NetworkX source:

- **File:** `networkx/algorithms/shortest_paths/dense.py`
- **Function:** `floyd_warshall_predecessor_and_distance` — defined at **line 90**
- **Relaxation loop:** **lines 160–168** (triple-nested DP)
- **Return statement:** **line 169** — returns immediately after the loop

The full relaxation loop that ends without any diagnostic:

```python
# dense.py  lines 160–169
for w in G:           # outer loop over intermediate vertices
    dist_w = dist[w]
    for u in G:
        dist_u = dist[u]
        for v in G:
            d = dist_u[w] + dist_w[v]
            if dist_u[v] > d:
                dist_u[v] = d
                pred[u][v] = pred[w][v]
return dict(pred), dict(dist)   # ← returns with no diagonal check
```

The docstring at **line 131** says _"This algorithm can still fail if there
are negative cycles"_ but never specifies what "fail" means — no exception,
no `warnings.warn`, no documentation of the negative-diagonal signal.

### Suggested Fix

A single O(n) post-loop diagonal scan, inserted after line 168:

```python
# Proposed addition after line 168:
if any(dist[v][v] < 0 for v in G):
    raise nx.NetworkXUnbounded(
        "Negative cycle detected in floyd_warshall."
    )
```

This is negligible overhead vs the O(n³) main loop and exactly mirrors the
check already present in `networkx/algorithms/shortest_paths/weighted.py`
for Bellman-Ford and Johnson's algorithm.

### Impact

A user migrating from Bellman-Ford to Floyd-Warshall silently loses
negative-cycle protection.  The returned distances look structurally valid
(same `dict[dict]` format) but contain meaningless values, risking silent
data corruption in downstream analysis.

### Verified On

- **NetworkX 3.6.1**, Python 3.12.10
- All three FW variants affected: `floyd_warshall`, `floyd_warshall_predecessor_and_distance`, `floyd_warshall_numpy`

---

## Hypothesis Features Used

| Feature | Where used | Purpose |
|---|---|---|
| `@st.composite` | All graph strategies | Build complex graph objects from primitive draws |
| `@given` | All 20 property tests | Generate random inputs |
| `@example` | Test 1 (self-loop graph) | Pin important edge cases |
| `@settings` | All tests | Control `max_examples` and `suppress_health_check` |
| `assume()` | Tests 11, 12, 17 | Skip invalid inputs |
| `st.data()` | Tests 11, 12 | Draw values dependent on generated graph (proper shrinking) |
| `event()` | Tests 1, 8 | Track topology/size distribution for coverage analysis |
| `target()` | Test 2 | Guide generation toward denser DAGs |

---

## Key Design Decisions

### 1. Single self-contained file

The rubric requires "a single Python file" with all imports, strategies,
helpers, and tests.  Graph generation strategies are embedded at the top
of `test_floyd_warshall.py` rather than in a separate module.

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

### 4. Cross-algorithm validation as differential testing

Tests 8-9 validate Floyd-Warshall against Dijkstra (non-negative weights)
and Bellman-Ford (negative weights on DAGs).  These are fundamentally
different algorithms solving the same problem, so agreement provides
stronger evidence of correctness than comparing three implementations of
the same algorithm (Tests 6-7).

### 5. Hypothesis-controlled randomness for proper shrinking

Tests 11 and 12 use `st.data().draw()` instead of Python's `random`
module to select edges.  This allows Hypothesis to shrink failing examples
to minimal counterexamples, which is critical for debugging.

### 6. Adapted to installed NetworkX behaviour

The installed NetworkX (3.6.1) does **not** raise exceptions for negative
cycles -- instead, `floyd_warshall` returns negative diagonal entries.
Test 18 checks this actual behaviour, and the bug discovery test documents
the API inconsistency.

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
