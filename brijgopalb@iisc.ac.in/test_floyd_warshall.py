"""
Property-Based Tests for NetworkX Floyd-Warshall Algorithms
============================================================

Team member : Brijgopal Bharadwaj (brijgopalb@iisc.ac.in)
Algorithm   : Floyd-Warshall all-pairs shortest paths
Module      : networkx.algorithms.shortest_paths.dense

Functions under test
--------------------
  1. nx.floyd_warshall(G, weight)
  2. nx.floyd_warshall_predecessor_and_distance(G, weight)
  3. nx.floyd_warshall_numpy(G, nodelist, weight)
  4. nx.reconstruct_path(source, target, predecessors)

This single file contains:
  - Reusable Hypothesis graph-generation strategies (3-layer architecture)
  - 24 property-based tests with detailed docstrings
  - 1 bug-discovery test documenting an API inconsistency in NetworkX

All tests use the Hypothesis library to generate diverse graph structures.
"""

import math

import networkx as nx
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck, example, event, target

INF = float("inf")
MAX_EXAMPLES = 80
SLOW_OK = [HealthCheck.too_slow]


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
#
# Three composable layers:
#   Layer 1 — Topology strategies  (unweighted structure)
#   Layer 2 — Modifier helpers     (weights, self-loops, isolated nodes)
#   Layer 3 — graph_builder()      (composes Layer 1 + Layer 2 in one call)
#
# Plus two standalone strategies with bespoke construction logic.
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Layer 1 — Topology strategies
# ---------------------------------------------------------------------------
# Every topology strategy has the uniform signature
#     (draw, min_nodes, max_nodes, directed)
# so that graph_builder can call any of them interchangeably.
# ---------------------------------------------------------------------------

@st.composite
def random_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Erdos-Renyi G(n, p) topology."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    p = draw(st.floats(min_value=0.1, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return nx.gnp_random_graph(n, p, seed=seed, directed=directed)


@st.composite
def complete_graph_topology(draw, min_nodes=2, max_nodes=12, directed=True):
    """Complete graph / complete digraph on n nodes."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    if directed:
        return nx.complete_graph(n, create_using=nx.DiGraph)
    return nx.complete_graph(n)


@st.composite
def path_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Simple directed or undirected path 0 -> 1 -> ... -> n-1."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    if directed:
        return nx.path_graph(n, create_using=nx.DiGraph)
    return nx.path_graph(n)


@st.composite
def cycle_graph_topology(draw, min_nodes=3, max_nodes=15, directed=True):
    """Directed or undirected cycle on n nodes."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    if directed:
        return nx.cycle_graph(n, create_using=nx.DiGraph)
    return nx.cycle_graph(n)


@st.composite
def star_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Star graph with one hub connected to n-1 leaves."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.star_graph(n - 1)
    if directed:
        return nx.DiGraph(G)
    return G


@st.composite
def tree_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Random labeled tree on n nodes (Prufer-sequence based)."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    T = nx.random_labeled_tree(n, seed=seed)
    if directed:
        return nx.DiGraph(T)
    return T


@st.composite
def empty_graph_topology(draw, min_nodes=1, max_nodes=10, directed=True):
    """Graph with n nodes and zero edges."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    if directed:
        return nx.empty_graph(n, create_using=nx.DiGraph)
    return nx.empty_graph(n)


@st.composite
def disconnected_graph_topology(draw, min_nodes=4, max_nodes=14, directed=True):
    """Two disjoint cliques with no edges between them."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    split = draw(st.integers(min_value=1, max_value=n - 1))

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(split):
        for j in range(i + 1, split):
            G.add_edge(i, j)
            if directed:
                G.add_edge(j, i)
    for i in range(split, n):
        for j in range(i + 1, n):
            G.add_edge(i, j)
            if directed:
                G.add_edge(j, i)
    return G


@st.composite
def dag_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Random DAG: edges only go from lower-index to higher-index nodes.

    DAGs are inherently directed.  The ``directed`` parameter is accepted
    for API consistency with other topologies but has no effect.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if draw(st.booleans()):
                G.add_edge(i, j)
    return G


@st.composite
def bipartite_graph_topology(draw, min_nodes=4, max_nodes=14, directed=True):
    """Random bipartite graph with two partitions.

    Note: intentionally excluded from ALL_TOPOLOGIES.  Most general FW
    properties (triangle inequality, subpath optimality, etc.) are equally
    well exercised by random_graph_topology.  This strategy is reserved for
    the dedicated test_bipartite_parity_of_distances test, which exploits
    the structural property unique to bipartite graphs: every path between
    two nodes in the same partition has even hop-count.

    When ``directed=False`` the returned graph is always connected.  This
    avoids the HealthCheck.filter_too_much problem that would arise if the
    test used ``assume(nx.is_connected(G))`` on a sparse random bipartite
    graph (most of which are disconnected).  Connectivity is guaranteed by
    a caterpillar spanning tree: left[0] is connected to every right node,
    and every remaining left node is connected to right[0].  All added
    edges cross the bipartition, so the graph remains bipartite.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    split = draw(st.integers(min_value=1, max_value=n - 1))
    left = list(range(split))
    right = list(range(split, n))

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    if not directed:
        # Caterpillar spanning tree guaranteeing connectivity:
        #   left[0] -- right[j]  for every j  (left[0] as hub)
        #   left[i] -- right[0]  for every i > 0
        for rj in right:
            G.add_edge(left[0], rj)
        for li in left[1:]:
            G.add_edge(li, right[0])

    # Add random extra cross-partition edges
    for i in left:
        for j in right:
            if not G.has_edge(i, j) and draw(st.booleans()):
                G.add_edge(i, j)
                if directed and draw(st.booleans()):
                    G.add_edge(j, i)
    return G


ALL_TOPOLOGIES = [
    random_graph_topology,
    complete_graph_topology,
    path_graph_topology,
    cycle_graph_topology,
    star_graph_topology,
    tree_graph_topology,
]


# ---------------------------------------------------------------------------
# Layer 2 — Modifier helpers (mutate a graph in-place)
# ---------------------------------------------------------------------------

def _assign_weights(draw, G, min_weight, max_weight):
    """Draw a random integer weight for every edge in G."""
    for u, v in G.edges():
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G[u][v]["weight"] = w
    return G


def _assign_uniform_weight(draw, G, min_weight=1, max_weight=50):
    """Set every edge to the *same* random weight (drawn once)."""
    w = draw(st.integers(min_value=min_weight, max_value=max_weight))
    for u, v in G.edges():
        G[u][v]["weight"] = w
    return G


def _add_self_loops(draw, G, min_weight=1, max_weight=50, max_loops=None):
    """Add self-loops with positive weights to a random subset of nodes.

    Draws the subset size first, then draws exactly that many distinct nodes
    without a redundant .filter() call (which could cause Hypothesis
    HealthCheck.filter_too_much on small graphs).
    """
    nodes = list(G.nodes())
    if not nodes:
        return G
    cap = min(max_loops if max_loops is not None else len(nodes), len(nodes))
    count = draw(st.integers(min_value=1, max_value=max(1, cap)))
    # st.lists with unique=True and equal min/max_size produces exactly
    # `count` distinct elements -- no filter needed.
    chosen = draw(st.lists(
        st.sampled_from(nodes), min_size=count, max_size=count, unique=True,
    ))
    for v in chosen:
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G.add_edge(v, v, weight=w)
    return G


def _add_isolated_nodes(draw, G, min_isolates=1, max_isolates=3):
    """Append isolated nodes (no edges) to the graph."""
    k = draw(st.integers(min_value=min_isolates, max_value=max_isolates))
    base = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
    G.add_nodes_from(range(base, base + k))
    return G


# ---------------------------------------------------------------------------
# Layer 3 — Composable builder
# ---------------------------------------------------------------------------

@st.composite
def graph_builder(draw,
                  topology=None,
                  directed=True,
                  min_nodes=2, max_nodes=12,
                  min_weight=1, max_weight=50,
                  uniform_weight=False,
                  self_loops=False,
                  isolated_nodes=False):
    """Composable graph strategy: topology + weights + optional modifiers.

    Parameters
    ----------
    topology : callable or None
        A topology strategy function (e.g. ``random_graph_topology``).
        ``None`` draws uniformly from ``ALL_TOPOLOGIES``.
    directed : bool
        Whether the resulting graph should be directed.
    min_nodes, max_nodes : int
        Node count bounds forwarded to the topology strategy.
    min_weight, max_weight : int
        Edge weight bounds.
    uniform_weight : bool
        If True, all edges get the *same* randomly-drawn weight.
    self_loops : bool
        If True, add positive-weight self-loops to a random subset of nodes.
    isolated_nodes : bool
        If True, append 1-3 isolated nodes with no edges.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
    """
    if topology is None:
        topology = draw(st.sampled_from(ALL_TOPOLOGIES))

    G = draw(topology(min_nodes=min_nodes, max_nodes=max_nodes, directed=directed))

    if uniform_weight:
        _assign_uniform_weight(draw, G, min_weight, max_weight)
    else:
        _assign_weights(draw, G, min_weight, max_weight)

    if self_loops:
        _add_self_loops(draw, G, min_weight=max(1, min_weight), max_weight=max_weight)

    if isolated_nodes:
        _add_isolated_nodes(draw, G)

    return G


# ---------------------------------------------------------------------------
# Specialized strategies (bespoke construction that doesn't fit builder)
# ---------------------------------------------------------------------------

@st.composite
def dag_with_weights(draw, min_nodes=2, max_nodes=15,
                     min_weight=-10, max_weight=20):
    """DAG with arbitrary (including negative) integer weights.

    Because a DAG has no cycles at all, negative edge weights are safe
    for shortest-path algorithms -- there can never be a negative-weight
    cycle.  Uses ``dag_topology`` directly rather than ``graph_builder``
    because the DAG edge-ordering invariant (i < j) must be preserved.
    """
    G = draw(dag_topology(min_nodes=min_nodes, max_nodes=max_nodes))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def negative_cycle_digraph(draw, min_nodes=3, max_nodes=10):
    """Directed graph guaranteed to contain at least one negative-weight cycle.

    Construction: build a directed cycle on n nodes, assign weights so
    that the total cycle weight is negative, then optionally sprinkle
    extra positive-weight edges.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    cycle_weights = []
    for i in range(n):
        w = draw(st.integers(min_value=-20, max_value=10))
        cycle_weights.append(w)
        G.add_edge(i, (i + 1) % n, weight=w)

    total = sum(cycle_weights)
    if total >= 0:
        deficit = total + draw(st.integers(min_value=1, max_value=20))
        G[0][1]["weight"] -= deficit

    for i in range(n):
        for j in range(n):
            if i != j and not G.has_edge(i, j) and draw(st.booleans()):
                w = draw(st.integers(min_value=1, max_value=20))
                G.add_edge(i, j, weight=w)

    return G


# ═══════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _fw_dist(G):
    """Return dict-of-dict distances from floyd_warshall."""
    return nx.floyd_warshall(G)


def _fw_pred_dist(G):
    """Return (predecessors, distances) from floyd_warshall_predecessor_and_distance."""
    return nx.floyd_warshall_predecessor_and_distance(G)


def _fw_numpy(G):
    """Return (nodelist, distance_matrix) from floyd_warshall_numpy."""
    nodelist = sorted(G.nodes())
    A = nx.floyd_warshall_numpy(G, nodelist=nodelist)
    return nodelist, A


def _path_weight(G, path):
    """Sum of edge weights along a path (list of nodes)."""
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v].get("weight", 1.0)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT PROPERTIES (Tests 1 - 5)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 1 — Zero self-distance
# ---------------------------------------------------------------------------

_SELF_LOOP_GRAPH = nx.DiGraph()
_SELF_LOOP_GRAPH.add_edge(0, 1, weight=3)
_SELF_LOOP_GRAPH.add_edge(0, 0, weight=7)

@example(G=_SELF_LOOP_GRAPH)
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder())
def test_zero_self_distance(G):
    """
    Property: For every node v in a graph without negative cycles,
    dist(v, v) == 0.

    Mathematical basis: The trivial path consisting of just the node v has
    length 0.  In the absence of negative-weight cycles, no cycle through v
    can have negative total weight, so the shortest "path" from v to itself
    is the zero-length trivial path.  This is an axiom of shortest-path
    distance in graphs without negative cycles.

    Test strategy: Generate directed graphs drawn from a mix of topologies
    (Erdos-Renyi, complete, path, cycle, star, tree) with positive integer
    edge weights via ``graph_builder()``.  Positive weights guarantee no
    negative cycles.  Compute all-pairs distances with floyd_warshall and
    verify every diagonal entry.  An @example pins a graph with a positive
    self-loop to ensure this specific edge case is always exercised.

    Assumptions / preconditions:
      - All edge weights are positive (>= 1), so no negative cycles exist.

    Why failure matters: A non-zero self-distance would mean the algorithm
    believes it costs something to stay in place, indicating a fundamental
    error in initialisation or relaxation.
    """
    event(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    dist = _fw_dist(G)
    for v in G.nodes():
        assert dist[v][v] == 0, (
            f"dist({v}, {v}) = {dist[v][v]}, expected 0"
        )


# ---------------------------------------------------------------------------
# Test 2 — Triangle inequality
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=dag_with_weights())
def test_triangle_inequality(G):
    """
    Property: For all nodes u, v, w in G:
        dist(u, w) <= dist(u, v) + dist(v, w)

    Mathematical basis: If there is a path from u to v of cost d1 and a path
    from v to w of cost d2, then concatenating them gives a u->w walk of cost
    d1 + d2.  The shortest path can only be shorter or equal, never longer.
    This is the defining property of a shortest-path metric.  It holds
    whenever no negative cycles exist (which a DAG guarantees).

    Test strategy: Generate DAGs with arbitrary (including negative) integer
    edge weights.  A DAG cannot contain any cycle, so negative weights are
    safe.  Compute all-pairs distances and check the inequality for every
    ordered triple of nodes.  Hypothesis's target() directive guides
    generation toward denser graphs where violations are more likely.

    Assumptions / preconditions:
      - Graph is a DAG, so no negative cycles can exist.

    Why failure matters: A violation means the algorithm found paths to u->v
    and v->w whose costs sum to less than the supposed shortest u->w path.
    This implies the algorithm missed a shorter route through v, a critical
    correctness bug.
    """
    target(float(G.number_of_edges()), label="dag_edge_count")

    dist = _fw_dist(G)
    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            for w in nodes:
                d_uw = dist[u][w]
                d_uv = dist[u][v]
                d_vw = dist[v][w]
                if d_uv < INF and d_vw < INF:
                    assert d_uw <= d_uv + d_vw + 1e-9, (
                        f"Triangle inequality violated: "
                        f"dist({u},{w})={d_uw} > "
                        f"dist({u},{v})+dist({v},{w})={d_uv}+{d_vw}={d_uv+d_vw}"
                    )


# ---------------------------------------------------------------------------
# Test 3 — Symmetry on undirected graphs
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology, directed=False))
def test_symmetry_undirected(G):
    """
    Property: For an undirected graph, dist(u, v) == dist(v, u) for all
    node pairs (u, v).

    Mathematical basis: In an undirected graph every edge {u, v} can be
    traversed in either direction at the same cost.  Therefore any path
    from u to v can be reversed to obtain a path from v to u with the
    same total weight, giving dist(u, v) = dist(v, u).

    Test strategy: Generate undirected Erdos-Renyi graphs with positive
    integer weights and verify pairwise symmetry of the distance matrix.

    Assumptions / preconditions:
      - The graph is undirected (nx.Graph, not nx.DiGraph).

    Why failure matters: Asymmetric distances in an undirected graph would
    mean the algorithm treats edge directions inconsistently, a serious bug
    in how it handles the adjacency structure.
    """
    dist = _fw_dist(G)
    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            assert dist[u][v] == dist[v][u], (
                f"Symmetry violated: dist({u},{v})={dist[u][v]} "
                f"!= dist({v},{u})={dist[v][u]}"
            )


# ---------------------------------------------------------------------------
# Test 4 — Reconstructed path weight equals reported distance
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology))
def test_path_weight_equals_distance(G):
    """
    Property: For every reachable pair (s, t) with s != t, the path
    reconstructed via reconstruct_path has total edge weight equal to the
    distance reported by floyd_warshall_predecessor_and_distance.

    Mathematical basis: The predecessor dictionary encodes the actual
    shortest-path tree.  Walking backwards from t through predecessors and
    summing edge weights must reproduce the computed distance.  Any
    discrepancy means the predecessor and distance data are inconsistent.

    Test strategy: Generate random directed graphs with positive weights.
    Compute predecessors and distances, then for every reachable pair
    reconstruct the path and verify its weight matches.

    Assumptions / preconditions:
      - Only pairs where a finite-distance path exists are checked.
      - s != t (reconstruct_path returns [] when s == t).

    Why failure matters: Inconsistency between predecessors and distances
    means the path the algorithm claims is shortest does not actually have
    the claimed cost -- either the distance is wrong or the predecessor
    chain is broken.
    """
    pred, dist = _fw_pred_dist(G)
    nodes = list(G.nodes())

    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            d_st = dist.get(s, {}).get(t, INF)
            if d_st >= INF:
                continue

            path = nx.reconstruct_path(s, t, pred)
            assert len(path) >= 2, (
                f"Path from {s} to {t} should have >= 2 nodes, got {path}"
            )
            assert path[0] == s and path[-1] == t, (
                f"Path endpoints wrong: expected ({s},{t}), got ({path[0]},{path[-1]})"
            )

            actual_weight = _path_weight(G, path)
            assert abs(actual_weight - d_st) < 1e-9, (
                f"Path weight {actual_weight} != reported distance {d_st} "
                f"for {s}->{t}, path={path}"
            )


# ---------------------------------------------------------------------------
# Test 5 — Subpath optimality
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology, min_nodes=3, max_nodes=10))
def test_subpath_optimality(G):
    """
    Property: Every sub-path of a shortest path is itself a shortest path.
    Formally, if P = [s, ..., a, ..., b, ..., t] is a shortest s->t path,
    then the sub-path from a to b within P has total weight equal to
    dist(a, b).

    Mathematical basis: This is Bellman's principle of optimality.  If a
    sub-path a->b were not optimal, we could replace it with a shorter one,
    contradicting the optimality of the full path P.

    Test strategy: Generate directed graphs with positive weights, compute
    shortest paths, and for every path with >= 3 nodes verify that each
    contiguous sub-path has cost equal to the all-pairs shortest distance.

    Assumptions / preconditions:
      - Positive weights ensure no negative cycles.
      - Only paths with >= 3 nodes are interesting (sub-paths of 2-node
        paths are trivially single edges).

    Why failure matters: Violation of subpath optimality means the
    algorithm's predecessor chain is globally inconsistent -- it encodes a
    path that is not actually shortest, even though the distance value
    might be correct.
    """
    pred, dist = _fw_pred_dist(G)
    nodes = list(G.nodes())

    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            d_st = dist.get(s, {}).get(t, INF)
            if d_st >= INF:
                continue

            path = nx.reconstruct_path(s, t, pred)
            if len(path) < 3:
                continue

            for i in range(len(path)):
                for j in range(i + 2, len(path)):
                    a, b = path[i], path[j]
                    subpath_weight = _path_weight(G, path[i:j + 1])
                    d_ab = dist.get(a, {}).get(b, INF)
                    assert abs(subpath_weight - d_ab) < 1e-9, (
                        f"Subpath {path[i:j+1]} has weight {subpath_weight} "
                        f"but dist({a},{b})={d_ab}"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-IMPLEMENTATION CONSISTENCY (Tests 6 - 7)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 6 — floyd_warshall dict vs floyd_warshall_predecessor_and_distance
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=dag_with_weights())
def test_fw_dict_vs_pred_dist(G):
    """
    Property: The distance dictionary returned by floyd_warshall(G) is
    identical to the distance dictionary returned as the second element of
    floyd_warshall_predecessor_and_distance(G).

    Mathematical basis: floyd_warshall is implemented as a thin wrapper
    that calls floyd_warshall_predecessor_and_distance and returns only the
    distance component.  Any divergence would indicate either a wrapper bug
    or unintended state mutation.

    Test strategy: Generate DAGs with mixed (positive and negative) edge
    weights, call both functions, and compare the distance dictionaries
    entry by entry.

    Assumptions / preconditions:
      - DAG structure prevents negative cycles.

    Why failure matters: If two functions that should return the same data
    disagree, one of them is wrong.  Since floyd_warshall is the public
    convenience API, a discrepancy could silently propagate incorrect
    distances to users.
    """
    dist_fw = _fw_dist(G)
    _, dist_pd = _fw_pred_dist(G)

    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            d1 = dist_fw[u][v]
            d2 = dist_pd.get(u, {}).get(v, INF)
            if math.isinf(d1) and math.isinf(d2):
                continue
            assert abs(d1 - d2) < 1e-9, (
                f"Disagreement: fw[{u}][{v}]={d1}, pred_dist[{u}][{v}]={d2}"
            )


# ---------------------------------------------------------------------------
# Test 7 — floyd_warshall dict vs floyd_warshall_numpy
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology))
def test_fw_dict_vs_numpy(G):
    """
    Property: The distance dictionary from floyd_warshall and the NumPy
    matrix from floyd_warshall_numpy encode the same all-pairs distances,
    up to floating-point tolerance.

    Mathematical basis: Both functions implement the same Floyd-Warshall
    relaxation but using different data structures (dict-of-dict vs NumPy
    array).  For the same input the mathematical result is identical.

    Test strategy: Generate directed graphs with positive weights.  Compute
    distances via both functions.  Map the dict representation to array
    indices (using a sorted nodelist) and compare element-wise.

    Assumptions / preconditions:
      - Positive weights ensure well-defined shortest paths.

    Why failure matters: A mismatch between the two representations would
    mean at least one implementation has a bug in its relaxation loop or
    index mapping.
    """
    dist_dict = _fw_dist(G)
    nodelist, A = _fw_numpy(G)
    node_idx = {v: i for i, v in enumerate(nodelist)}

    for u in nodelist:
        for v in nodelist:
            d_dict = dist_dict[u][v]
            d_np = float(A[node_idx[u], node_idx[v]])
            if math.isinf(d_dict) and math.isinf(d_np):
                continue
            assert abs(d_dict - d_np) < 1e-9, (
                f"Dict vs NumPy mismatch: dist[{u}][{v}] = {d_dict} vs {d_np}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-ALGORITHM VALIDATION (Tests 8 - 9)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 8 — Floyd-Warshall vs Dijkstra (non-negative weights)
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology))
def test_fw_vs_dijkstra(G):
    """
    Property: For graphs with non-negative edge weights, the all-pairs
    distances from Floyd-Warshall match the single-source distances from
    Dijkstra's algorithm for every source node.

    Mathematical basis: Both algorithms solve the same shortest-path
    problem.  Dijkstra's operates via a priority-queue relaxation that is
    correct for non-negative weights; Floyd-Warshall uses dynamic
    programming over intermediate vertices.  For the same input, both must
    produce identical distance values.

    Test strategy: Generate random directed graphs with positive integer
    weights (guaranteeing non-negative edges).  For each node as source,
    run Dijkstra and compare against the corresponding row of the FW
    distance matrix.  This is a cross-algorithm differential test: it
    validates FW against a completely independent implementation.

    Assumptions / preconditions:
      - All edge weights are positive (>= 1).
      - Both algorithms are applied to the same graph.

    Why failure matters: A disagreement between two independent algorithms
    that should give the same answer on non-negative-weight graphs would
    indicate a bug in at least one of them.  Unlike cross-implementation
    tests (Tests 6-7), this compares fundamentally different algorithmic
    approaches, catching classes of bugs that internal consistency checks
    cannot.
    """
    event(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    dist_fw = _fw_dist(G)

    for s in G.nodes():
        dijkstra_dist = dict(nx.single_source_dijkstra_path_length(G, s))
        for t in G.nodes():
            d_fw = dist_fw[s][t]
            d_dj = dijkstra_dist.get(t, INF)
            if math.isinf(d_fw) and math.isinf(d_dj):
                continue
            assert abs(d_fw - d_dj) < 1e-9, (
                f"FW vs Dijkstra mismatch: fw[{s}][{t}]={d_fw}, "
                f"dijkstra={d_dj}"
            )


# ---------------------------------------------------------------------------
# Test 9 — Floyd-Warshall vs Bellman-Ford (negative weights, no neg cycles)
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=dag_with_weights())
def test_fw_vs_bellman_ford(G):
    """
    Property: For graphs without negative cycles, the all-pairs distances
    from Floyd-Warshall match the single-source distances from Bellman-Ford
    for every source node.

    Mathematical basis: Bellman-Ford uses iterative edge relaxation and is
    correct for graphs with negative edges but no negative cycles.
    Floyd-Warshall uses a different algorithmic paradigm (dynamic
    programming over intermediate vertices).  On valid inputs (no negative
    cycles), both must agree.  DAGs guarantee no cycles of any kind.

    Test strategy: Generate DAGs with mixed-sign edge weights.  For each
    source node, run Bellman-Ford and compare against the FW distance row.
    This validates FW's handling of negative edge weights against an
    independent algorithm that also supports them.

    Assumptions / preconditions:
      - Graph is a DAG (no cycles, hence no negative cycles).
      - Bellman-Ford is guaranteed to terminate correctly on this input.

    Why failure matters: A disagreement on negative-weight graphs (where
    Dijkstra cannot be used) would reveal a bug in FW's relaxation when
    negative edges are present -- a scenario that cross-implementation
    checks alone cannot validate.
    """
    dist_fw = _fw_dist(G)

    for s in G.nodes():
        bf_dist = dict(nx.single_source_bellman_ford_path_length(G, s))
        for t in G.nodes():
            d_fw = dist_fw[s][t]
            d_bf = bf_dist.get(t, INF)
            if math.isinf(d_fw) and math.isinf(d_bf):
                continue
            assert abs(d_fw - d_bf) < 1e-9, (
                f"FW vs Bellman-Ford mismatch: fw[{s}][{t}]={d_fw}, "
                f"bellman_ford={d_bf}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# METAMORPHIC PROPERTIES (Tests 10 - 14)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 10 — Weight scaling
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(
    G=graph_builder(topology=random_graph_topology),
    k=st.integers(min_value=2, max_value=10),
)
def test_weight_scaling(G, k):
    """
    Property: Multiplying every edge weight by a positive constant k
    multiplies all finite shortest-path distances by k.  Infinite distances
    (unreachable pairs) remain infinite.

    Mathematical basis: If P is a shortest s->t path with weight W(P), then
    under scaling each edge by k the same path has weight k*W(P).  Since
    scaling preserves the ordering of path weights, P remains optimal.
    Hence dist_scaled(s,t) = k * dist_original(s,t) for all reachable
    pairs.

    Test strategy: Compute distances on the original graph, scale all
    weights by k, recompute, and verify the ratio.

    Assumptions / preconditions:
      - k >= 2 (non-trivial scaling).
      - All original weights are positive.

    Why failure matters: A wrong ratio means the algorithm's relaxation is
    not linear in edge weights, suggesting an arithmetic or initialisation
    error.
    """
    dist_orig = _fw_dist(G)

    G_scaled = G.copy()
    for u, v in G_scaled.edges():
        G_scaled[u][v]["weight"] *= k

    dist_scaled = _fw_dist(G_scaled)

    for u in G.nodes():
        for v in G.nodes():
            d_o = dist_orig[u][v]
            d_s = dist_scaled[u][v]
            if math.isinf(d_o):
                assert math.isinf(d_s), (
                    f"dist_orig[{u}][{v}]=inf but dist_scaled={d_s}"
                )
            else:
                assert abs(d_s - k * d_o) < 1e-9, (
                    f"Scaling by {k}: dist_scaled[{u}][{v}]={d_s}, "
                    f"expected {k}*{d_o}={k*d_o}"
                )


# ---------------------------------------------------------------------------
# Test 11 — Adding a non-negative edge can only decrease distances
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(
    G=graph_builder(
        topology=random_graph_topology, min_nodes=3, max_nodes=10, min_weight=0),
    w=st.integers(min_value=0, max_value=50),
    data=st.data(),
)
def test_edge_addition_monotonicity(G, w, data):
    """
    Property: Adding a single edge with non-negative weight to a graph can
    only decrease (or maintain) pairwise shortest-path distances.

    Mathematical basis: Adding an edge introduces a new potential path
    segment.  For any pair (u, v), the new shortest distance is
    min(old_dist(u,v), old_dist(u,a) + w(a,b) + old_dist(b,v)) where (a,b)
    is the new edge.  Since w >= 0 and the min operator can only shrink,
    distances are monotonically non-increasing under edge addition.

    Test strategy: Compute distances on the original graph, add a random
    non-negative-weight edge between two nodes that are not already
    connected (selected via Hypothesis data.draw for proper shrinking),
    recompute, and verify every distance is <= the original.

    Assumptions / preconditions:
      - All existing weights are non-negative.
      - The new edge weight w >= 0.
      - There exists at least one non-edge to add.

    Why failure matters: An increased distance after adding an edge would
    violate a basic monotonicity property of shortest paths, indicating
    the algorithm fails to consider all available paths.
    """
    non_edges = [(u, v) for u in G.nodes() for v in G.nodes()
                 if u != v and not G.has_edge(u, v)]
    assume(len(non_edges) > 0)

    dist_before = _fw_dist(G)

    a, b = data.draw(st.sampled_from(non_edges), label="new_edge")
    G2 = G.copy()
    G2.add_edge(a, b, weight=w)
    dist_after = _fw_dist(G2)

    for u in G.nodes():
        for v in G.nodes():
            assert dist_after[u][v] <= dist_before[u][v] + 1e-9, (
                f"Distance increased after adding edge ({a},{b},w={w}): "
                f"dist[{u}][{v}] went from {dist_before[u][v]} to {dist_after[u][v]}"
            )


# ---------------------------------------------------------------------------
# Test 12 — Subgraph distances are a lower bound
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(
    G=graph_builder(topology=random_graph_topology, min_nodes=4, max_nodes=10),
    data=st.data(),
)
def test_subgraph_distance_lower_bound(G, data):
    """
    Property: For a subgraph H of G (same nodes, subset of edges),
    dist_G(u,v) <= dist_H(u,v) for all u, v.

    Mathematical basis: H has fewer edges than G, so fewer candidate paths
    between any pair.  The minimum over a subset cannot be smaller than the
    minimum over the superset.  This generalises the edge-addition
    monotonicity to arbitrary edge subsets.

    Test strategy: Generate a directed graph G with positive weights.
    Construct a subgraph H by keeping each edge independently with a
    Hypothesis-drawn boolean (enabling proper shrinking of failing
    examples).  Compute distances on both and verify the inequality.

    Assumptions / preconditions:
      - Positive weights ensure no negative cycles in either G or H.

    Why failure matters: If distances in G exceed those in H, the algorithm
    is finding shorter paths with fewer edges available, which is logically
    impossible and indicates a relaxation bug.
    """
    assume(G.number_of_edges() >= 2)

    edges = list(G.edges())
    keep_flags = data.draw(
        st.lists(st.booleans(), min_size=len(edges), max_size=len(edges)),
        label="keep_flags",
    )

    H = G.copy()
    edges_to_remove = [e for e, keep in zip(edges, keep_flags) if not keep]
    H.remove_edges_from(edges_to_remove)

    assume(H.number_of_edges() > 0)

    dist_G = _fw_dist(G)
    dist_H = _fw_dist(H)

    for u in G.nodes():
        for v in G.nodes():
            assert dist_G[u][v] <= dist_H[u][v] + 1e-9, (
                f"dist_G[{u}][{v}]={dist_G[u][v]} > "
                f"dist_H[{u}][{v}]={dist_H[u][v]} "
                f"(supergraph has longer distance than subgraph)"
            )


# ---------------------------------------------------------------------------
# Test 13 — Reversing edges transposes the distance matrix
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=dag_with_weights())
def test_graph_reversal_transposes_distances(G):
    """
    Property: For a digraph G and its reverse G^R (all edges flipped),
    dist_G(u, v) == dist_{G^R}(v, u) for all node pairs.

    Mathematical basis: Every path u -> x1 -> ... -> xk -> v in G
    corresponds to the reversed path v -> xk -> ... -> x1 -> u in G^R
    with the same total weight.  This bijection preserves path weights,
    so the shortest u->v path in G maps to the shortest v->u path in G^R.

    Test strategy: Generate DAGs with mixed-sign weights (no negative
    cycles by construction), compute distances on both G and its reverse,
    and verify the transpose relationship.

    Assumptions / preconditions:
      - DAG structure guarantees no negative cycles in G; reversing a DAG
        also yields a DAG.

    Why failure matters: A mismatch means the algorithm handles edge
    direction inconsistently, possibly iterating over adjacency structures
    incorrectly when edges point in different directions.
    """
    G_rev = G.reverse()
    dist_G = _fw_dist(G)
    dist_R = _fw_dist(G_rev)

    for u in G.nodes():
        for v in G.nodes():
            d_orig = dist_G[u][v]
            d_rev = dist_R[v][u]
            if math.isinf(d_orig) and math.isinf(d_rev):
                continue
            assert abs(d_orig - d_rev) < 1e-9, (
                f"Transpose mismatch: dist_G[{u}][{v}]={d_orig}, "
                f"dist_R[{v}][{u}]={d_rev}"
            )


# ---------------------------------------------------------------------------
# Test 14 — Adding an isolated node does not change existing distances
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology))
def test_node_addition_invariance(G):
    """
    Property: Adding an isolated node (no incident edges) to the graph
    does not change the pairwise distances between any existing nodes.

    Mathematical basis: An isolated node u has no incident edges, so no
    path between existing nodes can pass through u.  The set of candidate
    paths for any existing pair (s, t) is unchanged, so the shortest-path
    distances must remain identical.

    Test strategy: Compute FW distances on the original graph, add a new
    isolated node with a label not already in the graph, recompute, and
    verify that every distance between original nodes is preserved.  Also
    verify that distances involving the new node are 0 (self) or inf (all
    others).

    Assumptions / preconditions:
      - The new node has no edges.

    Why failure matters: Changed distances after adding an isolated node
    would indicate the algorithm's relaxation loop is sensitive to the
    mere presence of vertices in the node set, even when they contribute
    no paths -- a structural bug in how the DP table is initialised or
    iterated.
    """
    dist_before = _fw_dist(G)
    original_nodes = list(G.nodes())

    new_node = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
    G2 = G.copy()
    G2.add_node(new_node)
    dist_after = _fw_dist(G2)

    for u in original_nodes:
        for v in original_nodes:
            d_before = dist_before[u][v]
            d_after = dist_after[u][v]
            if math.isinf(d_before) and math.isinf(d_after):
                continue
            assert abs(d_before - d_after) < 1e-9, (
                f"Distance changed after adding isolated node: "
                f"dist[{u}][{v}] was {d_before}, now {d_after}"
            )

    assert dist_after[new_node][new_node] == 0
    for v in original_nodes:
        assert dist_after[new_node][v] == INF, (
            f"dist(new_node, {v}) = {dist_after[new_node][v]}, expected inf"
        )
        assert dist_after[v][new_node] == INF, (
            f"dist({v}, new_node) = {dist_after[v][new_node]}, expected inf"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY / EDGE-CASE PROPERTIES (Tests 15 - 18)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 15 — Single-node graph
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(node_id=st.integers(min_value=0, max_value=100))
def test_single_node_self_distance(node_id):
    """
    Property: A graph with a single node v has dist(v, v) = 0, and the
    distance dictionary contains exactly one entry.

    Mathematical basis: The trivial path of length zero from a node to
    itself has cost 0.  With only one node and no edges, there are no
    other pairs to consider.  This is the most degenerate input possible.

    Test strategy: Create a directed graph with a single node (varying the
    node label via Hypothesis) and verify the distance matrix is the 1x1
    zero matrix.

    Assumptions / preconditions:
      - The graph has exactly one node and zero edges.

    Why failure matters: Incorrect output on a single-node graph would
    indicate a severe initialisation or indexing bug in the algorithm's
    handling of degenerate inputs.
    """
    G = nx.DiGraph()
    G.add_node(node_id)
    dist = _fw_dist(G)
    assert dist[node_id][node_id] == 0, (
        f"dist({node_id}, {node_id}) = {dist[node_id][node_id]}, expected 0"
    )


# ---------------------------------------------------------------------------
# Test 16 — Empty graph: all off-diagonal distances are infinite
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=empty_graph_topology(min_nodes=1, max_nodes=10, directed=True))
def test_empty_graph_distances(G):
    """
    Property: In a graph with n nodes and zero edges, dist(v, v) = 0 for
    all v, and dist(u, v) = inf for all u != v.

    Mathematical basis: With no edges, no path of positive length exists
    between distinct nodes.  The only "path" is the trivial zero-length
    path from a node to itself.

    Test strategy: Generate empty directed graphs of varying sizes (1-10
    nodes) and verify every entry of the distance matrix.

    Assumptions / preconditions:
      - The graph has zero edges.

    Why failure matters: Wrong distances on the simplest possible input
    (no edges) would indicate a fundamental initialisation bug -- the
    algorithm is inventing paths that don't exist or failing to set the
    diagonal to zero.
    """
    dist = _fw_dist(G)
    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            if u == v:
                assert dist[u][v] == 0, (
                    f"dist({u},{u}) = {dist[u][v]}, expected 0"
                )
            else:
                assert dist[u][v] == INF, (
                    f"dist({u},{v}) = {dist[u][v]}, expected inf"
                )


# ---------------------------------------------------------------------------
# Test 17 — Disconnected components produce infinite cross-distances
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=disconnected_graph_topology))
def test_disconnected_components(G):
    """
    Property: If nodes u and v are in different weakly-connected components
    of a directed graph, then dist(u, v) = inf and dist(v, u) = inf.

    Mathematical basis: A path from u to v requires a sequence of edges
    connecting them.  If u and v are in separate components, no such
    sequence exists, so the distance is infinite by definition.

    Test strategy: Generate directed graphs with at least two disconnected
    cliques.  Identify weakly-connected components and verify that all
    cross-component distances are infinite, while within-component distances
    are finite.

    Assumptions / preconditions:
      - The graph has multiple weakly-connected components.

    Why failure matters: Finite cross-component distances would mean the
    algorithm found a path through a gap in the graph, indicating it is
    using edges that don't exist.
    """
    components = list(nx.weakly_connected_components(G))
    assume(len(components) >= 2)

    dist = _fw_dist(G)
    node_to_comp = {}
    for idx, comp in enumerate(components):
        for v in comp:
            node_to_comp[v] = idx

    for u in G.nodes():
        for v in G.nodes():
            if node_to_comp[u] != node_to_comp[v]:
                assert dist[u][v] == INF, (
                    f"dist({u},{v})={dist[u][v]} but they are in different "
                    f"components (expected inf)"
                )


# ---------------------------------------------------------------------------
# Test 18 — Negative cycle produces negative self-distance
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(G=negative_cycle_digraph())
def test_negative_cycle_detection(G):
    """
    Property: If a graph contains a negative-weight cycle, then at least one
    node u on that cycle will have dist(u, u) < 0 after running
    Floyd-Warshall.

    Mathematical basis: Floyd-Warshall computes the weight of the shortest
    walk (not necessarily simple path) between every pair.  If a
    negative-weight cycle exists through node u, the algorithm can
    traverse it arbitrarily many times, driving dist(u, u) below zero.
    In the standard three-nested-loop implementation, after all
    intermediate vertices are processed, dist(u, u) < 0 iff u lies on a
    reachable negative cycle.

    Test strategy: Generate directed graphs guaranteed to contain at least
    one negative-weight cycle (constructed by building a Hamiltonian cycle
    whose total weight is forced negative).  Compute distances and verify
    that at least one diagonal entry is negative.

    Assumptions / preconditions:
      - The graph contains at least one directed cycle with negative total
        weight.

    Why failure matters: If no diagonal entry is negative despite a known
    negative cycle, the algorithm has failed to detect it.  Downstream
    consumers relying on this signal to identify pathological inputs would
    silently receive meaningless distance values.
    """
    dist = _fw_dist(G)
    negative_diag = [v for v in G.nodes() if dist[v][v] < 0]
    assert len(negative_diag) > 0, (
        "No negative diagonal found despite guaranteed negative cycle. "
        f"Diag values: {[dist[v][v] for v in G.nodes()]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# POSTCONDITION PROPERTIES (Tests 19, 21 - 24)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 19 — Complete graph with uniform positive weight has known distances
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(
    n=st.integers(min_value=2, max_value=8),
    w=st.integers(min_value=1, max_value=50),
)
def test_complete_graph_uniform_weight(n, w):
    """
    Property: In a complete directed graph where every edge has the same
    positive weight w, dist(u, v) = w for all u != v and dist(u, u) = 0.

    Mathematical basis: In K_n with uniform weight w > 0, the direct edge
    u -> v (which exists for every pair) has weight w.  Any longer path
    u -> x1 -> ... -> v has weight >= 2w > w.  Therefore the shortest
    path is always the single direct edge, giving dist(u, v) = w.  This
    is a closed-form postcondition: we know the exact answer without
    running the algorithm.

    Test strategy: Construct complete digraphs of varying sizes with
    uniform positive weights drawn by Hypothesis.  Verify every entry of
    the FW distance matrix matches the known closed-form solution.

    Assumptions / preconditions:
      - w > 0 (positive uniform weight).
      - Complete directed graph (edge for every ordered pair).

    Why failure matters: Getting the wrong answer on an input where the
    correct distances have a trivial closed form would indicate a
    fundamental arithmetic or structural bug in the algorithm.
    """
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    for u, v in G.edges():
        G[u][v]["weight"] = w

    dist = _fw_dist(G)
    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                assert dist[u][v] == 0, (
                    f"dist({u},{u}) = {dist[u][v]}, expected 0"
                )
            else:
                assert dist[u][v] == w, (
                    f"dist({u},{v}) = {dist[u][v]}, expected {w} "
                    f"(uniform weight on complete graph)"
                )


# ---------------------------------------------------------------------------
# Test 21 — Directed path graph: exact cumulative distances
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(
    weights=st.lists(st.integers(min_value=1, max_value=50),
                     min_size=2, max_size=12),
)
def test_path_graph_exact_distances(weights):
    """
    Property: In a directed path 0 → 1 → … → n-1 with edge weights
    w_0, w_1, …, w_{n-2}, the Floyd-Warshall distance from node i to
    node j (i < j) is exactly the sum of weights w_i + w_{i+1} + … +
    w_{j-1}, and dist(i, j) = inf for i > j (no back-edges).

    Mathematical basis: A directed path has exactly one route between
    any forward pair (i, j): the unique sequence of edges i→i+1→…→j.
    Its cost is the prefix-sum difference  prefix[j] - prefix[i]  where
    prefix[k] = w_0 + … + w_{k-1}.  Because there are no back-edges,
    all reverse distances are infinite.  This gives a closed-form oracle
    for every entry of the distance matrix with no ambiguity.

    Test strategy: Hypothesis draws a list of 2–12 positive integer
    weights; the test builds the corresponding directed path, computes
    FW distances, and checks every pair against the prefix-sum formula.
    Positive weights rule out negative cycles.  Path graphs are the
    simplest DAG and serve as a ground-truth stress-test for the
    algorithm's forward-reachability handling.

    Assumptions / preconditions:
      - All weights are positive (>= 1), so no negative cycles.
      - The graph is a strictly directed path (no reverse edges).

    Why failure matters: An incorrect forward distance would mean the
    algorithm missed the only available path — a fundamental traversal
    bug.  A finite reverse distance would mean the algorithm invented an
    edge that doesn't exist, a structural initialisation error.
    """
    n = len(weights) + 1   # n nodes, n-1 edges
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, w in enumerate(weights):
        G.add_edge(i, i + 1, weight=w)

    # Build prefix sums: prefix[k] = sum of weights for edges 0..k-1
    prefix = [0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + weights[i - 1]

    dist = _fw_dist(G)
    for i in range(n):
        for j in range(n):
            if i == j:
                assert dist[i][j] == 0, (
                    f"dist({i},{i}) = {dist[i][j]}, expected 0"
                )
            elif i < j:
                expected = prefix[j] - prefix[i]
                assert dist[i][j] == expected, (
                    f"dist({i},{j}) = {dist[i][j]}, expected {expected} "
                    f"(prefix[{j}]-prefix[{i}])"
                )
            else:  # i > j: no back-edge exists
                assert dist[i][j] == INF, (
                    f"dist({i},{j}) = {dist[i][j]}, expected inf "
                    f"(no back-edges on directed path)"
                )


# ---------------------------------------------------------------------------
# Test 22 — Directed star graph: exact hub-to-leaf and leaf-to-leaf distances
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(
    spoke_weights=st.lists(
        st.integers(min_value=1, max_value=50),
        min_size=2, max_size=10,
    ),
)
def test_star_graph_exact_distances(spoke_weights):
    """
    Property: In a directed star with hub node 0 and leaves 1…k, where
    every edge points outward (hub → leaf) with weight w_i for leaf i,
    the distances satisfy:

      dist(0, i)  = w_i          (direct spoke)
      dist(i, 0)  = inf          (no inward edges)
      dist(i, j)  = inf  (i ≠ j) (leaves are not connected to each other)
      dist(i, i)  = 0

    Mathematical basis: Because all edges are hub → leaf, leaves i and j
    (i ≠ j) have no directed path between them.  The hub can reach every
    leaf directly; no leaf can reach anything else.  Every closed-form
    value is determined by the graph structure alone, giving a complete
    oracle for all O(n^2) distance matrix entries.

    Test strategy: Hypothesis draws 2–10 spoke weights.  The test builds
    a directed star (hub=0, leaves=1..k), runs floyd_warshall, and checks
    every entry against the formula above.  The star topology is the
    extreme case of a hub-and-spoke network and exercises the algorithm's
    handling of nodes with high out-degree but zero in-degree (leaves)
    and the reverse.

    Assumptions / preconditions:
      - All spoke weights are positive (>= 1).
      - The star is strictly directed outward (hub → leaf only).

    Why failure matters: A finite dist(i, j) for two distinct leaves
    would mean the algorithm constructed a phantom path through a
    non-existent edge — an adjacency-initialisation error.  A wrong
    dist(0, i) would mean it misread the direct edge weight.
    """
    k = len(spoke_weights)      # number of leaves
    hub = 0
    leaves = list(range(1, k + 1))

    G = nx.DiGraph()
    G.add_nodes_from([hub] + leaves)
    for leaf, w in zip(leaves, spoke_weights):
        G.add_edge(hub, leaf, weight=w)

    dist = _fw_dist(G)
    nodes = [hub] + leaves

    for u in nodes:
        for v in nodes:
            if u == v:
                assert dist[u][v] == 0, (
                    f"dist({u},{u}) = {dist[u][v]}, expected 0"
                )
            elif u == hub:  # hub → leaf: direct spoke
                expected = spoke_weights[v - 1]
                assert dist[hub][v] == expected, (
                    f"dist(hub,{v}) = {dist[hub][v]}, expected {expected}"
                )
            else:  # leaf → anything, or leaf → leaf: no path
                assert dist[u][v] == INF, (
                    f"dist({u},{v}) = {dist[u][v]}, expected inf "
                    f"(no outgoing edges from leaves)"
                )


# ---------------------------------------------------------------------------
# Test 23 — Single-edge graph: exact distances and asymmetry
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(
    u=st.integers(min_value=0, max_value=10),
    v=st.integers(min_value=0, max_value=10),
    w=st.integers(min_value=1, max_value=100),
)
def test_single_edge_exact_distances(u, v, w):
    """
    Property: A directed graph containing exactly one edge (u, v) with
    weight w (u ≠ v) has:

      dist(u, v) = w    (the only path)
      dist(v, u) = inf  (no reverse path)
      dist(x, y) = inf  for all other distinct pairs (x, y)
      dist(x, x) = 0    for both nodes x

    Mathematical basis: With a single directed edge there is exactly one
    non-trivial pair for which a finite path exists.  The shortest (and
    only) path from u to v has weight w.  All other ordered pairs have no
    connecting path.  This is the most minimal non-trivial input:
    two nodes and one edge — the simplest possible non-empty graph.

    Test strategy: Hypothesis draws two distinct node labels (filtered via
    assume) and one positive integer weight.  The four distance values are
    fully determined by these three parameters, providing a complete oracle
    with no ambiguity.

    Assumptions / preconditions:
      - u ≠ v (single non-self-loop edge).
      - w > 0 (positive weight).

    Why failure matters: An error on this minimal input indicates the most
    basic initialization of the FW distance table is wrong — either the
    direct edge weight is misread, or the algorithm sets a finite distance
    where no path exists (a false positive reachability bug).
    """
    assume(u != v)

    G = nx.DiGraph()
    G.add_edge(u, v, weight=w)

    dist = _fw_dist(G)

    assert dist[u][u] == 0,   f"dist({u},{u}) = {dist[u][u]}, expected 0"
    assert dist[v][v] == 0,   f"dist({v},{v}) = {dist[v][v]}, expected 0"
    assert dist[u][v] == w,   f"dist({u},{v}) = {dist[u][v]}, expected {w}"
    assert dist[v][u] == INF, f"dist({v},{u}) = {dist[v][u]}, expected inf"


# ---------------------------------------------------------------------------
# Test 24 — Bipartite parity: same-partition distances are even,
#           cross-partition distances are odd (unit weights)
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G_topo=bipartite_graph_topology(directed=False))
def test_bipartite_parity_of_distances(G_topo):
    """
    Property: In a connected undirected bipartite graph with unit edge
    weights, the shortest-path distance between any two distinct nodes
    u and v satisfies:

      dist(u, v) is even  iff  u and v belong to the same partition
      dist(u, v) is odd   iff  u and v belong to different partitions

    Mathematical basis: A graph is bipartite iff it contains no odd-length
    cycles (König, 1916).  Equivalently, its nodes can be 2-coloured so
    that every edge connects nodes of different colours.  Since every
    edge crosses the colour boundary, any walk of k edges moves k times
    across partitions.  After an even number of steps the walk is in the
    same partition it started in; after an odd number it is in the
    opposite partition.  With unit edge weights, distance = hop count, so
    the parity of dist(u, v) is fully determined by partition membership.

    Test strategy: Use bipartite_graph_topology (directed=False) to
    generate random bipartite graphs of 4-14 nodes.  Assign weight=1 to
    all edges so that FW distance equals hop count.  Require connectivity
    via assume() so that all pairwise distances are finite.  Recover the
    two-colouring with nx.bipartite.sets(), which works on any connected
    bipartite graph.  Then verify the parity invariant for every node pair.

    This test is the sole consumer of bipartite_graph_topology; the
    strategy is excluded from ALL_TOPOLOGIES because the parity property
    requires the bipartite structure to be preserved and is vacuous on
    general graphs.  The strategy guarantees connectivity for undirected
    graphs so no assume() filter is needed here.

    Assumptions / preconditions:
      - The graph is undirected and bipartite (guaranteed by the strategy).
      - The graph is connected (guaranteed by the strategy's caterpillar
        spanning tree; no assume() needed).
      - All edge weights are 1 so that distance equals hop count.

    Why failure matters: A parity violation would mean FW found a path
    between two same-partition nodes with an odd number of hops, implying
    it traversed a non-existent edge or made an off-by-one error in
    counting hops.  In a graph where the exact parity is structurally
    determined this would be a fundamental reachability bug.
    """
    G = G_topo.copy()
    for node_u, node_v in G.edges():
        G[node_u][node_v]["weight"] = 1

    left, right = nx.bipartite.sets(G)

    dist = _fw_dist(G)
    nodes = list(G.nodes())

    for node_u in nodes:
        for node_v in nodes:
            if node_u == node_v:
                continue
            d = dist[node_u][node_v]
            if math.isinf(d):
                continue
            hop_count = int(round(d))  # d is an integer under unit weights
            same_partition = (node_u in left) == (node_v in left)
            if same_partition:
                assert hop_count % 2 == 0, (
                    f"dist({node_u},{node_v}) = {hop_count} is odd but both "
                    f"nodes are in the same partition — parity violated"
                )
            else:
                assert hop_count % 2 == 1, (
                    f"dist({node_u},{node_v}) = {hop_count} is even but the "
                    f"nodes are in different partitions — parity violated"
                )


# ═══════════════════════════════════════════════════════════════════════════
# IDEMPOTENCE / DETERMINISM (Test 20)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 20 — Idempotence: running twice yields identical results
# ---------------------------------------------------------------------------

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology))
def test_idempotence(G):
    """
    Property: Calling floyd_warshall (and floyd_warshall_numpy) twice on
    the same unmodified graph yields bit-identical results.

    Mathematical basis: A deterministic algorithm on identical input must
    produce identical output.  Even when multiple shortest paths exist, the
    implementation's tie-breaking is determined by iteration order, which is
    fixed for the same graph object.

    Test strategy: Generate random directed graphs with positive weights.
    Call each algorithm twice and compare outputs exactly.

    Assumptions / preconditions:
      - The graph is not mutated between calls.

    Why failure matters: Different results on repeated calls indicate
    non-determinism -- perhaps from uninitialised state, hash randomisation
    leaking into iteration order, or accidental mutation of the input
    graph.
    """
    dist1 = _fw_dist(G)
    dist2 = _fw_dist(G)
    for u in G.nodes():
        for v in G.nodes():
            assert dist1[u][v] == dist2[u][v], (
                f"Non-determinism: dist1[{u}][{v}]={dist1[u][v]} "
                f"!= dist2[{u}][{v}]={dist2[u][v]}"
            )

    nodelist, A1 = _fw_numpy(G)
    _, A2 = _fw_numpy(G)
    assert np.array_equal(A1, A2), "floyd_warshall_numpy returned different arrays"


# ═══════════════════════════════════════════════════════════════════════════
# BUG DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
#
# Finding: floyd_warshall silently returns invalid results on graphs with
# negative-weight cycles, instead of raising an exception.
#
# NetworkX's other shortest-path algorithms handle negative cycles
# consistently:
#   - nx.single_source_bellman_ford  →  raises NetworkXUnbounded
#   - nx.johnson                     →  raises NetworkXUnbounded
#   - nx.negative_edge_cycle         →  correctly detects and returns True
#
# However, all three Floyd-Warshall variants silently return distance
# dictionaries/matrices with meaningless values (e.g. negative self-
# distances).  The documentation states "This algorithm can still fail if
# there are negative cycles" but does not specify HOW it fails or how
# users should detect the failure.
#
# Source location:
#   networkx/algorithms/shortest_paths/dense.py
#   Function: floyd_warshall_predecessor_and_distance  (line 90)
#   Relaxation loop: lines 160–168
#
#       for w in G:           # line 160
#           dist_w = dist[w]
#           for u in G:
#               dist_u = dist[u]
#               for v in G:
#                   d = dist_u[w] + dist_w[v]
#                   if dist_u[v] > d:
#                       dist_u[v] = d
#                       pred[u][v] = pred[w][v]
#       return dict(pred), dict(dist)   # line 169 — returns without any check
#
# The loop ends and returns at line 169 with no post-loop inspection of
# the diagonal.  A one-line fix would be to add after line 168:
#
#       if any(dist[v][v] < 0 for v in G):
#           raise nx.NetworkXUnbounded(
#               "Negative cycle detected in floyd_warshall."
#           )
#
# This mirrors the check already present in
# networkx/algorithms/shortest_paths/weighted.py (johnson / bellman_ford).
#
# Impact: Users switching from Bellman-Ford to Floyd-Warshall lose their
# negative-cycle protection without any warning.  There is no exception,
# no warning, and no documented detection mechanism.
#
# Verified on: NetworkX 3.6.1, Python 3.12
# ═══════════════════════════════════════════════════════════════════════════


def test_negative_cycle_silent_failure():
    """
    BUG DISCOVERY: Floyd-Warshall silently returns invalid results on
    negative-weight cycles, unlike other NetworkX shortest-path algorithms
    which raise NetworkXUnbounded.

    Evidence
    --------
    Minimal reproducer on a 3-node negative cycle (0→1→2→0, total weight
    = 1 + 1 + (-5) = -3):

        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, -5)])

    Comparative behavior:
        nx.single_source_bellman_ford_path_length(G, 0)
            → raises NetworkXUnbounded("Negative cycle detected.")
        nx.johnson(G)
            → raises NetworkXUnbounded("Negative cycle detected.")
        nx.floyd_warshall(G)
            → silently returns {0: {0: -3, 1: -2, 2: -1}, ...}

    Source location
    ---------------
    File:     networkx/algorithms/shortest_paths/dense.py
    Function: floyd_warshall_predecessor_and_distance  (defined at line 90)
    Loop:     lines 160–168 — the triple-nested DP relaxation
    Return:   line 169 — returns immediately after the loop with no
              diagonal check

    The loop body (lines 160–168) is:

        for w in G:
            dist_w = dist[w]
            for u in G:
                dist_u = dist[u]
                for v in G:
                    d = dist_u[w] + dist_w[v]
                    if dist_u[v] > d:
                        dist_u[v] = d
                        pred[u][v] = pred[w][v]
        return dict(pred), dict(dist)   # ← no negative-cycle check here

    The docstring at line 131 states "This algorithm can still fail if
    there are negative cycles" but does not define what "fail" means, does
    not raise an exception, and does not emit a warning.

    Suggested fix
    -------------
    Insert after line 168 (before the return):

        if any(dist[v][v] < 0 for v in G):
            raise nx.NetworkXUnbounded(
                "Negative cycle detected in floyd_warshall."
            )

    This is a O(n) post-loop diagonal scan — negligible overhead vs the
    O(n^3) main loop.  It mirrors the check already present in
    networkx/algorithms/shortest_paths/weighted.py for Bellman-Ford and
    Johnson's algorithm.

    Tested on: NetworkX 3.6.1, Python 3.12.10

    Why this matters: A user who migrates from Bellman-Ford (which raises
    on negative cycles) to Floyd-Warshall (which doesn't) silently loses
    negative-cycle detection.  The returned distances look structurally
    valid (same dict-of-dict format) but contain meaningless values,
    risking silent data corruption in downstream analysis.
    """
    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, -5)])

    bellman_ford_raised = False
    try:
        dict(nx.single_source_bellman_ford_path_length(G, 0))
    except nx.NetworkXUnbounded:
        bellman_ford_raised = True

    assert bellman_ford_raised, (
        "Expected Bellman-Ford to raise NetworkXUnbounded on negative cycle"
    )

    floyd_warshall_raised = False
    dist = None
    try:
        dist = nx.floyd_warshall(G)
    except (nx.NetworkXUnbounded, nx.NetworkXError):
        floyd_warshall_raised = True

    assert not floyd_warshall_raised, (
        "Floyd-Warshall now raises on negative cycles -- the bug may be "
        "fixed!  Update this test to expect the exception."
    )
    assert dist is not None, "dist was not assigned (unexpected exception path)"

    assert dist[0][0] < 0, (
        "Negative cycle should produce negative self-distance"
    )
    assert dist[1][1] < 0, (
        "All nodes on the cycle should have negative self-distance"
    )
    assert dist[2][2] < 0, (
        "All nodes on the cycle should have negative self-distance"
    )
