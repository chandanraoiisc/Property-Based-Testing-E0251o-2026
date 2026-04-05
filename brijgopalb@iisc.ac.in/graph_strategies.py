"""
Reusable Hypothesis Strategies for Generating NetworkX Graphs
=============================================================

A library of composable graph-generation strategies built on Hypothesis.
Designed to be imported by any property-based test suite that needs diverse
NetworkX graph inputs.

Architecture
------------
Three layers, each composable with the next:

1. **Topology strategies** produce unweighted graph structure
   (``random_graph_topology``, ``complete_graph_topology``, …).

2. **Modifier strategies** mutate a graph in-place to add structural
   edge-cases: self-loops, isolated nodes, uniform weights, etc.

3. **``graph_builder``** is the single composable entry-point that
   combines a topology strategy + weight range + optional modifiers
   into one call.  Named convenience wrappers
   (``positive_weighted_digraph``, ``dag_with_weights``, …) are thin
   calls to ``graph_builder`` kept for backward compatibility.

Usage
-----
    # Builder style (preferred for new code)
    from graph_strategies import graph_builder, cycle_graph_topology

    @given(G=graph_builder(topology=cycle_graph_topology, min_weight=-5))
    def test_something(G):
        ...

    # Named-wrapper style (still supported)
    from graph_strategies import positive_weighted_digraph
    @given(G=positive_weighted_digraph())
    def test_something(G):
        ...
"""

import networkx as nx
import hypothesis.strategies as st


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 — Topology strategies (unweighted structure)
# ═══════════════════════════════════════════════════════════════════════

@st.composite
def random_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Erdős–Rényi G(n, p) topology."""
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
def single_node_topology(draw, directed=True):
    """Graph with exactly one node and no edges."""
    _ = draw(st.just(None))
    if directed:
        return nx.empty_graph(1, create_using=nx.DiGraph)
    return nx.empty_graph(1)


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
def dag_topology(draw, min_nodes=2, max_nodes=15):
    """Random DAG: edges only go from lower-index to higher-index nodes."""
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
    """Random bipartite graph with two partitions."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    split = draw(st.integers(min_value=1, max_value=n - 1))
    left = range(split)
    right = range(split, n)

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in left:
        for j in right:
            if draw(st.booleans()):
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


# ═══════════════════════════════════════════════════════════════════════
# Layer 2 — Modifier helpers (mutate a graph in-place)
# ═══════════════════════════════════════════════════════════════════════

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
    """Add self-loops with positive weights to a random subset of nodes."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    cap = max_loops if max_loops is not None else len(nodes)
    count = draw(st.integers(min_value=1, max_value=max(1, cap)))
    chosen = draw(st.lists(
        st.sampled_from(nodes), min_size=count, max_size=count, unique=True,
    ).filter(lambda lst: len(lst) == count))
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


# ═══════════════════════════════════════════════════════════════════════
# Layer 3 — Composable builder + named convenience wrappers
# ═══════════════════════════════════════════════════════════════════════

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


# ───────────────────────────────────────────────────────────────────────
# Named convenience wrappers (backward-compatible public API)
# ───────────────────────────────────────────────────────────────────────

@st.composite
def positive_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                              min_weight=1, max_weight=50):
    """Directed graph with positive integer weights (Erdos-Renyi)."""
    return draw(graph_builder(
        topology=random_graph_topology, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight))


@st.composite
def nonneg_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                            min_weight=0, max_weight=50):
    """Directed graph with non-negative integer weights (allows zero)."""
    return draw(graph_builder(
        topology=random_graph_topology, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight))


@st.composite
def undirected_nonneg_graph(draw, min_nodes=2, max_nodes=12,
                            min_weight=0, max_weight=50):
    """Undirected graph with non-negative integer weights."""
    return draw(graph_builder(
        topology=random_graph_topology, directed=False,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight))


@st.composite
def dag_with_weights(draw, min_nodes=2, max_nodes=15,
                     min_weight=-10, max_weight=20):
    """DAG with arbitrary (including negative) integer weights.

    Because a DAG has no cycles at all, negative edge weights are safe for
    shortest-path algorithms -- there can never be a negative-weight cycle.
    """
    G = draw(dag_topology(min_nodes=min_nodes, max_nodes=max_nodes))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def negative_cycle_digraph(draw, min_nodes=3, max_nodes=10):
    """Directed graph guaranteed to contain at least one negative-weight cycle.

    Construction: build a directed cycle on n nodes, assign weights so that
    the total cycle weight is negative, then optionally add extra edges.
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


@st.composite
def disconnected_weighted_digraph(draw, min_nodes=4, max_nodes=14,
                                  min_weight=1, max_weight=50):
    """Directed graph with multiple components and positive weights."""
    G = draw(disconnected_graph_topology(
        min_nodes=min_nodes, max_nodes=max_nodes, directed=True))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def positive_weighted_undirected(draw, min_nodes=2, max_nodes=12,
                                 min_weight=1, max_weight=50):
    """Undirected graph with positive integer weights."""
    return draw(graph_builder(
        topology=random_graph_topology, directed=False,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight))


@st.composite
def mixed_topology_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                                    min_weight=1, max_weight=50):
    """Draw from a mix of topologies for broader coverage."""
    return draw(graph_builder(
        topology=None, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight))


# ───────────────────────────────────────────────────────────────────────
# Hybrid edge-case strategies
# ───────────────────────────────────────────────────────────────────────

@st.composite
def graph_with_self_loops(draw, min_nodes=2, max_nodes=10,
                          min_weight=1, max_weight=50):
    """Directed graph with positive-weight self-loops on some nodes.

    Self-loops should not change shortest-path distances (dist(v,v) stays 0
    when all self-loop weights are positive).
    """
    return draw(graph_builder(
        topology=None, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight,
        self_loops=True))


@st.composite
def uniform_weight_digraph(draw, min_nodes=2, max_nodes=12,
                           min_weight=1, max_weight=50):
    """Directed graph where every edge has the same weight.

    When all edges have weight w, shortest-path distance equals
    w * (hop count of the BFS shortest path).
    """
    return draw(graph_builder(
        topology=None, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight,
        uniform_weight=True))


@st.composite
def graph_with_isolated_nodes(draw, min_nodes=3, max_nodes=10,
                              min_weight=1, max_weight=50):
    """Directed graph with guaranteed isolated vertices (degree 0).

    Isolated nodes must have dist(v, u) = inf for all u != v and
    dist(v, v) = 0.
    """
    return draw(graph_builder(
        topology=random_graph_topology, directed=True,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_weight=min_weight, max_weight=max_weight,
        isolated_nodes=True))
