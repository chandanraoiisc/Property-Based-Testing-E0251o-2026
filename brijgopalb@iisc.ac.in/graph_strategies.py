"""
Reusable Hypothesis Strategies for Generating NetworkX Graphs
=============================================================

A library of composable graph-generation strategies built on Hypothesis.
Designed to be imported by any property-based test suite that needs diverse
NetworkX graph inputs.

Topology strategies produce unweighted structure; weight strategies assign
edge attributes.  Convenience composites combine both for common use-cases
(positive-weighted digraphs, DAGs with negative weights, etc.).

Usage
-----
    from graph_strategies import positive_weighted_digraph, dag_with_weights
    from hypothesis import given

    @given(G=positive_weighted_digraph())
    def test_something(G):
        ...
"""

import networkx as nx
import hypothesis.strategies as st


# ───────────────────────────────────────────────────────────────────────
# Topology helpers (unweighted structure)
# ───────────────────────────────────────────────────────────────────────

@st.composite
def random_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Erdős–Rényi G(n, p) topology."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    p = draw(st.floats(min_value=0.1, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    if directed:
        G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    else:
        G = nx.gnp_random_graph(n, p, seed=seed, directed=False)
    return G


@st.composite
def complete_graph_topology(draw, min_nodes=2, max_nodes=12, directed=True):
    """Complete graph / complete digraph on n nodes."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    if directed:
        return nx.complete_graph(n, create_using=nx.DiGraph)
    return nx.complete_graph(n)


@st.composite
def path_graph_topology(draw, min_nodes=2, max_nodes=15, directed=True):
    """Simple directed or undirected path 0 → 1 → … → n-1."""
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
    """Random labeled tree on n nodes (Prüfer-sequence based)."""
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

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

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


# ───────────────────────────────────────────────────────────────────────
# Weight-assignment helpers
# ───────────────────────────────────────────────────────────────────────

def _assign_weights(draw, G, min_weight, max_weight):
    """Draw a random integer weight for every edge in G."""
    for u, v in G.edges():
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G[u][v]["weight"] = w
    return G


# ───────────────────────────────────────────────────────────────────────
# Convenience composite strategies (topology + weights)
# ───────────────────────────────────────────────────────────────────────

@st.composite
def positive_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                              min_weight=1, max_weight=50):
    """Directed graph with positive integer weights.

    Uses a random Erdős–Rényi topology.  Suitable for general shortest-path
    testing where negative weights are not needed.
    """
    G = draw(random_graph_topology(
        min_nodes=min_nodes, max_nodes=max_nodes, directed=True))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def nonneg_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                            min_weight=0, max_weight=50):
    """Directed graph with non-negative integer weights (allows zero)."""
    G = draw(random_graph_topology(
        min_nodes=min_nodes, max_nodes=max_nodes, directed=True))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def undirected_nonneg_graph(draw, min_nodes=2, max_nodes=12,
                            min_weight=0, max_weight=50):
    """Undirected graph with non-negative integer weights."""
    G = draw(random_graph_topology(
        min_nodes=min_nodes, max_nodes=max_nodes, directed=False))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def dag_with_weights(draw, min_nodes=2, max_nodes=15,
                     min_weight=-10, max_weight=20):
    """DAG with arbitrary (including negative) integer weights.

    Because a DAG has no cycles at all, negative edge weights are safe for
    shortest-path algorithms—there can never be a negative-weight cycle.
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
    G = draw(random_graph_topology(
        min_nodes=min_nodes, max_nodes=max_nodes, directed=False))
    return _assign_weights(draw, G, min_weight, max_weight)


@st.composite
def mixed_topology_weighted_digraph(draw, min_nodes=2, max_nodes=12,
                                    min_weight=1, max_weight=50):
    """Draw from a mix of topologies for broader coverage."""
    topology_fn = draw(st.sampled_from([
        random_graph_topology,
        complete_graph_topology,
        path_graph_topology,
        cycle_graph_topology,
        star_graph_topology,
        tree_graph_topology,
    ]))
    G = draw(topology_fn(min_nodes=min_nodes, max_nodes=max_nodes, directed=True))
    return _assign_weights(draw, G, min_weight, max_weight)
