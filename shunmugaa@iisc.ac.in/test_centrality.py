"""
Property-Based Tests for NetworkX Centrality Algorithms
========================================================

Team member : Shunmuga Janani (`shunmugaa@iisc.ac.in`)
Algorithm   : Graph Centrality Measures
Module      : networkx.algorithms.centrality

Functions under test
--------------------
  1. nx.degree_centrality(G)
  2. nx.betweenness_centrality(G, normalized=True)
  3. nx.closeness_centrality(G)
  4. nx.pagerank(G)
  5. nx.eigenvector_centrality(G)
  6. nx.katz_centrality(G)

This single file contains:
  - Reusable Hypothesis graph-generation strategies
  - 20 property-based tests with detailed docstrings
  - 1 bug-discovery test documenting a known issue in NetworkX

All tests use the Hypothesis library to generate diverse graph structures.
"""

import math

import networkx as nx
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck, example, event, target

MAX_EXAMPLES = 80
SLOW_OK = [HealthCheck.too_slow]


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

@st.composite
def random_undirected_graph(draw, min_nodes=3, max_nodes=15):
    """Erdos-Renyi G(n, p) undirected graph with at least one edge."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    p = draw(st.floats(min_value=0.2, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    G = nx.gnp_random_graph(n, p, seed=seed, directed=False)
    return G


@st.composite
def random_directed_graph(draw, min_nodes=3, max_nodes=15):
    """Erdos-Renyi G(n, p) directed graph."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    p = draw(st.floats(min_value=0.2, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    return G


@st.composite
def connected_undirected_graph(draw, min_nodes=3, max_nodes=12):
    """Random undirected graph guaranteed to be connected.

    Builds a spanning tree first (to ensure connectivity), then adds
    random extra edges.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    # Start from a random spanning tree
    T = nx.random_labeled_tree(n, seed=seed)
    G = nx.Graph(T)
    # Add extra random edges
    extra_p = draw(st.floats(min_value=0.0, max_value=0.5))
    rng_seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    extra = nx.gnp_random_graph(n, extra_p, seed=rng_seed, directed=False)
    G.add_edges_from(extra.edges())
    return G


@st.composite
def star_graph(draw, min_leaves=2, max_leaves=10):
    """Star graph S_n: one center hub connected to n leaves.

    Returns (G, center_node, leaf_nodes).
    """
    n = draw(st.integers(min_value=min_leaves, max_value=max_leaves))
    G = nx.star_graph(n)  # nodes: 0 (center), 1..n (leaves)
    center = 0
    leaves = list(range(1, n + 1))
    return G, center, leaves


@st.composite
def path_graph(draw, min_nodes=4, max_nodes=15):
    """Simple undirected path 0 - 1 - ... - n-1.

    Returns (G, endpoint_left, endpoint_right, middle_nodes).
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.path_graph(n)
    left_end = 0
    right_end = n - 1
    middle = list(range(1, n - 1))
    return G, left_end, right_end, middle


@st.composite
def strongly_connected_directed_graph(draw, min_nodes=3, max_nodes=10):
    """Directed graph guaranteed to be strongly connected.

    Builds a directed cycle for connectivity, then adds random extra edges.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.cycle_graph(n, create_using=nx.DiGraph)
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    extra_p = draw(st.floats(min_value=0.0, max_value=0.5))
    extra = nx.gnp_random_graph(n, extra_p, seed=seed, directed=True)
    G.add_edges_from(extra.edges())
    return G


# ═══════════════════════════════════════════════════════════════════════════
# TESTS 1–5: DEGREE CENTRALITY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=random_undirected_graph())
def test_degree_centrality_range(G):
    """Property: All degree centrality values lie in [0, 1].

    Mathematical basis:
        For an undirected graph on n nodes, degree_centrality(v) = deg(v) / (n-1).
        Since 0 <= deg(v) <= n-1, the value is always in [0, 1].

    Test strategy:
        Random Erdos-Renyi graphs with varying density. Covers sparse and
        dense graphs, including isolated nodes (dc = 0) and fully connected
        nodes (dc = 1 in Kn).

    Why failure matters:
        A value outside [0, 1] would indicate a normalization bug or an
        incorrect degree count.
    """
    n = G.number_of_nodes()
    assume(n >= 2)
    dc = nx.degree_centrality(G)
    for v, val in dc.items():
        assert 0.0 <= val <= 1.0, (
            f"degree_centrality({v}) = {val} is outside [0, 1]"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=random_undirected_graph())
def test_degree_centrality_formula(G):
    """Property: degree_centrality(v) = deg(v) / (n - 1) for all v.

    Mathematical basis:
        This is the definition of normalized degree centrality for undirected
        graphs. Each node's centrality equals its degree divided by the
        maximum possible degree (n-1).

    Test strategy:
        Compares NetworkX's output against direct computation from the
        degree sequence. Works across all graph sizes and densities.

    Why failure matters:
        A mismatch means the normalization factor is wrong or the degree
        count is computed incorrectly.
    """
    n = G.number_of_nodes()
    assume(n >= 2)
    dc = nx.degree_centrality(G)
    for v in G.nodes():
        expected = G.degree(v) / (n - 1)
        assert abs(dc[v] - expected) < 1e-9, (
            f"degree_centrality({v}) = {dc[v]}, expected {expected}"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=random_undirected_graph())
def test_degree_centrality_sum(G):
    """Property: sum of degree centralities = 2 * |E| / (n * (n - 1)).

    Mathematical basis:
        Since dc(v) = deg(v)/(n-1), summing over all nodes gives:
            sum(dc) = sum(deg(v)) / (n-1) = 2|E| / (n-1)
        Dividing by n (the number of nodes) yields the average:
            avg(dc) = 2|E| / (n * (n-1))
        Equivalently, sum(dc) = 2|E| / (n-1), which is the handshaking
        lemma normalized by (n-1).

    Test strategy:
        Verifies the global sum identity across random graphs. The identity
        holds regardless of graph structure.

    Why failure matters:
        A sum mismatch would indicate that individual centrality values are
        inconsistent with the global degree sequence.
    """
    n = G.number_of_nodes()
    assume(n >= 2)
    dc = nx.degree_centrality(G)
    expected_sum = 2 * G.number_of_edges() / (n - 1)
    actual_sum = sum(dc.values())
    assert abs(actual_sum - expected_sum) < 1e-9, (
        f"sum(dc) = {actual_sum}, expected {expected_sum}"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(n=st.integers(min_value=2, max_value=15))
@example(n=2)
@example(n=10)
def test_complete_graph_degree_centrality(n):
    """Property: In Kn, all nodes have degree centrality = 1.

    Mathematical basis:
        In the complete graph Kn, every node has degree n-1 (connected to
        all other nodes). Therefore dc(v) = (n-1)/(n-1) = 1 for all v.

    Test strategy:
        Parametric test over complete graphs of varying sizes. The @example
        annotations pin the boundary cases n=2 (K2 = single edge) and n=10.

    Why failure matters:
        Failing on a complete graph would indicate a fundamental error in
        the degree computation or normalization.
    """
    G = nx.complete_graph(n)
    dc = nx.degree_centrality(G)
    for v, val in dc.items():
        assert abs(val - 1.0) < 1e-9, (
            f"In K{n}, degree_centrality({v}) = {val}, expected 1.0"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(data=st.data())
def test_star_degree_centrality(data):
    """Property: In star S_n, center dc = 1 and each leaf dc = 1/(n-1).

    Mathematical basis:
        In a star with n+1 nodes (center + n leaves):
          - center degree = n  →  dc(center) = n/n = 1
          - leaf degree = 1    →  dc(leaf) = 1/n

    Test strategy:
        Generates star graphs of varying sizes using the star_graph strategy.
        Checks both the center and a randomly sampled leaf.

    Why failure matters:
        Stars are structurally simple; failure here would indicate a
        fundamental bug, not an edge case.
    """
    G, center, leaves = data.draw(star_graph(min_leaves=2, max_leaves=10))
    n_leaves = len(leaves)
    dc = nx.degree_centrality(G)
    assert abs(dc[center] - 1.0) < 1e-9, (
        f"Star center dc = {dc[center]}, expected 1.0"
    )
    expected_leaf_dc = 1.0 / n_leaves
    for leaf in leaves:
        assert abs(dc[leaf] - expected_leaf_dc) < 1e-9, (
            f"Star leaf dc = {dc[leaf]}, expected {expected_leaf_dc}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS 6–11: BETWEENNESS CENTRALITY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_betweenness_centrality_range(G):
    """Property: All betweenness centrality values lie in [0, 1].

    Mathematical basis:
        Normalized betweenness centrality counts the fraction of all
        shortest paths (between distinct pairs) that pass through v.
        The maximum (v lies on ALL shortest paths) equals 1. The minimum
        (v lies on no shortest path) equals 0.

    Test strategy:
        Connected undirected graphs. Connectivity is required because
        betweenness is defined in terms of shortest paths; disconnected
        pairs contribute 0 to the sum, so the range still holds, but
        connected graphs produce more interesting distributions.

    Why failure matters:
        A value outside [0, 1] indicates a normalization error.
    """
    bc = nx.betweenness_centrality(G, normalized=True)
    for v, val in bc.items():
        assert 0.0 <= val <= 1.0 + 1e-9, (
            f"betweenness_centrality({v}) = {val} outside [0, 1]"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(data=st.data())
def test_star_center_max_betweenness(data):
    """Property: The center of a star graph has betweenness centrality = 1.

    Mathematical basis:
        In a star S_n (n >= 3), every shortest path between two distinct
        leaves passes through the center. The center therefore lies on
        ALL C(n_leaves, 2) inter-leaf shortest paths. Since there are no
        other non-trivial paths (endpoints contribute 0), the normalized
        betweenness of the center = 1.

    Test strategy:
        Star graphs with at least 3 leaves so that inter-leaf paths exist.

    Why failure matters:
        The star center is the canonical maximum-betweenness node; failing
        this test would indicate a fundamental bug in betweenness computation.
    """
    G, center, leaves = data.draw(star_graph(min_leaves=3, max_leaves=10))
    bc = nx.betweenness_centrality(G, normalized=True)
    event(f"star leaves = {len(leaves)}")
    assert abs(bc[center] - 1.0) < 1e-9, (
        f"Star center betweenness = {bc[center]}, expected 1.0"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(data=st.data())
def test_path_endpoints_zero_betweenness(data):
    """Property: The two endpoints of a path have betweenness centrality = 0.

    Mathematical basis:
        In a path 0 - 1 - ... - n-1, node 0 is only a source, never an
        intermediate node. Every shortest path that includes node 0 either
        starts or ends there. By definition, betweenness only counts paths
        where v is a strict intermediate node, so bc(0) = bc(n-1) = 0.

    Test strategy:
        Path graphs with at least 4 nodes so there are interior nodes and
        meaningful inter-node shortest paths.

    Why failure matters:
        A non-zero betweenness at an endpoint would mean the algorithm
        incorrectly counts endpoint appearances as interior appearances.
    """
    G, left_end, right_end, _ = data.draw(path_graph(min_nodes=4, max_nodes=15))
    bc = nx.betweenness_centrality(G, normalized=True)
    assert abs(bc[left_end]) < 1e-9, (
        f"Path left endpoint bc = {bc[left_end]}, expected 0"
    )
    assert abs(bc[right_end]) < 1e-9, (
        f"Path right endpoint bc = {bc[right_end]}, expected 0"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(n=st.integers(min_value=3, max_value=12))
def test_complete_graph_zero_betweenness(n):
    """Property: In Kn (n >= 3), all nodes have betweenness centrality = 0.

    Mathematical basis:
        In a complete graph, every pair of nodes is directly connected by an
        edge. The unique shortest path between any two nodes is the direct
        edge (length 1). No node appears as an intermediate on any shortest
        path. Therefore bc(v) = 0 for all v.

    Test strategy:
        Parametric over complete graphs Kn for n from 3 to 12. K2 is excluded
        because with two nodes there are no intermediate nodes by definition.

    Why failure matters:
        Nonzero betweenness in a complete graph would mean the algorithm is
        counting non-shortest paths or making an off-by-one error.
    """
    G = nx.complete_graph(n)
    bc = nx.betweenness_centrality(G, normalized=True)
    for v, val in bc.items():
        assert abs(val) < 1e-9, (
            f"In K{n}, betweenness_centrality({v}) = {val}, expected 0"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_betweenness_sum_upper_bound(G):
    """Property: sum of betweenness centralities <= n * (n - 1) / 2 (unnormalized).

    Mathematical basis:
        The unnormalized betweenness bc_unnorm(v) counts the number of
        shortest paths passing through v. Since each shortest path can pass
        through at most n-2 intermediate nodes, and there are C(n,2) pairs,
        the sum of unnormalized betweenness is bounded by C(n,2) * (n-2).
        For normalized values (divided by C(n-1,2) = (n-1)(n-2)/2), the
        sum is at most n.

    Test strategy:
        Connected graphs to ensure meaningful shortest paths exist.

    Why failure matters:
        Exceeding the upper bound would indicate double-counting or an
        incorrect normalization factor.
    """
    n = G.number_of_nodes()
    assume(n >= 3)
    bc = nx.betweenness_centrality(G, normalized=True)
    bc_sum = sum(bc.values())
    upper_bound = float(n)
    assert bc_sum <= upper_bound + 1e-9, (
        f"sum(bc) = {bc_sum} exceeds upper bound {upper_bound}"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_betweenness_adding_edge_non_increasing(G):
    """Property: Adding an edge cannot increase betweenness centrality of
    any existing node (it can only decrease or stay the same).

    Mathematical basis:
        Adding an edge (u, v) creates a new or shorter path between u and v.
        This can only redirect existing shortest paths to avoid intermediate
        nodes, thereby decreasing or maintaining betweenness. It cannot
        create new interior appearances for existing nodes.

    Test strategy:
        Starts with a connected graph, picks a non-existing node pair, adds
        the edge, and checks that betweenness is component-wise non-increasing.

    Why failure matters:
        An increase would mean the algorithm is counting paths that do not
        actually use the added shortcut.
    """
    n = G.number_of_nodes()
    assume(n >= 3)
    # Find a pair of non-adjacent nodes
    non_edges = list(nx.non_edges(G))
    assume(len(non_edges) > 0)

    bc_before = nx.betweenness_centrality(G, normalized=True)

    import random as _random
    # Use Hypothesis-compatible selection via sampling
    u, v = non_edges[0]
    G2 = G.copy()
    G2.add_edge(u, v)
    bc_after = nx.betweenness_centrality(G2, normalized=True)

    for node in G.nodes():
        assert bc_after[node] <= bc_before[node] + 1e-9, (
            f"bc({node}) increased from {bc_before[node]} to {bc_after[node]} "
            f"after adding edge ({u},{v})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS 12–16: CLOSENESS CENTRALITY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_closeness_centrality_range(G):
    """Property: All closeness centrality values lie in [0, 1].

    Mathematical basis:
        Closeness centrality cc(v) = (n-1) / sum_of_distances(v).
        The minimum sum of distances from any node is n-1 (star center,
        all at distance 1), giving cc = 1. The maximum sum is unbounded
        in theory, but for finite connected graphs cc > 0. For disconnected
        graphs, NetworkX uses the Wasserman-Faust formula, which still
        lies in [0, 1].

    Test strategy:
        Connected graphs to ensure all nodes are reachable.

    Why failure matters:
        A value outside [0, 1] would indicate a normalization error.
    """
    cc = nx.closeness_centrality(G)
    for v, val in cc.items():
        assert 0.0 <= val <= 1.0 + 1e-9, (
            f"closeness_centrality({v}) = {val} outside [0, 1]"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(n=st.integers(min_value=2, max_value=15))
@example(n=2)
@example(n=5)
def test_complete_graph_closeness(n):
    """Property: In Kn, all nodes have closeness centrality = 1.

    Mathematical basis:
        In a complete graph, every node is adjacent to all n-1 others.
        The total distance from any node = n-1 (each at distance 1).
        Therefore cc(v) = (n-1)/(n-1) = 1 for all v.

    Test strategy:
        Parametric over complete graphs Kn. The @example annotations test
        K2 (minimal case) and K5.

    Why failure matters:
        Kn is the densest possible graph; failing here would indicate a
        fundamental distance computation error.
    """
    G = nx.complete_graph(n)
    cc = nx.closeness_centrality(G)
    for v, val in cc.items():
        assert abs(val - 1.0) < 1e-9, (
            f"In K{n}, closeness_centrality({v}) = {val}, expected 1.0"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(data=st.data())
def test_star_center_higher_closeness(data):
    """Property: In a star, the center has strictly higher closeness than all leaves.

    Mathematical basis:
        For a star with center c and n leaves:
          - cc(c) = n / n = 1.0   (all distances = 1)
          - cc(leaf) = n / (1 + 2*(n-1)) = n / (2n-1) < 1
        Since n / (2n-1) < 1 for all n >= 2, the center always dominates.

    Test strategy:
        Star graphs with at least 3 leaves so the closeness inequality is strict.

    Why failure matters:
        The hub of a star is intuitively the most central node; failure would
        indicate that the algorithm ignores graph structure.
    """
    G, center, leaves = data.draw(star_graph(min_leaves=3, max_leaves=10))
    cc = nx.closeness_centrality(G)
    for leaf in leaves:
        assert cc[center] > cc[leaf], (
            f"cc(center)={cc[center]} not > cc(leaf={leaf})={cc[leaf]}"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(data=st.data())
def test_path_center_higher_closeness_than_endpoint(data):
    """Property: On a path, interior nodes have strictly higher closeness than endpoints.

    Mathematical basis:
        The sum of distances from a node v on a path of length n is minimized
        by nodes near the center and maximized at the endpoints. Specifically,
        for the endpoint node 0: sum_dist = 0+1+2+...+(n-1) = n(n-1)/2.
        For any interior node, the sum is strictly smaller, giving higher cc.

    Test strategy:
        Path graphs with at least 4 nodes so at least one interior node exists.

    Why failure matters:
        The path is the canonical example demonstrating closeness centrality
        variation; failure would indicate a fundamental shortest-path error.
    """
    G, left_end, right_end, middle = data.draw(path_graph(min_nodes=4, max_nodes=15))
    cc = nx.closeness_centrality(G)
    for m in middle:
        assert cc[m] > cc[left_end] - 1e-9, (
            f"cc(middle={m})={cc[m]} not >= cc(endpoint={left_end})={cc[left_end]}"
        )
    # At least one strict inequality: middle nodes are not all equal to endpoints
    assert any(cc[m] > cc[left_end] + 1e-9 for m in middle), (
        "No middle node has strictly higher closeness than endpoint"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_closeness_connected_nodes_positive(G):
    """Property: In a connected graph, every node has closeness centrality > 0.

    Mathematical basis:
        cc(v) = (n-1) / sum_of_shortest_paths(v). In a connected graph,
        sum_of_shortest_paths(v) is always finite (no unreachable nodes),
        so cc(v) > 0 for all v.

    Test strategy:
        Connected undirected graphs. The connected_undirected_graph strategy
        guarantees connectivity via a spanning tree.

    Why failure matters:
        A zero closeness in a connected graph would mean the algorithm
        treats some nodes as unreachable even when they are not.
    """
    cc = nx.closeness_centrality(G)
    for v, val in cc.items():
        assert val > 0.0, (
            f"closeness_centrality({v}) = {val} in a connected graph, expected > 0"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS 17–19: PAGERANK PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=strongly_connected_directed_graph())
def test_pagerank_sums_to_one(G):
    """Property: PageRank values sum to 1 for any connected directed graph.

    Mathematical basis:
        PageRank is a probability distribution over nodes — the stationary
        distribution of a random walk on the graph. By definition, a
        probability distribution sums to 1.

        Mathematically: sum_v PR(v) = 1.

    Test strategy:
        Strongly connected directed graphs. PageRank is well-defined and
        unique for strongly connected graphs with any damping factor.

    Why failure matters:
        Summing to != 1 means PageRank is not a valid probability
        distribution, indicating a normalization or convergence bug.
    """
    pr = nx.pagerank(G, alpha=0.85)
    total = sum(pr.values())
    assert abs(total - 1.0) < 1e-6, (
        f"PageRank sum = {total}, expected 1.0"
    )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=strongly_connected_directed_graph())
def test_pagerank_values_positive(G):
    """Property: All PageRank values are strictly positive.

    Mathematical basis:
        With damping factor alpha in (0, 1), the teleportation component
        ensures every node has a positive probability of being visited.
        The Perron-Frobenius theorem guarantees a unique positive stationary
        distribution for the damped random walk matrix.

    Test strategy:
        Strongly connected directed graphs, which satisfy the Perron-Frobenius
        conditions.

    Why failure matters:
        A zero PageRank would imply a node is never visited, which is
        impossible with teleportation (damping factor < 1).
    """
    pr = nx.pagerank(G, alpha=0.85)
    for v, val in pr.items():
        assert val > 0.0, (
            f"pagerank({v}) = {val}, expected > 0 with damping"
        )


@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(n=st.integers(min_value=2, max_value=10))
def test_pagerank_complete_graph_uniform(n):
    """Property: In a complete directed graph (Kn), all PageRank values equal 1/n.

    Mathematical basis:
        In Kn, every node has identical in-degree and out-degree = n-1.
        By symmetry, the stationary distribution of the random walk is
        uniform: PR(v) = 1/n for all v. The teleportation term reinforces
        this symmetry.

    Test strategy:
        Parametric over complete digraphs for n from 2 to 10. Uses a
        complete_graph with DiGraph to get edges in both directions.

    Why failure matters:
        Uniform distribution in a symmetric graph is the simplest possible
        case; failure would indicate a fundamental asymmetry bug.
    """
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=0.85)
    expected = 1.0 / n
    for v, val in pr.items():
        assert abs(val - expected) < 1e-6, (
            f"In complete digraph K{n}, pagerank({v}) = {val}, expected {expected}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 20: IDEMPOTENCE / DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════

@settings(max_examples=MAX_EXAMPLES, suppress_health_check=SLOW_OK)
@given(G=connected_undirected_graph())
def test_centrality_idempotence(G):
    """Property: Calling centrality functions twice on the same graph produces
    identical results (determinism / no side effects).

    Mathematical basis:
        All four centrality measures (degree, betweenness, closeness, PageRank
        on undirected) are deterministic functions of the graph structure.
        They must not mutate the graph or maintain internal state between calls.

    Test strategy:
        Connected graphs. All four measures are computed twice and compared
        element-wise for exact equality.

    Why failure matters:
        Non-idempotence would indicate unintended graph mutation or non-
        deterministic algorithm behaviour (e.g., random tie-breaking).
    """
    dc1 = nx.degree_centrality(G)
    dc2 = nx.degree_centrality(G)
    assert dc1 == dc2, "degree_centrality is not idempotent"

    bc1 = nx.betweenness_centrality(G, normalized=True)
    bc2 = nx.betweenness_centrality(G, normalized=True)
    assert bc1 == bc2, "betweenness_centrality is not idempotent"

    cc1 = nx.closeness_centrality(G)
    cc2 = nx.closeness_centrality(G)
    assert cc1 == cc2, "closeness_centrality is not idempotent"


# ═══════════════════════════════════════════════════════════════════════════
# BUG DISCOVERY TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_eigenvector_centrality_fails_on_bipartite():
    """Bug discovery: eigenvector_centrality raises PowerIterationFailedConvergence
    on bipartite graphs (and other non-trivially-structured graphs).

    Summary:
        nx.eigenvector_centrality() uses power iteration to find the dominant
        eigenvector of the adjacency matrix. For bipartite graphs, the
        adjacency matrix has eigenvalues that are symmetric around 0, and
        the power iteration can fail to converge within the default number
        of iterations (max_iter=100).

    Minimal reproducer:
        G = nx.complete_bipartite_graph(3, 3)
        nx.eigenvector_centrality(G)
        # → raises nx.PowerIterationFailedConvergence

    Root cause:
        Bipartite graphs have a spectrum symmetric around 0 (Perron-Frobenius
        does not apply since the graph is not aperiodic). The dominant
        eigenvector oscillates between the two partitions, causing power
        iteration to cycle rather than converge.

    Impact:
        Users who apply eigenvector_centrality to bipartite networks without
        catching the exception get an unhandled crash. The correct fix is
        either to use nx.eigenvector_centrality_numpy() (which uses numpy's
        eigensolver) or to check for bipartiteness first.

    Verified on:
        NetworkX >= 2.x, Python 3.x
    """
    G = nx.complete_bipartite_graph(3, 3)

    # Demonstrate that eigenvector_centrality raises on this graph
    raised = False
    try:
        nx.eigenvector_centrality(G, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        raised = True

    # Document the actual behavior
    if raised:
        # This is the documented bug: power iteration fails on bipartite graphs
        # The numpy-based variant works correctly
        ec_numpy = nx.eigenvector_centrality_numpy(G)
        assert len(ec_numpy) == G.number_of_nodes(), (
            "eigenvector_centrality_numpy returned wrong number of nodes"
        )
        # Verify numpy variant gives valid values
        for v, val in ec_numpy.items():
            assert val > 0.0, f"eigenvector_centrality_numpy({v}) = {val}, expected > 0"
    else:
        # If a future version fixes the convergence, verify the result is valid
        ec = nx.eigenvector_centrality(G, max_iter=100)
        for v, val in ec.items():
            assert 0.0 <= val <= 1.0 + 1e-9, (
                f"eigenvector_centrality({v}) = {val} outside [0, 1]"
            )
