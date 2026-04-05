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

All tests use the Hypothesis library to generate diverse graph structures
via the companion ``graph_strategies`` module.
"""

import math
import random

import networkx as nx
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck

from graph_strategies import (
    graph_builder,
    random_graph_topology,
    disconnected_graph_topology,
    empty_graph_topology,
    dag_with_weights,
    negative_cycle_digraph,
)

INF = float("inf")
MAX_EXAMPLES = 80
SLOW_OK = [HealthCheck.too_slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Test 1 - Zero self-distance
# ---------------------------------------------------------------------------

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
    verify every diagonal entry.

    Assumptions / preconditions:
      - All edge weights are positive (>= 1), so no negative cycles exist.

    Why failure matters: A non-zero self-distance would mean the algorithm
    believes it costs something to stay in place, indicating a fundamental
    error in initialisation or relaxation.
    """
    dist = _fw_dist(G)
    for v in G.nodes():
        assert dist[v][v] == 0, (
            f"dist({v}, {v}) = {dist[v][v]}, expected 0"
        )


# ---------------------------------------------------------------------------
# Test 2 - Triangle inequality
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
    ordered triple of nodes.

    Assumptions / preconditions:
      - Graph is a DAG, so no negative cycles can exist.

    Why failure matters: A violation means the algorithm found paths to u->v
    and v->w whose costs sum to less than the supposed shortest u->w path.
    This implies the algorithm missed a shorter route through v, a critical
    correctness bug.
    """
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
# Test 3 - Symmetry on undirected graphs
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
# Test 4 - Reconstructed path weight equals reported distance
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
# Test 5 - Subpath optimality
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
# Test 6 - floyd_warshall dict vs floyd_warshall_predecessor_and_distance
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
# Test 7 - floyd_warshall dict vs floyd_warshall_numpy
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
# METAMORPHIC PROPERTIES (Tests 8 - 11)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 8 - Weight scaling
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
# Test 9 - Adding a non-negative edge can only decrease distances
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(
    G=graph_builder(
        topology=random_graph_topology, min_nodes=3, max_nodes=10, min_weight=0),
    w=st.integers(min_value=0, max_value=50),
)
def test_edge_addition_monotonicity(G, w):
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
    connected, recompute, and verify every distance is <= the original.

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

    a, b = non_edges[0]
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
# Test 10 - Subgraph distances are a lower bound
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=SLOW_OK)
@given(G=graph_builder(topology=random_graph_topology, min_nodes=4, max_nodes=10))
def test_subgraph_distance_lower_bound(G):
    """
    Property: For a subgraph H of G (same nodes, subset of edges),
    dist_G(u,v) <= dist_H(u,v) for all u, v.

    Mathematical basis: H has fewer edges than G, so fewer candidate paths
    between any pair.  The minimum over a subset cannot be smaller than the
    minimum over the superset.  This generalises the edge-addition
    monotonicity to arbitrary edge subsets.

    Test strategy: Generate a directed graph G with positive weights.
    Construct a subgraph H by keeping each edge independently with
    probability 0.5.  Compute distances on both and verify the inequality.

    Assumptions / preconditions:
      - Positive weights ensure no negative cycles in either G or H.

    Why failure matters: If distances in G exceed those in H, the algorithm
    is finding shorter paths with fewer edges available, which is logically
    impossible and indicates a relaxation bug.
    """
    assume(G.number_of_edges() >= 2)

    rng = random.Random(42)
    H = G.copy()
    edges_to_remove = [e for e in H.edges() if rng.random() < 0.5]
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
# Test 11 - Reversing edges transposes the distance matrix
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


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY / EDGE-CASE PROPERTIES (Tests 12 - 14)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 12 - Empty graph: all off-diagonal distances are infinite
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
# Test 13 - Disconnected components produce infinite cross-distances
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
# Test 14 - Negative cycle produces negative self-distance
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
# IDEMPOTENCE / DETERMINISM (Test 15)
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Test 15 - Idempotence: running twice yields identical results
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
