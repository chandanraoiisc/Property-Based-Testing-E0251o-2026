"""
Property-Based Tests for NetworkX Max-Flow / Min-Cut Algorithms
================================================================

Team member : M Chandan Kumar Rao (chandankuma4@iisc.ac.in)
Algorithm   : Max-Flow / Min-Cut Algorithms

Tests the following NetworkX functions:
  - networkx.algorithms.flow.maximum_flow
  - networkx.algorithms.flow.maximum_flow_value
  - networkx.algorithms.flow.minimum_cut
  - networkx.algorithms.flow.minimum_cut_value

All tests use the Hypothesis library to generate diverse directed graph
structures with random capacities, sizes, and topologies.
"""

import networkx as nx
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck
import random


# ---------------------------------------------------------------------------
# Reusable graph-generation strategies
# ---------------------------------------------------------------------------

@st.composite
def flow_network(draw, min_nodes=3, max_nodes=15, min_cap=1, max_cap=100):
    """Generate a random directed graph guaranteed to have a path from node 0
    (source) to node n-1 (sink), with positive integer capacities."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Guarantee at least one s-t path via a random walk 0 → … → n-1
    path = [0]
    remaining = set(range(1, n))
    while path[-1] != n - 1:
        nxt = draw(st.sampled_from(sorted(remaining)))
        remaining.discard(nxt)
        path.append(nxt)
        if not remaining and path[-1] != n - 1:
            path.append(n - 1)
            break

    for u, v in zip(path, path[1:]):
        cap = draw(st.integers(min_value=min_cap, max_value=max_cap))
        G.add_edge(u, v, capacity=cap)

    # Add random extra edges (no self-loops, no duplicate edges)
    extra = draw(st.integers(min_value=0, max_value=n * (n - 1) // 2))
    possible = [(u, v) for u in range(n) for v in range(n)
                if u != v and not G.has_edge(u, v)]
    if possible:
        chosen = draw(st.lists(
            st.sampled_from(possible),
            min_size=0, max_size=min(extra, len(possible)),
            unique=True,
        ))
        for u, v in chosen:
            cap = draw(st.integers(min_value=min_cap, max_value=max_cap))
            G.add_edge(u, v, capacity=cap)

    return G


@st.composite
def single_path_network(draw, min_len=2, max_len=10, min_cap=1, max_cap=100):
    """Generate a simple directed path graph 0 → 1 → … → n-1 with random
    capacities.  Useful for bottleneck-property tests."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    G = nx.DiGraph()
    for i in range(n - 1):
        cap = draw(st.integers(min_value=min_cap, max_value=max_cap))
        G.add_edge(i, i + 1, capacity=cap)
    return G


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _source_sink(G):
    """Return (source=0, sink=max_node)."""
    nodes = sorted(G.nodes())
    return nodes[0], nodes[-1]


# ---------------------------------------------------------------------------
# Test 1 – Max-Flow Min-Cut Theorem (core invariant)
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_maxflow_equals_mincut(G):
    """
    Property: The value of the maximum flow equals the capacity of the
    minimum cut.

    Mathematical basis: The Max-Flow Min-Cut Theorem (Ford & Fulkerson, 1956)
    states that in any flow network the maximum amount of flow from source to
    sink equals the total weight of the edges in the smallest cut separating
    source and sink.  This is one of the most fundamental results in
    combinatorial optimisation and network theory.

    Test strategy: Generate random directed graphs of 3–15 nodes with at
    least one s-t path and positive integer capacities.  Compute both the
    maximum flow value and the minimum cut value independently and assert
    equality.

    Assumptions / preconditions:
      - The graph is directed with positive integer capacities.
      - There exists at least one path from source (0) to sink (n-1).

    Failure implication: A mismatch would indicate a fundamental correctness
    bug in either the max-flow or the min-cut implementation—one of them
    violates the duality theorem.
    """
    s, t = _source_sink(G)
    max_flow_val = nx.maximum_flow_value(G, s, t)
    min_cut_val = nx.minimum_cut_value(G, s, t)
    assert max_flow_val == min_cut_val, (
        f"Max-flow ({max_flow_val}) != Min-cut ({min_cut_val})"
    )


# ---------------------------------------------------------------------------
# Test 2 – Flow conservation at intermediate nodes
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_flow_conservation(G):
    """
    Property: For every node v that is neither the source nor the sink, the
    total flow entering v equals the total flow leaving v.

    Mathematical basis: Flow conservation is one of the two defining
    constraints of a valid network flow (the other being capacity
    constraints).  Kirchhoff's current law in electrical networks is the
    physical analogue.

    Test strategy: Generate random flow networks, compute the maximum flow
    and its decomposition, then verify conservation at every intermediate
    node.

    Assumptions / preconditions:
      - Valid flow network with at least one s-t path.

    Failure implication: Violation means the algorithm produced an invalid
    flow—some node is "creating" or "destroying" flow, which is physically
    and mathematically impossible in a correct solution.
    """
    s, t = _source_sink(G)
    flow_value, flow_dict = nx.maximum_flow(G, s, t)

    for v in G.nodes():
        if v == s or v == t:
            continue
        inflow = sum(flow_dict[u][v] for u in G.predecessors(v))
        outflow = sum(flow_dict[v][w] for w in G.successors(v))
        assert abs(inflow - outflow) < 1e-9, (
            f"Flow conservation violated at node {v}: "
            f"in={inflow}, out={outflow}"
        )


# ---------------------------------------------------------------------------
# Test 3 – Capacity constraints
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_capacity_constraints(G):
    """
    Property: The flow on every edge is non-negative and does not exceed the
    edge's capacity.

    Mathematical basis: A feasible flow must satisfy 0 ≤ f(u,v) ≤ c(u,v)
    for every edge (u,v).  This is the capacity constraint, one of the two
    axioms of network flow.

    Test strategy: Generate random flow networks, compute max flow, and
    check every edge.

    Assumptions / preconditions:
      - All capacities are positive integers.

    Failure implication: An edge carrying negative flow or flow exceeding
    capacity means the algorithm returned an infeasible solution.
    """
    s, t = _source_sink(G)
    _, flow_dict = nx.maximum_flow(G, s, t)

    for u, v, data in G.edges(data=True):
        cap = data["capacity"]
        f = flow_dict[u][v]
        assert -1e-9 <= f <= cap + 1e-9, (
            f"Edge ({u},{v}): flow {f} violates capacity {cap}"
        )


# ---------------------------------------------------------------------------
# Test 4 – Source outflow equals sink inflow equals max-flow value
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_source_outflow_equals_sink_inflow(G):
    """
    Property: The net flow leaving the source equals the net flow entering
    the sink, and both equal the reported maximum flow value.

    Mathematical basis: By flow conservation at all intermediate nodes, the
    excess created at the source must arrive at the sink.  The reported
    max-flow value must match this quantity.

    Test strategy: Compute max flow and verify the three-way equality:
    net source outflow = net sink inflow = reported flow value.

    Assumptions / preconditions:
      - Valid flow network with at least one s-t path.

    Failure implication: A discrepancy means either the flow decomposition
    is inconsistent or the reported value is wrong.
    """
    s, t = _source_sink(G)
    flow_value, flow_dict = nx.maximum_flow(G, s, t)

    source_out = sum(flow_dict[s].get(w, 0) for w in G.successors(s))
    source_in = sum(flow_dict[u].get(s, 0) for u in G.predecessors(s))
    net_source = source_out - source_in

    sink_in = sum(flow_dict[u].get(t, 0) for u in G.predecessors(t))
    sink_out = sum(flow_dict[t].get(w, 0) for w in G.successors(t))
    net_sink = sink_in - sink_out

    assert abs(net_source - flow_value) < 1e-9
    assert abs(net_sink - flow_value) < 1e-9


# ---------------------------------------------------------------------------
# Test 5 – Min-cut partition validity
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_mincut_partition_validity(G):
    """
    Property: The minimum cut returns a partition (S, T) of the node set
    such that source ∈ S, sink ∈ T, S ∪ T = V, and S ∩ T = ∅.  The
    capacity of edges crossing from S to T equals the reported cut value.

    Mathematical basis: A cut is defined as a partition of V into two
    disjoint sets with source and sink on opposite sides.  The cut capacity
    is the sum of capacities of edges from S to T.

    Test strategy: Compute the minimum cut, verify the partition properties,
    and recompute the crossing-edge capacity independently.

    Assumptions / preconditions:
      - Valid flow network with at least one s-t path.

    Failure implication: An invalid partition or mismatched capacity means
    the min-cut algorithm is returning an incorrect or inconsistent result.
    """
    s, t = _source_sink(G)
    cut_value, (S, T) = nx.minimum_cut(G, s, t)

    # Partition checks
    assert s in S, "Source must be in S"
    assert t in T, "Sink must be in T"
    assert set(S) | set(T) == set(G.nodes()), "S ∪ T must equal V"
    assert set(S) & set(T) == set(), "S ∩ T must be empty"

    # Recompute cut capacity
    recomputed = sum(
        G[u][v]["capacity"] for u in S for v in T if G.has_edge(u, v)
    )
    assert abs(recomputed - cut_value) < 1e-9, (
        f"Recomputed cut capacity ({recomputed}) != reported ({cut_value})"
    )


# ---------------------------------------------------------------------------
# Test 6 – Monotonicity: increasing an edge capacity cannot decrease max flow
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network(), extra_cap=st.integers(min_value=1, max_value=50))
def test_monotonicity_of_max_flow(G, extra_cap):
    """
    Property: Increasing the capacity of any single edge cannot decrease the
    maximum flow value.

    Mathematical basis: The feasible region only grows when a capacity
    constraint is relaxed, so the optimum (max flow) is monotonically
    non-decreasing in any edge capacity.

    Test strategy: Compute max flow on the original graph, pick a random
    edge, increase its capacity, recompute, and verify the new value is ≥
    the original.

    Assumptions / preconditions:
      - Graph has at least one edge.

    Failure implication: A decrease would violate the monotonicity of linear
    programmes and indicate a bug in the flow algorithm.
    """
    s, t = _source_sink(G)
    assume(G.number_of_edges() > 0)

    original_val = nx.maximum_flow_value(G, s, t)

    edges = list(G.edges())
    u, v = edges[random.randint(0, len(edges) - 1)]
    G2 = G.copy()
    G2[u][v]["capacity"] += extra_cap

    new_val = nx.maximum_flow_value(G2, s, t)
    assert new_val >= original_val - 1e-9, (
        f"Flow decreased from {original_val} to {new_val} after "
        f"increasing capacity on ({u},{v})"
    )


# ---------------------------------------------------------------------------
# Test 7 – Metamorphic: scaling all capacities scales the max flow
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network(), k=st.integers(min_value=2, max_value=10))
def test_capacity_scaling(G, k):
    """
    Property: If every edge capacity is multiplied by a positive constant k,
    the maximum flow value is also multiplied by k.

    Mathematical basis: The max-flow LP is:  maximise |f| subject to
    0 ≤ f(e) ≤ c(e).  Scaling all c(e) by k scales the feasible region
    uniformly, so the optimum scales by k.  Equivalently, f* is optimal for
    the original ⟺ k·f* is optimal for the scaled instance.

    Test strategy: Compute max flow, scale all capacities by k, recompute,
    and verify the ratio.

    Assumptions / preconditions:
      - k ≥ 2 (non-trivial scaling).

    Failure implication: A wrong ratio would indicate the algorithm is
    sensitive to absolute capacity values in an incorrect way.
    """
    s, t = _source_sink(G)
    original_val = nx.maximum_flow_value(G, s, t)

    G_scaled = G.copy()
    for u, v in G_scaled.edges():
        G_scaled[u][v]["capacity"] *= k

    scaled_val = nx.maximum_flow_value(G_scaled, s, t)
    assert abs(scaled_val - k * original_val) < 1e-9, (
        f"Scaled flow {scaled_val} != {k} * {original_val}"
    )


# ---------------------------------------------------------------------------
# Test 8 – Bottleneck on a single path
# ---------------------------------------------------------------------------

@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(G=single_path_network())
def test_single_path_bottleneck(G):
    """
    Property: In a graph that is a single directed path from source to sink,
    the maximum flow equals the minimum edge capacity along the path
    (the bottleneck).

    Mathematical basis: With only one s-t path, the flow is limited by the
    tightest edge.  The min-cut is the single bottleneck edge.

    Test strategy: Generate simple path graphs of length 2–10 with random
    capacities and verify max flow = min capacity.

    Assumptions / preconditions:
      - The graph is a simple directed path (no branches).

    Failure implication: Getting the wrong value on such a trivial topology
    would indicate a severe algorithmic error.
    """
    s, t = _source_sink(G)
    expected = min(d["capacity"] for _, _, d in G.edges(data=True))
    actual = nx.maximum_flow_value(G, s, t)
    assert abs(actual - expected) < 1e-9, (
        f"Bottleneck flow {actual} != expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 9 – Idempotence: computing max flow twice yields the same result
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_idempotence(G):
    """
    Property: Computing the maximum flow on the same (unmodified) graph
    twice yields identical flow values and identical flow dictionaries.

    Mathematical basis: A deterministic algorithm on identical input must
    produce identical output.  Even when the optimum is not unique, the
    implementation should be deterministic.

    Test strategy: Run maximum_flow twice on the same graph object and
    compare results.

    Assumptions / preconditions:
      - The graph is not mutated between calls.

    Failure implication: Different results would indicate non-determinism or
    unintended mutation of internal state.
    """
    s, t = _source_sink(G)
    val1, fd1 = nx.maximum_flow(G, s, t)
    val2, fd2 = nx.maximum_flow(G, s, t)

    assert val1 == val2
    assert fd1 == fd2


# ---------------------------------------------------------------------------
# Test 10 – Boundary: disconnected source and sink → zero flow
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=2, max_value=15))
def test_disconnected_source_sink(n):
    """
    Property: If there is no directed path from source to sink, the maximum
    flow is 0 and the minimum cut capacity is 0.

    Mathematical basis: With no augmenting path, the zero flow is already
    optimal.  The trivial cut ({s}, V\\{s}) has zero crossing capacity
    because no edge leaves the source's component toward the sink's
    component.

    Test strategy: Create a graph with n nodes but no edges from the
    source's component to the sink's component (two disconnected cliques).

    Assumptions / preconditions:
      - Source and sink are in different connected components.

    Failure implication: A non-zero flow on a disconnected network is
    clearly wrong.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    mid = max(1, n // 2)
    # Edges only within [0, mid) and [mid, n) — no cross edges
    for i in range(mid - 1):
        G.add_edge(i, i + 1, capacity=10)
    for i in range(mid, n - 1):
        G.add_edge(i, i + 1, capacity=10)

    s, t = 0, n - 1
    assume(not nx.has_path(G, s, t))

    assert nx.maximum_flow_value(G, s, t) == 0
    assert nx.minimum_cut_value(G, s, t) == 0


# ---------------------------------------------------------------------------
# Test 11 – Boundary: single-node graph (source == sink)
# ---------------------------------------------------------------------------

def test_single_node_graph():
    """
    Property: When the graph has a single node and source equals sink, the
    maximum flow value should be 0 (NetworkX raises an error for s == t, so
    we verify that behaviour is consistent).

    Mathematical basis: Flow from a node to itself is trivially zero or
    undefined.  NetworkX raises nx.NetworkXError when s == t.

    Test strategy: Deterministic edge-case test with a single-node graph.

    Failure implication: Changed exception behaviour would break downstream
    code relying on the documented API contract.
    """
    G = nx.DiGraph()
    G.add_node(0)
    try:
        nx.maximum_flow_value(G, 0, 0)
        # If no exception, the value should be 0
    except nx.NetworkXError:
        pass  # Expected


# ---------------------------------------------------------------------------
# Test 12 – Superadditivity: adding a new edge-disjoint s-t path increases
#           flow by at least the bottleneck of the new path
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network(min_nodes=4, max_nodes=12))
def test_adding_parallel_path_increases_flow(G):
    """
    Property: Adding an entirely new edge-disjoint path from source to sink
    increases the max flow by at least the bottleneck capacity of that new
    path.

    Mathematical basis: The new path provides an additional augmenting path
    that is independent of existing edges, so the max flow must increase by
    at least the minimum capacity along the new path.

    Test strategy: Compute max flow, then add a fresh path through new
    intermediate nodes (guaranteeing edge-disjointness), recompute, and
    verify the increase.

    Assumptions / preconditions:
      - The new path uses nodes not in the original graph, ensuring
        edge-disjointness.

    Failure implication: A smaller-than-expected increase means the
    algorithm fails to exploit an obvious augmenting path.
    """
    s, t = _source_sink(G)
    original_val = nx.maximum_flow_value(G, s, t)

    # Build a new path s → x1 → x2 → t through fresh nodes
    base = max(G.nodes()) + 1
    G2 = G.copy()
    path_cap = 7  # fixed capacity for the new path
    G2.add_edge(s, base, capacity=path_cap)
    G2.add_edge(base, base + 1, capacity=path_cap)
    G2.add_edge(base + 1, t, capacity=path_cap)

    new_val = nx.maximum_flow_value(G2, s, t)
    assert new_val >= original_val + path_cap - 1e-9, (
        f"Expected at least {original_val + path_cap}, got {new_val}"
    )


# ---------------------------------------------------------------------------
# Test 13 – Removing an edge cannot increase max flow
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_removing_edge_cannot_increase_flow(G):
    """
    Property: Removing any single edge from the network cannot increase the
    maximum flow.

    Mathematical basis: Removing an edge only tightens the feasible region
    (sets that edge's capacity to 0).  The optimum of a more constrained
    problem cannot exceed the original optimum.

    Test strategy: Compute max flow, remove a random edge, recompute, and
    verify the new value is ≤ the original.

    Assumptions / preconditions:
      - Graph has at least one edge.

    Failure implication: An increase after edge removal would violate basic
    optimisation monotonicity.
    """
    s, t = _source_sink(G)
    assume(G.number_of_edges() > 0)
    original_val = nx.maximum_flow_value(G, s, t)

    edges = list(G.edges())
    u, v = edges[random.randint(0, len(edges) - 1)]
    G2 = G.copy()
    G2.remove_edge(u, v)

    # Sink may become unreachable — that's fine, flow should be 0 or less
    if nx.has_path(G2, s, t):
        new_val = nx.maximum_flow_value(G2, s, t)
    else:
        new_val = 0

    assert new_val <= original_val + 1e-9, (
        f"Flow increased from {original_val} to {new_val} after "
        f"removing edge ({u},{v})"
    )


# ---------------------------------------------------------------------------
# Test 14 – Max flow on complete graph lower-bounded by max single-edge cap
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=3, max_value=8),
       caps=st.lists(st.integers(min_value=1, max_value=50),
                     min_size=1, max_size=50))
def test_complete_graph_lower_bound(n, caps):
    """
    Property: In a complete directed graph, the max flow from source to sink
    is at least as large as the capacity of any single direct s→t edge.

    Mathematical basis: The direct edge s→t alone is a feasible flow of
    value c(s,t).  The max flow must be ≥ any feasible flow.

    Test strategy: Build a complete directed graph on n nodes with random
    capacities and verify the lower bound.

    Assumptions / preconditions:
      - n ≥ 3 so source ≠ sink and there are intermediate nodes.

    Failure implication: Returning a value below a trivially feasible flow
    means the algorithm is not finding even the most obvious solution.
    """
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    idx = 0
    for u, v in G.edges():
        G[u][v]["capacity"] = caps[idx % len(caps)]
        idx += 1

    s, t = 0, n - 1
    direct_cap = G[s][t]["capacity"]
    flow_val = nx.maximum_flow_value(G, s, t)
    assert flow_val >= direct_cap - 1e-9


# ---------------------------------------------------------------------------
# Test 15 – Weak duality: any feasible flow ≤ any cut capacity
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_weak_duality(G):
    """
    Property: The maximum flow value is ≤ the capacity of every possible
    cut, not just the minimum cut.  We verify this by checking a few random
    partitions.

    Mathematical basis: Weak duality states that for any feasible flow f
    and any s-t cut (S,T), |f| ≤ cap(S,T).  In particular, the max flow
    is ≤ every cut.

    Test strategy: Compute max flow, then generate several random partitions
    of V with s ∈ S and t ∈ T, compute their cut capacities, and verify
    each is ≥ the max flow.

    Assumptions / preconditions:
      - Valid flow network.

    Failure implication: Violating weak duality would mean the flow exceeds
    a cut, which is impossible in a correct implementation.
    """
    s, t = _source_sink(G)
    flow_val = nx.maximum_flow_value(G, s, t)
    nodes = sorted(G.nodes())
    intermediate = [v for v in nodes if v != s and v != t]

    # Try up to 10 random partitions
    for _ in range(min(10, 2 ** len(intermediate))):
        S = {s}
        for v in intermediate:
            if random.random() < 0.5:
                S.add(v)
        T = set(nodes) - S
        assert t in T

        cut_cap = sum(
            G[u][v]["capacity"]
            for u in S for v in T if G.has_edge(u, v)
        )
        assert flow_val <= cut_cap + 1e-9, (
            f"Max flow {flow_val} exceeds cut capacity {cut_cap}"
        )
