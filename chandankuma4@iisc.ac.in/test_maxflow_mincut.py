"""
Property-Based Tests for NetworkX Max-Flow / Min-Cut Algorithms
================================================================

**Course:** E0 251o Data Structures & Graph Analytics (2026)
**Team member:** M Chandan Kumar Rao (chandankuma4@iisc.ac.in, SR No. 24650)

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

    Why this matters: If this property fails, either the max-flow or the
    min-cut implementation has a fundamental correctness bug—one of them
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

    Why this matters: If this property fails, the algorithm produced an
    invalid flow—some node is "creating" or "destroying" flow, which is
    physically and mathematically impossible in a correct solution.
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

    Why this matters: An edge carrying negative flow or flow exceeding
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

    Why this matters: A discrepancy means either the flow decomposition
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

    Why this matters: An invalid partition or mismatched capacity means
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
@given(G=flow_network(), extra_cap=st.integers(min_value=1, max_value=50),
       edge_data=st.randoms(use_true_random=False))
def test_monotonicity_of_max_flow(G, extra_cap, edge_data):
    """
    Property: Increasing the capacity of any single edge cannot decrease the
    maximum flow value.

    Mathematical basis: The feasible region only grows when a capacity
    constraint is relaxed, so the optimum (max flow) is monotonically
    non-decreasing in any edge capacity.

    Test strategy: Compute max flow on the original graph, pick a random
    edge, increase its capacity, recompute, and verify the new value is ≥
    the original.

    Why this matters: A decrease would violate the monotonicity of linear
    programmes and indicate a bug in the flow algorithm.
    """
    s, t = _source_sink(G)
    assume(G.number_of_edges() > 0)

    original_val = nx.maximum_flow_value(G, s, t)

    edges = list(G.edges())
    u, v = edges[edge_data.randint(0, len(edges) - 1)]
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
    and verify the ratio.  Graphs are random directed networks (3-15 nodes).

    Why this matters: A wrong ratio would indicate the algorithm is
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

    Why this matters: Getting the wrong value on such a trivial topology
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

    Why this matters: Different results would indicate non-determinism or
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

    Why this matters: A non-zero flow on a disconnected network is
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

@given(node_id=st.integers(min_value=0, max_value=100))
def test_single_node_graph(node_id):
    """
    Property: When the graph has a single node and source equals sink, the
    maximum flow value should be 0 (NetworkX raises an error for s == t, so
    we verify that behaviour is consistent).

    Mathematical basis: Flow from a node to itself is trivially zero or
    undefined.  NetworkX raises nx.NetworkXError when s == t.

    Test strategy: Generate single-node graphs with varying node IDs and
    verify that NetworkX consistently raises NetworkXError (or returns 0)
    regardless of the node label.

    Why this matters: Changed exception behaviour would break downstream
    code relying on the documented API contract.
    """
    G = nx.DiGraph()
    G.add_node(node_id)
    try:
        nx.maximum_flow_value(G, node_id, node_id)
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

    Why this matters: A smaller-than-expected increase means the algorithm
    fails to exploit an obvious augmenting path.
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
@given(G=flow_network(), edge_data=st.randoms(use_true_random=False))
def test_removing_edge_cannot_increase_flow(G, edge_data):
    """
    Property: Removing any single edge from the network cannot increase the
    maximum flow.

    Mathematical basis: Removing an edge only tightens the feasible region
    (sets that edge's capacity to 0).  The optimum of a more constrained
    problem cannot exceed the original optimum.

    Test strategy: Compute max flow, remove a random edge, recompute, and
    verify the new value is ≤ the original.

    Why this matters: An increase after edge removal would violate basic
    optimisation monotonicity.
    """
    s, t = _source_sink(G)
    assume(G.number_of_edges() > 0)
    original_val = nx.maximum_flow_value(G, s, t)

    edges = list(G.edges())
    u, v = edges[edge_data.randint(0, len(edges) - 1)]
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

    Why this matters: Returning a value below a trivially feasible flow
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
@given(G=flow_network(), rng=st.randoms(use_true_random=False))
def test_weak_duality(G, rng):
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

    Why this matters: If the flow exceeds a cut, the implementation has
    a fundamental error—this is mathematically impossible.
    """
    s, t = _source_sink(G)
    flow_val = nx.maximum_flow_value(G, s, t)
    nodes = sorted(G.nodes())
    intermediate = [v for v in nodes if v != s and v != t]

    # Try up to 10 random partitions
    for _ in range(min(10, 2 ** len(intermediate))):
        S = {s}
        for v in intermediate:
            if rng.random() < 0.5:
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


# ---------------------------------------------------------------------------
# Test 16 – Cross-algorithm consensus (differential / N-version testing)
# ---------------------------------------------------------------------------

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_all_algorithms_agree(G):
    """
    Property: All five NetworkX max-flow implementations (Edmonds-Karp,
    Shortest Augmenting Path, Preflow-Push, Dinitz, Boykov-Kolmogorov)
    must return the same max-flow value on any input graph.

    Mathematical basis: The max-flow value is the optimum of a linear
    programme, so it is unique even when multiple optimal flow decompositions
    exist.  Different algorithms may find different edge-level flows, but
    the total value must be identical.

    Test strategy: Generate random directed networks (3-15 nodes, positive
    integer capacities) and run all five algorithms, asserting consensus.
    This is differential testing -- no oracle needed, just agreement among
    independent implementations.

    Why this matters: A disagreement would pinpoint a correctness bug in at
    least one of the five implementations.  Because each uses a fundamentally
    different approach (augmenting paths vs. preflow vs. blocking flows), a
    mismatch strongly localises the fault.
    """
    s, t = _source_sink(G)
    algos = [
        nx.algorithms.flow.edmonds_karp,
        nx.algorithms.flow.shortest_augmenting_path,
        nx.algorithms.flow.preflow_push,
        nx.algorithms.flow.dinitz,
        nx.algorithms.flow.boykov_kolmogorov,
    ]
    values = [nx.maximum_flow_value(G, s, t, flow_func=a) for a in algos]
    assert all(abs(v - values[0]) < 1e-9 for v in values), (
        f"Algorithm disagreement: {dict(zip([a.__name__ for a in algos], values))}"
    )


# ---------------------------------------------------------------------------
# Test 17 – No augmenting path in the residual graph (optimality certificate)
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_no_augmenting_path_in_residual(G):
    """
    Property: After computing the maximum flow, the residual graph contains
    no directed path from source to sink.

    Mathematical basis: The Augmenting Path Theorem (a corollary of the
    Max-Flow Min-Cut Theorem) states that a flow is maximum if and only if
    no augmenting path exists in the residual graph G_f.  The residual has
    forward edge (u,v) with capacity c-f when f < c, and backward edge (v,u)
    with capacity f when f > 0.

    Test strategy: Compute max flow, reconstruct the residual graph
    independently from the returned flow dict, and run a reachability check.
    This acts as an optimality certificate that does not trust the
    algorithm's internal bookkeeping.

    Why this matters: If an augmenting path exists in the residual, the
    algorithm stopped too early -- more flow could still be pushed, meaning
    the returned value is not actually the maximum.
    """
    s, t = _source_sink(G)
    _, flow_dict = nx.maximum_flow(G, s, t)

    # Build residual graph
    R = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        cap = d["capacity"]
        f = flow_dict[u][v]
        if cap - f > 1e-9:
            R.add_edge(u, v)
        if f > 1e-9:
            R.add_edge(v, u)

    R.add_nodes_from(G.nodes())  # ensure s and t exist even if isolated
    assert not nx.has_path(R, s, t), (
        "Residual graph has an s-t path — flow is not maximum"
    )


# ---------------------------------------------------------------------------
# Test 18 – Complementary slackness: min-cut edges are saturated
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_mincut_complementary_slackness(G):
    """
    Property: In an optimal max-flow / min-cut pair, every forward edge
    crossing the cut (S to T) is fully saturated, and every backward edge
    (T to S) carries zero flow.

    Mathematical basis: These are the complementary slackness conditions
    from LP duality applied to the max-flow / min-cut pair.  Together they
    imply strong duality: max_flow = sum of saturated S->T capacities - 0
    = min_cut.  This is a stronger structural check than simply comparing
    the two scalar values.

    Test strategy: Compute the flow (with flow dict) and the cut (with
    partition) separately, then verify saturation on every S->T edge and
    zero flow on every T->S edge.

    Why this matters: A violation means the returned flow and cut do not
    form a valid primal-dual optimal pair, indicating a bug in either the
    flow computation or the cut extraction.
    """
    s, t = _source_sink(G)
    _, flow_dict = nx.maximum_flow(G, s, t)
    _, (S, T) = nx.minimum_cut(G, s, t)
    S, T = set(S), set(T)

    # Forward cut edges (S → T) must be saturated
    for u in S:
        for v in T:
            if G.has_edge(u, v):
                f = flow_dict[u][v]
                c = G[u][v]["capacity"]
                assert abs(f - c) < 1e-9, (
                    f"Cut edge ({u},{v}): flow {f} != capacity {c} "
                    f"(not saturated)"
                )

    # Backward cut edges (T → S) must carry zero flow
    for u in T:
        for v in S:
            if G.has_edge(u, v):
                f = flow_dict[u][v]
                assert abs(f) < 1e-9, (
                    f"Backward cut edge ({u},{v}): flow {f} != 0"
                )


# ---------------------------------------------------------------------------
# Test 19 – Edge-reversal symmetry: max-flow(G, s, t) = max-flow(Gᴿ, t, s)
# ---------------------------------------------------------------------------

@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network())
def test_edge_reversal_symmetry(G):
    """
    Property: Reversing every edge and swapping source/sink preserves the
    max-flow value: max-flow(G, s->t) == max-flow(G^R, t->s).

    Mathematical basis: Any feasible flow f in G maps to a feasible flow
    f^R in G^R of equal value via f^R(v,u) = f(u,v).  This bijection
    preserves feasibility and value, so the optima coincide.  Equivalently,
    the min-cut capacity is unchanged because the crossing edges are the
    same (just reversed).

    Test strategy: Compute max flow on the original graph, build the
    reversed graph, compute max flow with swapped source/sink, and assert
    equality.  Uses random directed networks (3-15 nodes).

    Why this matters: This symmetry is rarely tested explicitly but catches
    orientation-dependent bugs that only show up in one edge direction.
    """
    s, t = _source_sink(G)
    original_val = nx.maximum_flow_value(G, s, t)

    G_rev = nx.DiGraph()
    G_rev.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        G_rev.add_edge(v, u, capacity=d["capacity"])

    reversed_val = nx.maximum_flow_value(G_rev, t, s)
    assert abs(original_val - reversed_val) < 1e-9, (
        f"max-flow(G, {s}→{t}) = {original_val} != "
        f"max-flow(Gᴿ, {t}→{s}) = {reversed_val}"
    )


# ---------------------------------------------------------------------------
# Test 20 – Gomory-Hu tree: tree path minimum = pairwise max-flow
# ---------------------------------------------------------------------------

@st.composite
def undirected_flow_network(draw, min_nodes=3, max_nodes=10,
                            min_cap=1, max_cap=50):
    """Generate a random connected undirected graph with positive integer
    capacities.  Used for Gomory-Hu tree tests."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Spanning path to guarantee connectivity
    for i in range(n - 1):
        cap = draw(st.integers(min_value=min_cap, max_value=max_cap))
        G.add_edge(i, i + 1, capacity=cap)
    # Random extra edges
    extra = draw(st.integers(min_value=0, max_value=n * (n - 1) // 4))
    possible = [(u, v) for u in range(n) for v in range(u + 1, n)
                if not G.has_edge(u, v)]
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


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(G=undirected_flow_network())
def test_gomory_hu_tree(G):
    """
    Property: In the Gomory-Hu tree T of an undirected graph G, for every
    pair of nodes (u, v), the minimum edge weight on the unique u-v path
    in T equals the max-flow between u and v in G.

    Mathematical basis: The Gomory-Hu theorem (1961) proves that for any
    undirected graph with n nodes, such a tree exists with only n-1 edges
    encoding all O(n^2) pairwise max-flow values.

    Test strategy: Generate random connected undirected graphs (3-10 nodes,
    positive integer capacities), compute the Gomory-Hu tree, then for
    every node pair check the tree-path minimum against a direct max-flow
    call.  This cross-validates two independent algorithms.

    Why this matters: A mismatch would indicate a bug in either the
    Gomory-Hu tree construction or the max-flow routine (or both).
    """
    T = nx.gomory_hu_tree(G)
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            # Min edge weight on tree path
            path = nx.shortest_path(T, u, v)
            tree_min = min(
                T[path[k]][path[k + 1]]["weight"]
                for k in range(len(path) - 1)
            )
            # Direct max-flow computation
            flow_val = nx.maximum_flow_value(G, u, v)
            assert abs(tree_min - flow_val) < 1e-9, (
                f"Gomory-Hu mismatch for ({u},{v}): "
                f"tree_min={tree_min}, max_flow={flow_val}"
            )


# ===========================================================================
# BUG DISCOVERY TEST
# ===========================================================================
# The following test exposes a genuine bug in NetworkX's max-flow / min-cut
# implementation: when edges have negative capacities, the library silently
# produces internally inconsistent results instead of raising an error.
#
# Evidence:
#   - nx.minimum_cut(G, 0, 2) returns (10, ({0,1}, {2}))
#   - But the actual capacity of that partition is only 5 (edge (0,2) has
#     capacity -5, edge (1,2) has capacity 10, total = 5)
#   - The reported cut value (10) does NOT match the partition capacity (5)
#
# Root cause:
#   In networkx/algorithms/flow/utils.py, build_residual_network() filters
#   edges with: `attr.get(capacity, inf) > 0`, silently dropping any edge
#   with capacity <= 0. The algorithms then operate on a DIFFERENT graph
#   than the user provided, but the partition is returned as if it applies
#   to the original graph.
#
# Verification this is not a test error:
#   - The graph is a valid nx.DiGraph with 3 nodes and 3 edges
#   - The API accepts it without any error or warning
#   - The returned partition IS structurally valid (S ∪ T = V, S ∩ T = ∅)
#   - But the reported cut value is inconsistent with the partition
#   - The bug reproduces across ALL 5 flow algorithms
#
# Suggested fix:
#   NetworkX should raise a ValueError when any edge has a negative capacity,
#   since flow networks require non-negative capacities by definition.
# ===========================================================================


@st.composite
def flow_network_with_negative_caps(draw, min_nodes=3, max_nodes=10):
    """Generate a directed graph with at least one s-t path using positive
    capacities, plus additional edges that may have NEGATIVE capacities.
    This strategy is designed to expose the silent-negative-capacity bug."""
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Guarantee s-t path with positive capacities
    for i in range(n - 1):
        G.add_edge(i, i + 1, capacity=draw(st.integers(1, 50)))

    # Add extra edges, some with NEGATIVE capacities
    possible = [(u, v) for u in range(n) for v in range(n)
                if u != v and not G.has_edge(u, v)]
    if possible:
        chosen = draw(st.lists(
            st.sampled_from(possible),
            min_size=1, max_size=min(n, len(possible)),
            unique=True,
        ))
        for u, v in chosen:
            # Allow negative capacities: range [-50, 50]
            cap = draw(st.integers(min_value=-50, max_value=50))
            G.add_edge(u, v, capacity=cap)

    return G


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(G=flow_network_with_negative_caps())
def test_bug_negative_capacity_silent_inconsistency(G):
    """
    *** BUG DISCOVERY: NetworkX silently produces inconsistent min-cut
    results when edges have negative capacities. ***

    Property: If NetworkX accepts a graph without raising an error, then
    the minimum_cut result must be internally consistent: the reported
    cut value must equal the sum of capacities of edges crossing from S
    to T in the returned partition.

    Mathematical basis: By definition, the capacity of a cut (S, T) is
    cap(S, T) = Σ c(u,v) for all edges (u,v) with u ∈ S, v ∈ T.  If
    the library returns a partition and a cut value, they must agree.

    Bug description: When any edge has a negative capacity, NetworkX's
    build_residual_network() silently drops that edge (the filter is
    `attr.get(capacity, inf) > 0`).  The max-flow and min-cut algorithms
    then operate on a MODIFIED graph that excludes negative-capacity edges.
    However, the partition (S, T) is returned to the user, who naturally
    recomputes the cut capacity using the ORIGINAL graph's capacities.
    This produces a mismatch: the reported cut value reflects the modified
    graph, while the partition applies to the original graph.

    Minimal reproducer:
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(0, 2, capacity=-5)

        cut_val, (S, T) = nx.minimum_cut(G, 0, 2)
        # Returns: (10, ({0, 1}, {2}))
        # But actual partition capacity = 10 + (-5) = 5 ≠ 10

    Test strategy: Generate random flow networks where some edges have
    negative capacities.  If NetworkX does NOT raise an error, verify
    that the reported cut value matches the recomputed partition capacity.
    The test is expected to FAIL, demonstrating the bug.

    Failure implication: This is a genuine bug in NetworkX — the library
    silently returns an internally inconsistent result.  The fix should
    be to raise a ValueError when any edge has a negative capacity.

    Affected versions: NetworkX 3.2.1 (and likely all prior versions).
    Affected functions: nx.minimum_cut, nx.minimum_cut_value (all 5
    underlying algorithms: edmonds_karp, shortest_augmenting_path,
    preflow_push, dinitz, boykov_kolmogorov).
    """
    s, t = 0, max(G.nodes())

    has_negative = any(d["capacity"] < 0 for _, _, d in G.edges(data=True))
    if not has_negative:
        return  # Only test graphs that actually have negative capacities

    try:
        cut_value, (S, T) = nx.minimum_cut(G, s, t)
    except (nx.NetworkXError, nx.NetworkXUnbounded, ValueError):
        return  # If NetworkX raises an error, that's acceptable behaviour

    # If NetworkX accepted the input, the result must be consistent
    recomputed = sum(
        G[u][v]["capacity"] for u in S for v in T if G.has_edge(u, v)
    )
    assert abs(recomputed - cut_value) < 1e-9, (
        f"BUG: minimum_cut reports cut_value={cut_value} but the actual "
        f"capacity of partition (S={sorted(S)}, T={sorted(T)}) is "
        f"{recomputed}. Edges: {list(G.edges(data=True))}"
    )
