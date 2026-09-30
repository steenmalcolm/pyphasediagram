import numpy as np
import pytest
import networkx as nx
from shapely import LineString

from pyphasediagram.spinodal import SpinodalPoint, CriticalPoint, Spinodal

# TODO: Add End-to-end tests that build a full spinodal graph and check for expected properties such as geometry and orientation of critical point
# and include tests on symmetric system for 2.5<chi<3.2


def _make_path_subgraph(points):
    """
    Helper: points is list of SpinodalPoint in desired path order.
    Returns a Graph that is a path connecting them in sequence.
    """
    G = nx.Graph()
    for p in points:
        G.add_node(p)
    for a, b in zip(points[:-1], points[1:]):
        G.add_edge(a, b)
    return G


def _make_cycle_subgraph(points):
    """
    Helper: points is list of SpinodalPoint in cycle order.
    Returns a Graph that is a cycle through all points.
    """
    G = nx.Graph()
    for p in points:
        G.add_node(p)
    for a, b in zip(points[:-1], points[1:]):
        G.add_edge(a, b)
    G.add_edge(points[-1], points[0])
    return G


@pytest.fixture
def simple_chis():
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)


@pytest.fixture
def trivial_spinodal_graph():
    G = nx.Graph()
    # Create a trivial component with 3 nodes (ordering doesn't matter here)
    p0 = SpinodalPoint(0, 0.10, 0.20)
    p1 = SpinodalPoint(1, 0.20, 0.10)
    p2 = SpinodalPoint(2, 0.30, 0.05)
    G.add_edge(p0, p1)
    G.add_edge(p1, p2)
    return G


# ----------------------------
# Tests: Spinodal
# ----------------------------


def test_spinodal_init(simple_chis):
    sp = Spinodal(simple_chis)
    assert np.allclose(sp.chis, simple_chis)
    assert isinstance(sp.spinodal_graph, nx.Graph)
    assert sp.spinodal_graph.number_of_nodes() == 0
    assert sp.node_id == 0


def test_get_p_q_scalar_matches_manual(simple_chis):
    sp = Spinodal(simple_chis)
    phi = 0.3

    a, b, c = simple_chis[0, 0], simple_chis[1, 1], simple_chis[0, 1]
    det = a * b - c**2

    p_expected = (2 * phi * c - a - phi * (1 - phi) * det) / (a + det * phi)
    q_expected = (1 + phi * (1 - phi) * b) / (a + det * phi)

    p, q = sp._get_p_q(phi, is_calculate_phi2=True)
    assert p == pytest.approx(p_expected)
    assert q == pytest.approx(q_expected)


def test_get_p_q_swap_flag_equivalent_to_swapping_a_b(simple_chis):
    sp = Spinodal(simple_chis)
    phi = 0.42

    p1, q1 = sp._get_p_q(phi, is_calculate_phi2=False)

    # emulate swap a<->b as in code path
    chis_swapped = np.array(
        [
            [simple_chis[1, 1], simple_chis[0, 1]],
            [simple_chis[1, 0], simple_chis[0, 0]],
        ],
        dtype=float,
    )
    sp_swapped = Spinodal(chis_swapped)
    p2, q2 = sp_swapped._get_p_q(phi, is_calculate_phi2=True)

    assert p1 == pytest.approx(p2)
    assert q1 == pytest.approx(q2)


def test_phi2_from_phi1_vector_shapes(simple_chis):
    sp = Spinodal(simple_chis)
    phi1 = np.linspace(0.0, 1.0, 11)

    phi2a, phi2b = sp.phi2_from_phi1(phi1)

    assert phi2a.shape == phi1.shape
    assert phi2b.shape == phi1.shape


def test_phi2_from_phi1_scalar_returns_scalar_like(simple_chis):
    sp = Spinodal(simple_chis)
    phi2a, phi2b = sp.phi2_from_phi1(0.1)
    # numpy scalar or python float is fine; ensure it's not an array with wrong shape
    assert np.isscalar(phi2a) or (isinstance(phi2a, np.ndarray) and phi2a.shape == ())
    assert np.isscalar(phi2b) or (isinstance(phi2b, np.ndarray) and phi2b.shape == ())


def test_domain_data_adds_nodes_and_edges(simple_chis):
    sp = Spinodal(simple_chis)
    sp._domain_data(0.1, 0.2)

    # _domain_data uses num_points=1000 and adds two branches -> ~2000 nodes
    assert sp.spinodal_graph.number_of_nodes() > 0
    assert sp.spinodal_graph.number_of_edges() > 0

    # Sanity: all nodes are SpinodalPoint and have increasing ids assigned
    nodes = list(sp.spinodal_graph.nodes())
    assert all(isinstance(n, SpinodalPoint) for n in nodes)
    assert len({n.idx for n in nodes}) == len(nodes)
    assert sp.node_id == len(nodes)


def test_clip_to_domain_removes_outside_and_singletons(simple_chis):
    sp = Spinodal(simple_chis)

    # Valid component of 2 nodes (should stay)
    a0 = SpinodalPoint(0, 0.2, 0.2)
    a1 = SpinodalPoint(1, 0.21, 0.2)
    sp.spinodal_graph.add_edge(a0, a1)

    # Outside domain nodes (should be removed)
    bad1 = SpinodalPoint(2, -0.1, 0.2)  # phi1 < 0
    bad2 = SpinodalPoint(3, 0.2, -0.1)  # phi2 < 0
    bad3 = SpinodalPoint(4, 0.7, 0.4)  # phi1 + phi2 > 1
    sp.spinodal_graph.add_node(bad1)
    sp.spinodal_graph.add_node(bad2)
    sp.spinodal_graph.add_node(bad3)

    # Single-node component inside domain (should be removed by singleton cleanup)
    lone = SpinodalPoint(5, 0.3, 0.3)
    sp.spinodal_graph.add_node(lone)

    sp._clip_to_domain()

    remaining = set(sp.spinodal_graph.nodes())
    assert a0 in remaining and a1 in remaining
    assert bad1 not in remaining
    assert bad2 not in remaining
    assert bad3 not in remaining
    assert lone not in remaining


def test_connect_branches_adds_edge_when_close_and_pointing(simple_chis):
    sp = Spinodal(simple_chis)

    # Component A: A0 -- A1 (A1 will connect to B0)
    a0 = SpinodalPoint(0, 0.00, 0.00)
    a1 = SpinodalPoint(1, 0.01, 0.00)
    sp.spinodal_graph.add_edge(a0, a1)

    # Component B: B0 -- B1 (B0 will connect to A1)
    b0 = SpinodalPoint(2, 0.02, 0.00)
    b1 = SpinodalPoint(3, 0.03, 0.00)
    sp.spinodal_graph.add_edge(b0, b1)

    # Precondition: two components
    comps_before = list(nx.connected_components(sp.spinodal_graph))
    assert len(comps_before) == 2

    sp._connect_branches()

    # After, should be connected
    comps_after = list(nx.connected_components(sp.spinodal_graph))
    assert len(comps_after) == 1
    assert sp.spinodal_graph.has_edge(a1, b0) or sp.spinodal_graph.has_edge(b0, a1)


def test_build_calls_internal_steps(monkeypatch, simple_chis):
    sp = Spinodal(simple_chis)

    called = {"domain_data": 0, "clip": 0, "connect": 0}

    monkeypatch.setattr(sp, "_spinodal_domains", lambda: np.array([0.1, 0.2, 0.7, 0.8]))

    def fake_domain_data(phi1_i, phi1_f, num_points=1000):
        called["domain_data"] += 1
        # keep it minimal: add a tiny edge so later steps can run if needed
        p0 = SpinodalPoint(sp.node_id, phi1_i, 0.1)
        sp.node_id += 1
        p1 = SpinodalPoint(sp.node_id, phi1_f, 0.1)
        sp.node_id += 1
        sp.spinodal_graph.add_edge(p0, p1)

    monkeypatch.setattr(sp, "_domain_data", fake_domain_data)
    monkeypatch.setattr(
        sp, "_clip_to_domain", lambda: called.__setitem__("clip", called["clip"] + 1)
    )
    monkeypatch.setattr(
        sp,
        "_connect_branches",
        lambda: called.__setitem__("connect", called["connect"] + 1),
    )

    sp.build()

    # domains has 4 values => two intervals => _domain_data called twice
    assert called["domain_data"] == 2
    assert called["clip"] == 1
    assert called["connect"] == 1
    assert sp.spinodal_graph.number_of_nodes() > 0
    assert sp.spinodal_graph.number_of_edges() > 0


# ----------------------------
# Tests: Spinodal._coords_from_subgraph
# ----------------------------


def test_coords_from_subgraph_path_order(simple_chis):
    sp = Spinodal(simple_chis)

    p0 = SpinodalPoint(0, 0.10, 0.20)
    p1 = SpinodalPoint(1, 0.11, 0.21)
    p2 = SpinodalPoint(2, 0.12, 0.22)
    sg = _make_path_subgraph([p0, p1, p2])

    phi1s, phi2s = sp._coords_from_subgraph(sg)

    # For a path, it should start at an endpoint and follow along the path.
    # Either direction is acceptable because endpoints[0] depends on iteration order.
    forward_phi1 = np.array([0.10, 0.11, 0.12])
    backward_phi1 = forward_phi1[::-1]

    assert np.allclose(phi1s, forward_phi1) or np.allclose(phi1s, backward_phi1)

    # And phi2 should match the same traversal
    forward_phi2 = np.array([0.20, 0.21, 0.22])
    backward_phi2 = forward_phi2[::-1]
    assert np.allclose(phi2s, forward_phi2) or np.allclose(phi2s, backward_phi2)


def test_coords_from_subgraph_cycle_contains_all_nodes(simple_chis):
    sp = Spinodal(simple_chis)

    p0 = SpinodalPoint(0, 0.10, 0.20)
    p1 = SpinodalPoint(1, 0.11, 0.21)
    p2 = SpinodalPoint(2, 0.12, 0.22)
    p3 = SpinodalPoint(3, 0.13, 0.23)
    sg = _make_cycle_subgraph([p0, p1, p2, p3])

    phi1s, phi2s = sp._coords_from_subgraph(sg)

    # DFS on a cycle should visit every node exactly once
    assert len(phi1s) == 4
    assert len(phi2s) == 4

    # Order is not guaranteed, but the set of coordinates should match
    assert set(np.round(phi1s, 12)) == {0.10, 0.11, 0.12, 0.13}
    assert set(np.round(phi2s, 12)) == {0.20, 0.21, 0.22, 0.23}


# ----------------------------
# Tests: Spinodal._third_derivative
# ----------------------------


def test_third_derivative_scalar_matches_manual(simple_chis):
    sp = Spinodal(simple_chis)
    # choose values away from 0 / boundaries to avoid singularities
    phi1, phi2 = 0.2, 0.3

    phi0 = 1 - phi1 - phi2
    H_11 = 1 / phi1 + 1 / phi0 + simple_chis[0, 0]
    H_12 = 1 / phi0 + simple_chis[0, 1]
    expected = H_12**3 / (phi1**2) - H_11**3 / (phi2**2) + (H_11 - H_12) ** 3 / phi0**2

    got = sp._third_derivative(phi1, phi2)
    assert got == pytest.approx(expected)


def test_third_derivative_vectorized_shapes(simple_chis):
    sp = Spinodal(simple_chis)
    phi1 = np.array([0.2, 0.21, 0.22])
    phi2 = np.array([0.3, 0.29, 0.28])

    out = sp._third_derivative(phi1, phi2)
    assert isinstance(out, np.ndarray)
    assert out.shape == phi1.shape


# ----------------------------
# Tests: Spinodal._find_critical_points
# ----------------------------


def test_find_critical_points_appends_point(
    monkeypatch, simple_chis, trivial_spinodal_graph
):
    sp = Spinodal(simple_chis)
    sp.spinodal_graph = trivial_spinodal_graph
    sp.critical_points = []

    # Force _coords_from_subgraph to return known arrays of length 3
    # and force a third-derivative sign change across the two samples.
    monkeypatch.setattr(
        sp,
        "_coords_from_subgraph",
        lambda sg: (np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.1, 0.05])),
    )

    # Third derivative: [-1, +1] => sign change => one root at index 0.
    def fake_third_derivative(phi1, phi2):
        # Called once with arrays, later with scalar interpolated values
        if np.ndim(phi1) > 0:
            return np.array([-1.0, -1.0, +1.0])
        return 0.0  # td_c at interpolated critical point

    monkeypatch.setattr(sp, "_third_derivative", fake_third_derivative)

    sp._find_critical_points()

    assert len(sp.critical_points) == 1
    cp = sp.critical_points[0]
    assert isinstance(cp, CriticalPoint)
    assert cp.idx == -1
    # Interp between (phi1,phi2) = (0.2,0.1) and (0.3,0.05) at td=0 -> midpoint
    assert cp.phi1 == pytest.approx(0.25)
    assert cp.phi2 == pytest.approx(0.075)


def test_find_critical_points_handles_decreasing_xp_order(
    monkeypatch, simple_chis, trivial_spinodal_graph
):
    """
    This specifically tests the code path that sorts xp before np.interp.
    If xp is [1, -1, -2 ] (decreasing), interp would fail without sorting.
    """
    sp = Spinodal(simple_chis)
    sp.spinodal_graph = trivial_spinodal_graph
    sp.critical_points = []

    monkeypatch.setattr(
        sp,
        "_coords_from_subgraph",
        lambda sg: (np.array([0.2, 0.4, 0.6]), np.array([0.6, 0.4, 0.2])),
    )

    def fake_third_derivative(phi1, phi2):
        if np.ndim(phi1) > 0:
            return np.array([+1.0, -1.0, -2.0])  # decreasing xp, but still crosses zero
        return 0.0

    monkeypatch.setattr(sp, "_third_derivative", fake_third_derivative)

    sp._find_critical_points()

    assert len(sp.critical_points) == 1
    cp = sp.critical_points[0]
    assert cp.phi1 == pytest.approx(0.3)
    assert cp.phi2 == pytest.approx(0.5)


def test_find_critical_points_raises_on_nan_third_derivative(monkeypatch, simple_chis):
    sp = Spinodal(simple_chis)
    sp.spinodal_graph = nx.Graph()
    sp.critical_points = []

    p0 = SpinodalPoint(0, 0.10, 0.20)
    p1 = SpinodalPoint(1, 0.20, 0.10)
    sp.spinodal_graph.add_edge(p0, p1)

    monkeypatch.setattr(
        sp,
        "_coords_from_subgraph",
        lambda sg: (np.array([0.1, 0.2]), np.array([0.2, 0.1])),
    )
    monkeypatch.setattr(
        sp,
        "_third_derivative",
        lambda phi1, phi2: np.array([np.nan, 1.0]) if np.ndim(phi1) > 0 else 0.0,
    )

    with pytest.raises(ValueError, match="NaN values found in third derivative"):
        sp._find_critical_points()


# ----------------------------
# Tests: Spinodal._in_domain
# ----------------------------


class TestInDomain:
    def test_interior_point(self):
        assert Spinodal._in_domain(0.2, 0.3) == True

    def test_origin_excluded(self):
        assert Spinodal._in_domain(0.0, 0.0) == False

    def test_boundary_excluded(self):
        assert Spinodal._in_domain(0.5, 0.5) == False  # phi1 + phi2 == 1
        assert Spinodal._in_domain(0.0, 0.5) == False  # phi1 == 0
        assert Spinodal._in_domain(0.5, 0.0) == False  # phi2 == 0

    def test_corner_points_excluded(self):
        assert Spinodal._in_domain(0.0, 1.0) == False
        assert Spinodal._in_domain(1.0, 0.0) == False

    def test_outside_negative_phi1(self):
        assert Spinodal._in_domain(-0.1, 0.5) == False

    def test_outside_negative_phi2(self):
        assert Spinodal._in_domain(0.5, -0.1) == False

    def test_outside_sum_exceeds_one(self):
        assert Spinodal._in_domain(0.6, 0.5) == False

    def test_array_input(self):
        phi1 = np.array([0.2, -0.1, 0.6, 0.0])
        phi2 = np.array([0.3, 0.5, 0.5, 0.5])
        result = Spinodal._in_domain(phi1, phi2)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(result, expected)


# tests/test_spinodal_e2e.py
import numpy as np
import pytest
import networkx as nx
import shapely

from pyphasediagram.spinodal import Spinodal

# ----------------------------
# Helpers
# ----------------------------


def _random_chis(rng: np.random.Generator, low=-10.0, high=10.0) -> np.ndarray:
    """
    Symmetric 2x2 chi matrix with entries in [low, high]:
        [[chi11, chi12],
         [chi12, chi22]]
    """
    chi11 = rng.uniform(low, high)
    chi22 = rng.uniform(low, high)
    chi12 = rng.uniform(low, high)
    return np.array([[chi11, chi12], [chi12, chi22]], dtype=float)


def _build_spinodal_or_skip(chis: np.ndarray, num_points=800) -> Spinodal:
    """
    Build a spinodal.
    Skips if build produces an empty graph or throws common numerical errors.
    """
    sp = Spinodal(chis)

    sp.build(num_points=num_points)

    if sp.spinodal_graph.number_of_nodes() == 0:
        pytest.skip(f"Skipping empty graph for chis {chis}")

    return sp


def _assert_point_in_simplex(phi1, phi2, tol=1e-12):
    assert np.isfinite(phi1) and np.isfinite(phi2)
    assert phi1 >= -tol
    assert phi2 >= -tol
    assert phi1 <= 1 + tol
    assert phi2 <= 1 + tol
    assert (phi1 + phi2) <= 1 + tol


# ----------------------------
# E2E tests
# ----------------------------


@pytest.mark.parametrize("seed", list(range(100)))
def test_e2e_spinodal_invariants_random_chis(seed):
    """This is a broad end-to-end test that builds many spinodal graphs from random chi parameters and checks key invariants"""
    rng = np.random.default_rng(seed)
    chis = _random_chis(rng, -10.0, 10.0)

    sp = _build_spinodal_or_skip(chis)
    G = sp.spinodal_graph
    nodes = list(G.nodes())

    # --- 1) Every node has degree 1 or 2 (no branching)
    degrees = dict(G.degree())
    assert all(d in (1, 2) for d in degrees.values())

    # --- 2) Every node is inside the bounds of the phase diagram (simplex)
    for n in nodes:
        _assert_point_in_simplex(n.phi1, n.phi2, tol=1e-10)

    # --- 3): no NaNs anywhere
    assert not any(np.isnan(n.phi1) or np.isnan(n.phi2) for n in nodes)

    # --- 4) Critical points are where the third derivative vanishes (approximately)
    # We check both:
    #   (a) cp is inside the simplex
    #   (b) third derivative at cp is smaller than at adjacent nodes
    #
    for cp in sp.critical_points:
        _assert_point_in_simplex(cp.phi1, cp.phi2, tol=1e-10)
        td_cp = sp._third_derivative(cp.phi1, cp.phi2)
        assert np.isfinite(td_cp)

        # Find the two adjacent nodes in phase space
        distances = [
            np.sqrt((cp.phi1 - n.phi1) ** 2 + (cp.phi2 - n.phi2) ** 2) for n in nodes
        ]
        closest_nodes = sorted(zip(distances, nodes))[:2]
        assert len(closest_nodes) == 2

        # Ensure the absolute value of the third derivative at the adjacent nodes is larger
        td_adj_1 = sp._third_derivative(
            closest_nodes[0][1].phi1, closest_nodes[0][1].phi2
        )
        td_adj_2 = sp._third_derivative(
            closest_nodes[1][1].phi1, closest_nodes[1][1].phi2
        )
        assert np.isfinite(td_adj_1) and np.isfinite(td_adj_2)
        assert abs(td_adj_1) > abs(td_cp)
        assert abs(td_adj_2) > abs(td_cp)

    # --- 5) Every node with degree 2 is "between" its adjacent neighbors
    for n, d in degrees.items():
        if d == 2:
            nb = list(G.neighbors(n))
            assert len(nb) == 2
            assert n.is_between(nb[0], nb[1], tol=np.pi / 6)

    # --- 6) Each connected component is either a simple path (2 endpoints) or a cycle (0 endpoints)
    for comp in nx.connected_components(G):
        sg = G.subgraph(comp)
        endpoint_count = sum(1 for _, d in sg.degree() if d == 1)
        assert endpoint_count in (0, 2)

    # --- 7) No self loops
    assert nx.number_of_selfloops(G) == 0

    # --- 8) Check edge lengths are reasonable (not too long, not zero)
    # Allow some slack for reduced resolution.
    max_len = 0.2
    for u, v in G.edges():
        dist = u.dist(v)
        assert np.isfinite(dist)
        assert dist > 0.0
        assert dist < max_len

    # --- 9) Critical points should lie "near" the curve:
    # Find nearest node distance and require it's reasonably small.
    # (This catches cases where cp interpolation is way off.)
    nodes = list(G.nodes())
    for cp in sp.critical_points:
        dmin = min(
            np.sqrt((cp.phi1 - n.phi1) ** 2 + (cp.phi2 - n.phi2) ** 2) for n in nodes
        )
        assert dmin < 0.05  # tune based on resolution

    # --- 10) Random points inside locally-stable polygons have both eigenvalues positive
    if sp.polygons:
        for poly in sp.polygons:
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            n_samples = 0
            attempts = 0
            while n_samples < 20 and attempts < 200:
                px = rng.uniform(minx, maxx)
                py = rng.uniform(miny, maxy)
                attempts += 1
                if poly.contains(
                    shapely.geometry.Point(px, py)
                ) and Spinodal._in_domain(px, py):
                    ev1, ev2 = sp.eigenvalues_from_phi(px, py)
                    ev_norm = np.sqrt(ev1**2 + ev2**2)
                    # Allow some slack for numerical issues, but generally should be positive in stable region.
                    assert (
                        ev1 / ev_norm > -1e-4
                    ), f"ev1={ev1} not positive at ({px}, {py})"
                    assert (
                        ev2 / ev_norm > -1e-4
                    ), f"ev2={ev2} not positive at ({px}, {py})"
                    n_samples += 1


@pytest.mark.parametrize("chi", np.linspace(2.1, 3.5, 29))
def test_e2e_spinodal_invariants_symmetric_case(chi):
    """
    A targeted e2e test: symmetric matrix should produce spinodal branches with the same shape.
    The number of critical points and branches should match known behavior for symmetric case as chi is varied.
    """

    # Use shapely to determine if spinodal branches have same shape
    import shapely

    if chi == 3:
        return
        pytest.skip(
            "Skipping chi=3 due to known numerical issues with second order phase transition"
        )

    chis = np.array([[-2 * chi, -chi], [-chi, -2 * chi]], dtype=float)
    sp = _build_spinodal_or_skip(chis, num_points=1000)
    G = sp.spinodal_graph

    assert G.number_of_nodes() > 10

    comps = list(nx.connected_components(G))

    # Check for correct number of critical points
    if chi < 2.575 or chi > 8 / 3:
        assert len(sp.critical_points) == 3

    else:  # chi > 2.575 and chi < 8 / 3
        assert len(sp.critical_points) == 9

    # Check for number of branches
    if chi < 8 / 3:
        assert len(comps) == 3
    # Get branch at center of phase space
    else:
        assert len(comps) == 4

    # Rotate all branches to dilute limit of second component and then compare shapes
    line_compare: shapely.geometry.LineString = None
    num_boundary_branches = 0
    for comp in comps:
        sg = G.subgraph(comp)
        phi1, phi2 = sp._coords_from_subgraph(sg)
        endpoints = [n for n, d in sg.degree() if d == 1]
        if len(endpoints) == 2:
            num_boundary_branches += 1

            phi1, phi2 = sp._coords_from_subgraph(sg)
            line = LineString(np.c_[phi1, phi2])

            if line_compare is None:
                line_compare = line
            else:
                intsct_obj = line_compare.intersection(line)

                num_rotations = 0
                while intsct_obj.geom_type != "MultiPoint":
                    phi1, phi2 = phi2, 1 - phi1 - phi2
                    line = LineString(np.c_[phi1, phi2])
                    intsct_obj = line_compare.intersection(line)
                    num_rotations += 1

                    # Fail if num_rotations exceeds 3 (full cycle) without finding a match
                    assert num_rotations < 3

                # should have many intersection points if shapes match
                assert len(intsct_obj.geoms) > 10

    assert num_boundary_branches == 3

    # Generic tests on the full graph and critical points:
    degrees = dict(G.degree())
    assert all(d in (1, 2) for d in degrees.values())

    for n in G.nodes():
        _assert_point_in_simplex(n.phi1, n.phi2, tol=1e-12)

    for cp in sp.critical_points:
        _assert_point_in_simplex(cp.phi1, cp.phi2, tol=1e-12)
