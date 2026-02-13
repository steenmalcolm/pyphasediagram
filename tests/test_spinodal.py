# test_spinodal.py
import numpy as np
import pytest
import networkx as nx

from pyphasediagram.stepper.spinodal import SpinodalPoint, Spinodal


# ----------------------------
# Tests: SpinodalPoint
# ----------------------------


def test_init_casts_types():
    pt = SpinodalPoint("3", "0.25", 0.5)
    assert isinstance(pt.idx, int)
    assert isinstance(pt.phi1, float)
    assert isinstance(pt.phi2, float)
    assert pt.idx == 3
    assert pt.phi1 == 0.25
    assert pt.phi2 == 0.5


def test_dist_euclidean():
    a = SpinodalPoint(0, 0.0, 0.0)
    b = SpinodalPoint(1, 3.0, 4.0)
    assert a.dist(b) == pytest.approx(5.0)
    assert b.dist(a) == pytest.approx(5.0)


def test_dist_raises_for_non_spinodalpoint():
    a = SpinodalPoint(0, 0.0, 0.0)
    with pytest.raises(NotImplementedError, match="SpinodalPoint instances"):
        _ = a.dist((0.0, 0.0))


def test_repr_formatting_three_decimals():
    pt = SpinodalPoint(0, 0.123456, 0.9)
    s = repr(pt)
    assert s.startswith("SpinodalPoint(")
    assert "phi1=0.123" in s
    assert "phi2=0.900" in s
    assert "0.123456" not in s


def test_add_returns_numpy_array_and_values():
    a = SpinodalPoint(0, 0.2, 0.3)
    b = SpinodalPoint(1, 0.4, 0.1)
    out = a + b
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([0.6, 0.4]))


def test_add_raises_for_non_spinodalpoint():
    a = SpinodalPoint(0, 0.2, 0.3)
    with pytest.raises(NotImplementedError, match="Addition can only be performed"):
        _ = a + 1


def test_sub_returns_numpy_array_and_values():
    a = SpinodalPoint(0, 0.2, 0.3)
    b = SpinodalPoint(1, 0.4, 0.1)
    out = a - b
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([-0.2, 0.2]))


def test_sub_raises_for_non_spinodalpoint():
    a = SpinodalPoint(0, 0.2, 0.3)
    with pytest.raises(NotImplementedError, match="Subtraction can only be performed"):
        _ = a - "nope"


def test_plot_calls_matplotlib_scatter(monkeypatch):
    # Patch the pyplot object as used inside your module (spinodal.plt)
    import pyphasediagram.stepper.spinodal as mod

    calls = {}

    def fake_scatter(x, y, s=10, color=None):
        calls["x"] = x
        calls["y"] = y
        calls["s"] = s
        calls["color"] = color

    monkeypatch.setattr(mod.plt, "scatter", fake_scatter)

    pt = SpinodalPoint(0, 0.12, 0.34)
    pt.plot(s=42, color="red")

    assert calls["x"] == 0.12
    assert calls["y"] == 0.34
    assert calls["s"] == 42
    assert calls["color"] == "red"


def test_is_between_true_for_colinear_opposite_directions():
    pt = SpinodalPoint(0, 0.0, 0.0)
    pt1 = SpinodalPoint(1, 1.0, 0.0)
    pt2 = SpinodalPoint(2, -1.0, 0.0)
    assert pt.is_between(pt1, pt2) == True


def test_is_between_false_for_right_angle():
    pt = SpinodalPoint(0, 0.0, 0.0)
    pt1 = SpinodalPoint(1, 1.0, 0.0)
    pt2 = SpinodalPoint(2, 0.0, 1.0)
    assert pt.is_between(pt1, pt2) == False


def test_is_between_raises_for_non_spinodalpoint():
    pt = SpinodalPoint(0, 0.0, 0.0)
    pt1 = SpinodalPoint(1, 1.0, 0.0)
    with pytest.raises(NotImplementedError, match="is_between can only be computed"):
        _ = pt.is_between(pt1, (0.0, 1.0))


# ----------------------------
# Tests: Spinodal
# ----------------------------


@pytest.fixture
def simple_chis():
    # a=b=1, c=0 is a nice baseline that tends to keep discriminants well-behaved
    return np.array([[-5, -2.5], [-5, -2.5]], dtype=float)


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

    with pytest.warns(RuntimeWarning, match=r"(invalid value|divide by zero)"):
        phi2a, phi2b = sp.phi2_from_phi1(phi1)

    assert phi2a.shape == phi1.shape
    assert phi2b.shape == phi1.shape
    # And explicitly acknowledge NaNs are allowed/expected:
    assert np.isnan(phi2a).any() or np.isnan(phi2b).any()


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

    def fake_domain_data(phi1_i, phi1_f):
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
