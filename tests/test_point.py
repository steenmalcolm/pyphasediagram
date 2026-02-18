import pytest
import numpy as np
from pyphasediagram.point import Point, SpinodalPoint, CriticalPoint

# ----------------------------
# Tests: Point
# ----------------------------


def test_init_casts_types():
    pt = Point("3", "0.25", 0.5)
    assert isinstance(pt.idx, int)
    assert isinstance(pt.phi1, float)
    assert isinstance(pt.phi2, float)
    assert pt.idx == 3
    assert pt.phi1 == 0.25
    assert pt.phi2 == 0.5


def test_dist_euclidean():
    a = Point(0, 0.0, 0.0)
    b = Point(1, 3.0, 4.0)
    assert a.dist(b) == pytest.approx(5.0)
    assert b.dist(a) == pytest.approx(5.0)


def test_dist_euclidean_spinodal_point():
    a = SpinodalPoint(0, 0.0, 0.0)
    b = SpinodalPoint(1, 3.0, 4.0)
    assert a.dist(b) == pytest.approx(5.0)


def test_dist_raises_for_non_spinodalpoint():
    a = Point(0, 0.0, 0.0)
    with pytest.raises(NotImplementedError, match="Point instances"):
        _ = a.dist((0.0, 0.0))


def test_repr_formatting_three_decimals():
    pt = Point(0, 0.123456, 0.9)
    s = repr(pt)
    assert s.startswith("Point(")
    assert "phi1=0.123" in s
    assert "phi2=0.900" in s
    assert "0.123456" not in s


def test_add_returns_numpy_array_and_values():
    a = Point(0, 0.2, 0.3)
    b = Point(1, 0.4, 0.1)
    out = a + b
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([0.6, 0.4]))


def test_add_raises_for_non_spinodalpoint():
    a = Point(0, 0.2, 0.3)
    with pytest.raises(NotImplementedError, match="Addition can only be performed"):
        _ = a + 1


def test_sub_returns_numpy_array_and_values():
    a = Point(0, 0.2, 0.3)
    b = Point(1, 0.4, 0.1)
    out = a - b
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([-0.2, 0.2]))


def test_sub_raises_for_non_spinodalpoint():
    a = Point(0, 0.2, 0.3)
    with pytest.raises(NotImplementedError, match="Subtraction can only be performed"):
        _ = a - "nope"


def test_plot_calls_matplotlib_scatter(monkeypatch):
    # Patch the pyplot object
    import pyphasediagram.point as mod

    calls = {}

    def fake_scatter(x, y, s=10, color=None):
        calls["x"] = x
        calls["y"] = y
        calls["s"] = s
        calls["color"] = color

    monkeypatch.setattr(mod.plt, "scatter", fake_scatter)

    pt = Point(0, 0.12, 0.34)
    pt.plot(s=42, color="red")

    assert calls["x"] == 0.12
    assert calls["y"] == 0.34
    assert calls["s"] == 42
    assert calls["color"] == "red"


def test_is_between_true_for_colinear_opposite_directions():
    pt = Point(0, 0.0, 0.0)
    pt1 = Point(1, 1.0, 0.0)
    pt2 = Point(2, -1.0, 0.0)
    assert pt.is_between(pt1, pt2) == True


def test_is_between_false_for_right_angle():
    pt = Point(0, 0.0, 0.0)
    pt1 = Point(1, 1.0, 0.0)
    pt2 = Point(2, 0.0, 1.0)
    assert pt.is_between(pt1, pt2) == False


def test_is_between_raises_for_non_spinodalpoint():
    pt = Point(0, 0.0, 0.0)
    pt1 = Point(1, 1.0, 0.0)
    with pytest.raises(NotImplementedError, match="is_between can only be computed"):
        _ = pt.is_between(pt1, (0.0, 1.0))
