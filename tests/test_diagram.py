from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from mpltern import TernaryAxes
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon

from pyphasediagram.diagram import PhaseDiagram, _polygon_exteriors


@pytest.fixture
def diagram_with_plot_data():
    """Return a phase diagram with lightweight stand-ins for built results."""
    diagram = PhaseDiagram(np.zeros((3, 3)))

    spinodal_graph = nx.Graph()
    spinodal_graph.add_edge((0.20, 0.20), (0.30, 0.25))

    def coords_from_subgraph(subgraph):
        coords = np.asarray(list(subgraph.nodes), dtype=float)
        return coords[:, 0], coords[:, 1]

    diagram.spinodal = SimpleNamespace(
        spinodal_graph=spinodal_graph,
        _coords_from_subgraph=coords_from_subgraph,
        critical_points=[SimpleNamespace(phi1=0.25, phi2=0.25)],
    )

    section = SimpleNamespace(
        phis=np.array(
            [
                [[0.10, 0.15], [0.20, 0.25]],
                [[0.60, 0.55], [0.10, 0.15]],
            ]
        )
    )
    two_phase_polygons = [
        MultiPolygon(
            [
                Polygon([(0.10, 0.10), (0.20, 0.10), (0.15, 0.20)]),
                Polygon([(0.55, 0.10), (0.65, 0.10), (0.60, 0.20)]),
            ]
        )
    ]
    three_phase_polygons = [
        Polygon([(0.20, 0.20), (0.45, 0.20), (0.20, 0.45)])
    ]
    diagram.binodal = SimpleNamespace(
        binodal_sections=[section],
        two_phase_polygons=two_phase_polygons,
        three_phase_polygons=three_phase_polygons,
    )
    return diagram


def test_phase_diagram_reduces_full_interaction_matrix():
    chis = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    diagram = PhaseDiagram(chis)

    expected = np.array([[-2.0, 0.0], [0.0, -4.0]])
    np.testing.assert_array_equal(diagram.chis, expected)


@pytest.mark.parametrize(
    "chis",
    [
        pytest.param(np.zeros((2, 2)), id="reduced-matrix"),
        pytest.param(np.zeros((3,)), id="one-dimensional"),
        pytest.param(np.zeros((4, 4)), id="four-components"),
    ],
)
def test_phase_diagram_rejects_non_3x3_interaction_matrix(chis):
    with pytest.raises(ValueError, match="requires a 3x3 interaction matrix"):
        PhaseDiagram(chis)


def test_to_ternary_coordinates_adds_implicit_component():
    phi1 = np.array([0.20, 0.30])
    phi2 = np.array([0.30, 0.10])

    phi0, converted_phi1, converted_phi2 = PhaseDiagram._to_ternary_coordinates(
        phi1, phi2
    )

    np.testing.assert_allclose(phi0, [0.50, 0.60])
    np.testing.assert_array_equal(converted_phi1, phi1)
    np.testing.assert_array_equal(converted_phi2, phi2)


def test_polygon_exteriors_recurses_into_geometry_collections():
    polygon = Polygon([(0.10, 0.10), (0.20, 0.10), (0.15, 0.20)])
    nested_polygon = Polygon([(0.50, 0.10), (0.60, 0.10), (0.55, 0.20)])
    collection = GeometryCollection(
        [
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            polygon,
            MultiPolygon([nested_polygon]),
        ]
    )

    exteriors = list(_polygon_exteriors([collection]))

    assert len(exteriors) == 2
    np.testing.assert_array_equal(exteriors[0], np.asarray(polygon.exterior.coords))
    np.testing.assert_array_equal(
        exteriors[1], np.asarray(nested_polygon.exterior.coords)
    )


@pytest.mark.parametrize("method_name", ["plot", "plot_phase_counts"])
def test_phase_diagram_plot_methods_create_ternary_axes(
    diagram_with_plot_data, method_name
):
    method = getattr(diagram_with_plot_data, method_name)

    fig, ax = method()

    try:
        assert isinstance(ax, TernaryAxes)
        assert ax.get_figure() is fig
        assert ax.get_tlabel() == r"$\phi_0$"
        assert ax.get_llabel() == r"$\phi_1$"
        assert ax.get_rlabel() == r"$\phi_2$"
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_phase_diagram_plot_uses_phi0_phi1_phi2_axis_order(
    diagram_with_plot_data,
):
    fig, ax = diagram_with_plot_data.plot()

    try:
        expected_tlr = np.array(
            [
                [0.60, 0.20, 0.20],
                [0.45, 0.30, 0.25],
            ]
        )
        expected_xy = ax.transProjection.transform(expected_tlr)
        actual_xy = np.column_stack(ax.lines[0].get_data())

        np.testing.assert_allclose(actual_xy, expected_xy)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("method_name", ["plot", "plot_phase_counts"])
def test_phase_diagram_plot_methods_always_create_new_figures(
    diagram_with_plot_data, method_name
):
    method = getattr(diagram_with_plot_data, method_name)

    first_fig, first_ax = method()
    second_fig, second_ax = method()

    try:
        assert first_fig is not second_fig
        assert first_ax is not second_ax
    finally:
        plt.close(first_fig)
        plt.close(second_fig)


@pytest.mark.parametrize("method_name", ["plot", "plot_phase_counts"])
def test_phase_diagram_plot_methods_do_not_accept_axes(
    diagram_with_plot_data, method_name
):
    method = getattr(diagram_with_plot_data, method_name)

    with pytest.raises(TypeError):
        method(ax=None)


def test_plot_phase_counts_fills_polygon_and_multipolygon_parts(
    diagram_with_plot_data,
):
    fig, ax = diagram_with_plot_data.plot_phase_counts()

    try:
        # One domain polygon, two parts of the MultiPolygon, and one
        # three-phase polygon.
        assert len(ax.patches) == 4
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "1 phase",
            "2 phases",
            "3 phases",
        ]
    finally:
        plt.close(fig)


def test_plot_summary_uses_two_ternary_axes(diagram_with_plot_data):
    fig, axes = diagram_with_plot_data.plot_summary()

    try:
        assert len(axes) == 2
        assert all(isinstance(ax, TernaryAxes) for ax in axes)
        fig.canvas.draw()
    finally:
        plt.close(fig)
