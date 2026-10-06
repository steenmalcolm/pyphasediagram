import numpy as np
import pytest

from pyphasediagram.diagram import PhaseDiagram


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
