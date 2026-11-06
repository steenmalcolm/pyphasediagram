import numpy as np
import pytest

from pyphasediagram.stepper import Stepper
from pyphasediagram.utils import exchange_chemical_potentials, free_energy, reduce_chis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def chis_full():
    return np.array([[0.0, 2.7, 2.9], [2.7, 0.0, 3.1], [2.9, 3.1, 0.0]])


@pytest.fixture
def chis_reduced(chis_full):
    chi_01, chi_02, chi_12 = chis_full[0, 1], chis_full[0, 2], chis_full[1, 2]
    off = chi_12 - chi_01 - chi_02
    return np.array([[-2 * chi_01, off], [off, -2 * chi_02]])


@pytest.fixture
def phi():
    return np.array([0.2, 0.3])


@pytest.fixture
def phis():
    return np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.1]])


@pytest.fixture
def compositions(phi, phis):
    """Single composition and batches of one, two, and three compositions."""
    return [phi, phi[:, None], phis[:, :2], phis]


# ---------------------------------------------------------------------------
# reduce_chis
# ---------------------------------------------------------------------------
class TestReduceChis:
    def test_known_values(self, chis_full):
        expected = np.array([[-5.4, -2.5], [-2.5, -5.8]])
        assert np.allclose(reduce_chis(chis_full), expected)

    def test_shape_and_symmetry(self, chis_full):
        reduced = reduce_chis(chis_full)
        assert reduced.shape == (2, 2)
        assert np.allclose(reduced, reduced.T)

    def test_ignores_diagonal_entries(self, chis_full):
        modified = chis_full.copy()
        np.fill_diagonal(modified, [10.0, -3.0, 7.0])
        assert np.allclose(reduce_chis(modified), reduce_chis(chis_full))

    def test_reduced_input_returned_unchanged(self, chis_reduced):
        assert np.allclose(reduce_chis(chis_reduced), chis_reduced)

    def test_idempotent(self, chis_full):
        once = reduce_chis(chis_full)
        assert np.allclose(reduce_chis(once), once)

    def test_accepts_nested_lists(self, chis_full):
        assert np.allclose(reduce_chis(chis_full.tolist()), reduce_chis(chis_full))

    def test_four_component_known_values(self):
        chis = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ]
        )
        expected = np.array(
            [
                [-2.0, 1.0, 1.0],
                [1.0, -4.0, 1.0],
                [1.0, 1.0, -6.0],
            ]
        )

        assert np.allclose(reduce_chis(chis, ncomp=4), expected)

    def test_four_component_reduced_input_returned_unchanged(self):
        chis_reduced = np.array(
            [
                [-2.0, 1.0, 1.0],
                [1.0, -4.0, 1.0],
                [1.0, 1.0, -6.0],
            ]
        )

        assert np.allclose(reduce_chis(chis_reduced, ncomp=4), chis_reduced)

    @pytest.mark.parametrize("ncomp", [0, 1])
    def test_rejects_fewer_than_two_components(self, ncomp):
        with pytest.raises(ValueError, match="ncomp must be at least 2"):
            reduce_chis(np.zeros((1, 1)), ncomp=ncomp)

    @pytest.mark.parametrize("shape", [(4, 4), (3, 2), (2, 3), (3,), (9,)])
    def test_rejects_invalid_shape(self, shape):
        with pytest.raises(ValueError, match="must have shape"):
            reduce_chis(np.zeros(shape))

    def test_rejects_shape_inconsistent_with_ncomp(self):
        with pytest.raises(ValueError, match="must have shape"):
            reduce_chis(np.zeros((2, 2)), ncomp=4)

    def test_two_component_limit(self):
        chis = np.array(
            [
                [0.0, 3.0],
                [3.0, 0.0],
            ]
        )

        expected = np.array([[-6.0]])

        np.testing.assert_allclose(reduce_chis(chis, ncomp=2), expected)


# ---------------------------------------------------------------------------
# free_energy
# ---------------------------------------------------------------------------
class TestFreeEnergy:
    def test_ideal_mixture_value(self, phi):
        """With zero interactions, f is the ideal entropy of mixing."""
        expected = 0.2 * np.log(0.2) + 0.3 * np.log(0.3) + 0.5 * np.log(0.5)
        assert np.isclose(free_energy(phi, np.zeros((3, 3))), expected)

    def test_known_value_with_interactions(self, chis_full, compositions):
        """Compare with the full ternary free energy, including solvent interactions."""
        for phi in compositions:
            phi_0 = 1.0 - phi.sum(axis=0)
            ideal = phi_0 * np.log(phi_0) + np.sum(phi * np.log(phi), axis=0)
            interaction = (
                chis_full[0, 1] * phi_0 * phi[0]
                + chis_full[0, 2] * phi_0 * phi[1]
                + chis_full[1, 2] * phi[0] * phi[1]
            )
            actual = free_energy(phi, chis_full)
            assert actual.shape == phi.shape[1:]
            np.testing.assert_allclose(actual, ideal + interaction)

    def test_single_composition_is_scalar(self, chis_full, phi):
        assert np.ndim(free_energy(phi, chis_full)) == 0

    def test_vectorized_matches_loop(self, chis_full, phis):
        f = free_energy(phis, chis_full)
        assert f.shape == (3,)
        for k in range(3):
            assert np.isclose(f[k], free_energy(phis[:, k], chis_full))

    def test_gradient_is_exchange_chemical_potential(self, chis_full, phi):
        """mu_i - mu_0 = df/dphi_i."""
        h = 1e-6
        grad = np.array(
            [
                (free_energy(phi + h * e, chis_full) - free_energy(phi - h * e, chis_full))
                / (2 * h)
                for e in np.eye(2)
            ]
        )
        assert np.allclose(grad, exchange_chemical_potentials(phi, chis_full), atol=1e-6)


# ---------------------------------------------------------------------------
# exchange_chemical_potentials
# ---------------------------------------------------------------------------
class TestExchangeChemicalPotentials:
    def test_ideal_mixture_value(self, phi):
        mu = exchange_chemical_potentials(phi, np.zeros((3, 3)))
        assert np.allclose(mu, np.log(phi) - np.log(0.5))

    def test_known_value_with_interactions(self, chis_full, compositions):
        """Differentiate the full ternary interaction energy relative to the solvent."""
        for phi in compositions:
            phi_0 = 1.0 - phi.sum(axis=0)
            solvent_interaction = chis_full[0, 1] * phi[0] + chis_full[0, 2] * phi[1]
            expected = (
                np.log(phi)
                - np.log(phi_0)
                + np.array(
                    [
                        chis_full[0, 1] * phi_0 + chis_full[1, 2] * phi[1],
                        chis_full[0, 2] * phi_0 + chis_full[1, 2] * phi[0],
                    ]
                )
                - solvent_interaction
            )
            actual = exchange_chemical_potentials(phi, chis_full)
            assert actual.shape == phi.shape
            np.testing.assert_allclose(actual, expected)

    def test_output_shape_matches_input(self, chis_full, phi, phis):
        assert exchange_chemical_potentials(phi, chis_full).shape == (2,)
        assert exchange_chemical_potentials(phis, chis_full).shape == (2, 3)

    def test_vectorized_matches_loop(self, chis_full, phis):
        mu = exchange_chemical_potentials(phis, chis_full)
        for k in range(3):
            assert np.allclose(mu[:, k], exchange_chemical_potentials(phis[:, k], chis_full))

    def test_matches_stepper_residual(self, chis_full, phis):
        """The difference between two phases equals the stepper's chemical-potential
        residual."""
        phi_a, phi_b = phis[:, 0], phis[:, 1]
        stepper = Stepper(reduce_chis(chis_full))
        residual = np.asarray(stepper.residual(np.concatenate([phi_a, phi_b])))
        delta_mu = exchange_chemical_potentials(phi_a, chis_full) - exchange_chemical_potentials(
            phi_b, chis_full
        )
        assert np.allclose(residual[:2], delta_mu)


@pytest.mark.parametrize("function", [free_energy, exchange_chemical_potentials])
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (3, 2), (2, 3), (3,), (9,), ()])
def test_thermodynamic_functions_reject_invalid_chis_shape(function, shape, phi):
    with pytest.raises(ValueError, match=r"chis must have shape \(3, 3\)"):
        function(phi, np.zeros(shape))


@pytest.mark.parametrize("function", [free_energy, exchange_chemical_potentials])
def test_thermodynamic_functions_reject_reduced_chis(function, chis_reduced, phi):
    with pytest.raises(ValueError, match=r"chis must have shape \(3, 3\)"):
        function(phi, chis_reduced.tolist())


@pytest.mark.parametrize("function", [free_energy, exchange_chemical_potentials])
def test_thermodynamic_functions_accept_nested_lists(function, chis_full, compositions):
    for phi in compositions:
        actual = function(phi.tolist(), chis_full.tolist())
        np.testing.assert_allclose(actual, function(phi, chis_full))
