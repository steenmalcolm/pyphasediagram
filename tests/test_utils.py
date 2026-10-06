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

    @pytest.mark.parametrize("shape", [(4, 4), (3, 2), (2, 3), (3,), (9,)])
    def test_rejects_invalid_shape(self, shape):
        with pytest.raises(ValueError, match="2x2 or a 3x3"):
            reduce_chis(np.zeros(shape))


# ---------------------------------------------------------------------------
# free_energy
# ---------------------------------------------------------------------------
class TestFreeEnergy:

    def test_ideal_mixture_value(self):
        """With zero interactions, f is the ideal entropy of mixing."""
        phi = np.array([0.2, 0.3])
        expected = 0.2 * np.log(0.2) + 0.3 * np.log(0.3) + 0.5 * np.log(0.5)
        assert np.isclose(free_energy(phi, np.zeros((3, 3))), expected)

    def test_interaction_term_has_factor_one_half(self, chis_reduced):
        phi = np.array([0.2, 0.3])
        ideal = free_energy(phi, np.zeros((2, 2)))
        interaction = 0.5 * phi @ chis_reduced @ phi
        assert np.isclose(free_energy(phi, chis_reduced), ideal + interaction)

    def test_full_and_reduced_chis_agree(self, chis_full, chis_reduced):
        phi = np.array([0.25, 0.35])
        assert np.isclose(free_energy(phi, chis_full), free_energy(phi, chis_reduced))

    def test_single_composition_is_scalar(self, chis_full):
        assert np.ndim(free_energy([0.2, 0.3], chis_full)) == 0

    def test_vectorized_matches_loop(self, chis_full):
        phis = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.1]])  # (2, 3)
        f = free_energy(phis, chis_full)
        assert f.shape == (3,)
        for k in range(3):
            assert np.isclose(f[k], free_energy(phis[:, k], chis_full))

    def test_gradient_is_exchange_chemical_potential(self, chis_full):
        """mu_i - mu_0 = df/dphi_i."""
        phi = np.array([0.2, 0.3])
        h = 1e-6
        grad = np.array(
            [
                (free_energy(phi + h * e, chis_full) - free_energy(phi - h * e, chis_full))
                / (2 * h)
                for e in np.eye(2)
            ]
        )
        assert np.allclose(grad, exchange_chemical_potentials(phi, chis_full), atol=1e-6)

    @pytest.mark.parametrize("shape", [(3,), (1, 4), ()])
    def test_rejects_invalid_phi_shape(self, chis_full, shape):
        with pytest.raises(ValueError, match="leading axis of length 2"):
            free_energy(np.full(shape, 0.2), chis_full)

    def test_rejects_invalid_chis_shape(self):
        with pytest.raises(ValueError, match="2x2 or a 3x3"):
            free_energy([0.2, 0.3], np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# exchange_chemical_potentials
# ---------------------------------------------------------------------------
class TestExchangeChemicalPotentials:

    def test_ideal_mixture_value(self):
        phi = np.array([0.2, 0.3])
        mu = exchange_chemical_potentials(phi, np.zeros((3, 3)))
        assert np.allclose(mu, np.log(phi) - np.log(0.5))

    def test_known_value_with_interactions(self, chis_reduced):
        phi = np.array([0.2, 0.3])
        expected = np.log(phi) - np.log(0.5) + chis_reduced @ phi
        assert np.allclose(exchange_chemical_potentials(phi, chis_reduced), expected)

    def test_full_and_reduced_chis_agree(self, chis_full, chis_reduced):
        phi = np.array([0.25, 0.35])
        assert np.allclose(
            exchange_chemical_potentials(phi, chis_full),
            exchange_chemical_potentials(phi, chis_reduced),
        )

    def test_output_shape_matches_input(self, chis_full):
        assert exchange_chemical_potentials([0.2, 0.3], chis_full).shape == (2,)
        phis = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.1]])
        assert exchange_chemical_potentials(phis, chis_full).shape == (2, 3)

    def test_vectorized_matches_loop(self, chis_full):
        phis = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.1]])
        mu = exchange_chemical_potentials(phis, chis_full)
        for k in range(3):
            assert np.allclose(mu[:, k], exchange_chemical_potentials(phis[:, k], chis_full))

    def test_matches_stepper_residual(self, chis_full):
        """The difference between two phases equals the stepper's chemical-potential residual."""
        phi_a = np.array([0.1, 0.2])
        phi_b = np.array([0.3, 0.4])
        stepper = Stepper(reduce_chis(chis_full))
        residual = np.asarray(stepper.residual(np.concatenate([phi_a, phi_b])))
        delta_mu = exchange_chemical_potentials(
            phi_a, chis_full
        ) - exchange_chemical_potentials(phi_b, chis_full)
        assert np.allclose(residual[:2], delta_mu)

    @pytest.mark.parametrize("shape", [(3,), (1, 4), ()])
    def test_rejects_invalid_phi_shape(self, chis_full, shape):
        with pytest.raises(ValueError, match="leading axis of length 2"):
            exchange_chemical_potentials(np.full(shape, 0.2), chis_full)

    def test_rejects_invalid_chis_shape(self):
        with pytest.raises(ValueError, match="2x2 or a 3x3"):
            exchange_chemical_potentials([0.2, 0.3], np.zeros((4, 4)))
