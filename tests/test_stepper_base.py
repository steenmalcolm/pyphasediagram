# tests/test_base_stepper.py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

# Import your BaseStepper from your package
from pyphasediagram.stepper.base import BaseStepper


@pytest.fixture(scope="module")
def chis():
    """
    Chi matrix for symmetric case
    """
    return jnp.array([[-5.0, -2.5], [-2.5, -5]], dtype=jnp.float64)


@pytest.fixture(scope="module")
def phi():
    """
    A valid composition vector for two phases (ternary mixture represented with 2 independent comps per phase).
    Shape is (4,) = (phi1_a, phi2_a, phi1_b, phi2_b).
    """
    # Ensure both phases have sum < 1 so phi0_phase = 1 - sum > 0
    return jnp.array([0.20, 0.30, 0.25, 0.35], dtype=jnp.float64)


class DummyTernaryStepper(BaseStepper):
    """Minimal concrete stepper for unit testing BaseStepper internals."""

    MAX_STEPS = 25  # keep tests fast

    def __init__(self, chis):
        self.chis = jnp.asarray(chis)
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
        This mirrors the TernaryStepper residual structure.
        """
        phis_phase = phi.reshape(2, -1)  # (2, 2)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ self.chis.T  # (2, 2)
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2,2)
        mu_diff = mu[0] - mu[1]  # (2,)

        pi = -jnp.log(phi0_phase) + 0.5 * (phis_phase * chi_prod).sum(axis=1)  # (2,)
        pi_diff = pi[0] - pi[1]  # ()

        return jnp.hstack((mu_diff, pi_diff))  # (3,)

    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Stop when phases become too similar or invalid composition (JAX-traceable)."""
        close = jnp.linalg.norm(phi[2:] - phi[:2]) < 1e-3
        invalid = jnp.logical_or(jnp.any(phi < 0.0), (phi[0] + phi[1]) > 1.0)
        invalid = jnp.logical_or(invalid, (phi[2] + phi[3]) > 1.0)
        return jnp.logical_or(close, invalid)


@pytest.fixture()
def stepper(chis):
    return DummyTernaryStepper(chis)


def test_residual_shape_and_finite(stepper, phi):
    r = stepper.residual(phi)
    assert r.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_jacobian_shape(stepper, phi):
    J = stepper._jac_fn(phi)
    assert J.shape == (3, 4)
    assert bool(jnp.all(jnp.isfinite(J)))


def test_tangent_is_in_right_nullspace(stepper, phi):
    """
    For a 3x4 Jacobian, the tangent should be in the right nullspace: J @ v ~= 0.
    """
    J = stepper._jac_fn(phi)
    v = stepper._tangent_vec(phi)

    assert v.shape == (4,)
    # Allow modest tolerance: depends on conditioning and SVD numerics.
    assert float(jnp.linalg.norm(J @ v)) < 1e-8


def test_projection_returns_correct_shapes(stepper, phi):
    v_n, E, svmin = stepper._projection(phi)
    assert v_n.shape == (4,)
    assert E.shape == (3,)
    # svmin is scalar
    assert np.ndim(np.asarray(svmin)) == 0


def test_projection_reduces_residual_norm(stepper, phi):
    """
    One projection step should not increase residual norm (usually decreases).
    """
    r0 = jnp.linalg.norm(stepper.residual(phi))
    v_n, E, _ = stepper._projection(phi)
    r1 = jnp.linalg.norm(E)
    # E returned from _projection is the residual at phi (not at phi+v_n),
    # so compare against r0 computed consistently.
    assert float(r1) <= float(r0) + 1e-12

    # More meaningful: evaluate residual after applying the Newton step.
    phi2 = phi + v_n
    r2 = jnp.linalg.norm(stepper.residual(phi2))
    assert float(r2) <= float(r0) + 1e-10


def test_proj_refine_converges_or_flags(stepper, phi):
    """
    Projection refinement should either converge or set the failed flag.
    """
    # _proj_refine is a helper used inside jit loop; it must be callable in eager too.
    phi_f, res_f, sv_f, failed = stepper._proj_refine(phi)

    assert phi_f.shape == (4,)
    assert np.ndim(np.asarray(res_f)) == 0
    assert np.ndim(np.asarray(sv_f)) == 0
    assert np.ndim(np.asarray(failed)) == 0

    # If it didn't fail, expect a small residual.
    if not bool(np.asarray(failed)):
        assert float(np.asarray(res_f)) < 1e-8


def test_run_executes_and_returns_shapes(stepper, phi):
    """
    End-to-end sanity test: run should return phis with shape (2, 2, K) and sv list length K.
    """
    # Use a float v_init to avoid dtype issues in lax.select
    v_init = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=phi.dtype)

    phis, svs = stepper.run(
        np.asarray(phi), np.asarray(v_init), delta_0=2e-4, delta_1=1e-3
    )

    assert isinstance(phis, np.ndarray)
    assert isinstance(svs, np.ndarray)

    assert phis.ndim == 3
    assert phis.shape[0] == 2
    assert phis.shape[1] == 2
    assert phis.shape[2] == svs.shape[0]
    assert phis.shape[2] >= 1


def test_run_points_are_valid_compositions(stepper, phi):
    """
    Basic invariants on output points:
    - each phase has components >= 0
    - sum of independent comps per phase <= 1
    """
    v_init = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=phi.dtype)
    phis, svs = stepper.run(
        np.asarray(phi), np.asarray(v_init), delta_0=2e-4, delta_1=1e-3
    )

    # phis is (2 phases, 2 comps, K)
    assert np.all(np.isfinite(phis))
    assert np.all(phis >= -1e-12)  # allow tiny numerical undershoot
    sums = phis.sum(axis=1)  # (2, K)
    assert np.all(sums <= 1.0 + 1e-10)
