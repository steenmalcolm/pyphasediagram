import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    return np.array([0.20, 0.30, 0.25, 0.35], dtype=jnp.float64)


@pytest.fixture()
def v_init():
    """Initial guess for the Newton step."""
    return np.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float64)


class DummyStepper(BaseStepper):
    """Minimal concrete stepper for unit testing BaseStepper internals."""

    MAX_STEPS = 25  # keep tests fast

    def __init__(self, chis):
        self.chis = jnp.asarray(chis)
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
        This mirrors the Stepper residual structure.
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
        """Stop when phases become too similar or invalid composition."""
        close = jnp.linalg.norm(phi[2:] - phi[:2]) < 1e-3
        invalid = jnp.logical_or(jnp.any(phi < 0.0), (phi[0] + phi[1]) > 1.0)
        invalid = jnp.logical_or(invalid, (phi[2] + phi[3]) > 1.0)
        return jnp.logical_or(close, invalid)


@pytest.fixture()
def stepper(chis):
    return DummyStepper(chis)


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
    v_n, res, svmin = stepper._projection(phi)
    assert v_n.shape == (4,)
    assert res.shape == (3,)
    # svmin is scalar
    assert np.ndim(np.asarray(svmin)) == 0


def test_projection_reduces_residual_norm(stepper, phi):
    """
    One projection step should not increase residual norm (usually decreases).
    """
    r0 = jnp.linalg.norm(stepper.residual(phi))
    v_n, res, _ = stepper._projection(phi)
    r1 = jnp.linalg.norm(res)
    # res returned from _projection is the residual at phi (not at phi+v_n),
    # so compare against r0 computed consistently.
    assert np.isclose(np.asarray(r1), np.asarray(r0), atol=1e-10)

    # evaluate residual after applying the Newton step.
    phi2 = phi + v_n
    r2 = jnp.linalg.norm(stepper.residual(phi2))
    # Only decreases if residual is small to begin with.
    if r0 < 1:
        assert float(r2) < float(r0)


def test_projection_loop_converges_or_flags(stepper, phi):
    """
    Projection refinement should either converge or set the failed flag.
    """
    # _projection_loop is a helper used inside jit loop; it must be callable in eager too.
    phi_f, res_f, sv_f, failed = stepper._projection_loop(phi)

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

    phis, svs, flags = stepper.run(phi, np.asarray(v_init), delta_0=2e-4, delta_1=1e-3)

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
    phis, svs, flags = stepper.run(phi, v_init, delta_0=2e-4, delta_1=1e-3)

    # phis is (2 phases, 2 comps, K)
    assert np.all(np.isfinite(phis))
    assert np.all(phis >= -1e-12)  # allow tiny numerical undershoot
    sums = phis.sum(axis=1)  # (2, K)
    assert np.all(sums <= 1.0 + 1e-10)


########################################
# Flag tests
########################################


class FlagStepperBase(BaseStepper):
    """
    A deterministic BaseStepper for testing run() flag decoding.

    We override _projection/_projection_loop/_tangent_vec so we can:
      - avoid depending on jacobian/SVD numerics
      - deterministically trigger status bits
      - keep JIT-compatibility (all control flow stays in JAX)
    """

    MAX_STEPS = 12  # keep tests fast

    def __init__(self):
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        # Not used, but required.
        return jnp.zeros((3,), dtype=phi.dtype)

    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        # Default: never terminate (tests can override).
        return jnp.array(False)

    def _projection(self, phi: jnp.ndarray):
        # Return a harmless "projection" result so _run_impl can initialize sv0.
        v_n = jnp.zeros_like(phi)
        res = (
            jnp.zeros((3,), dtype=phi.dtype) + 1e-12
        )  # Projection will fail if exactly 0
        svmin = jnp.array(1.0, dtype=phi.dtype)
        return v_n, res, svmin

    def _projection_loop(self, phi: jnp.ndarray):
        # Default: no-op projection, never fails.
        phi_f = phi
        res_f = jnp.array(0.0, dtype=phi.dtype)
        sv_f = jnp.array(1.0, dtype=phi.dtype)
        failed = jnp.array(False)
        return phi_f, res_f, sv_f, failed

    def _tangent_vec(self, phi: jnp.ndarray) -> jnp.ndarray:
        # tangent step has no effect
        return jnp.array([0.0, 0.0, 0.0, 0.0], dtype=phi.dtype)


def test_run_sets_max_steps_flag(phi, v_init):
    class MaxStepsStepper(FlagStepperBase):
        # ensure cycle check can't stop us before MAX_STEPS
        CYCLE_MIN_STEPS = 10_000

        def is_terminate(self, phi):
            return jnp.array(False)

    stepper = MaxStepsStepper()
    phis, _, flags = stepper.run(phi, v_init, delta_0=1e-3, delta_1=1e-3)

    assert "MAX_STEPS" in flags
    assert phis.shape[-1] == stepper.MAX_STEPS + 1


def test_run_sets_cycled_flag(phi, v_init):
    class CycledStepper(FlagStepperBase):

        CYCLE_MIN_STEPS = 3  # Decrease to trigger cycle faster
        CYCLE_INIT_TOL = 1e-12  # require exact return

        def __init__(self, phi0):
            self._phi0 = jnp.asarray(phi0)
            super().__init__()

        def _projection_loop(self, phi):
            # Force the projection to always return exactly the initial point.
            # Then dist(phi - phi0) == 0, so CYCLED triggers once step>=CYCLE_MIN_STEPS.
            phi_f = self._phi0
            res_f = jnp.array(0.0, dtype=phi.dtype)
            sv_f = jnp.array(1.0, dtype=phi.dtype)
            failed = jnp.array(False)
            return phi_f, res_f, sv_f, failed

        def is_terminate(self, phi):
            return jnp.array(False)

    stepper = CycledStepper(phi)
    phis, _, flags = stepper.run(phi, v_init, delta_0=1e-3, delta_1=1e-3)
    phis_T = np.transpose(phis, (2, 0, 1)).reshape(-1, 4)

    assert "CYCLED" in flags
    assert (np.linalg.norm(phis_T - phi, axis=1) < stepper.CYCLE_INIT_TOL).any()


def test_run_sets_projection_fail_flag(phi, v_init):
    class ProjectionFailStepper(FlagStepperBase):
        CYCLE_MIN_STEPS = 10_000  # don't let cycle interfere

        def _projection_loop(self, phi):
            # Always report projection failure.
            phi_f = phi
            res_f = jnp.array(0.0, dtype=phi.dtype)
            sv_f = jnp.array(1.0, dtype=phi.dtype)
            failed = jnp.array(True)
            return phi_f, res_f, sv_f, failed

    stepper = ProjectionFailStepper()
    phis, _, flags = stepper.run(phi, v_init, delta_0=1e-3, delta_1=1e-3)

    assert "PROJECTION_FAIL" in flags


def test_run_sets_nan_flag(phi, v_init):
    class NanStepper(FlagStepperBase):
        CYCLE_MIN_STEPS = 10_000  # don't let cycle interfere

        def _projection_loop(self, phi):
            # Inject a NaN into the projected phi.
            phi_f = phi.at[0].set(jnp.nan)
            res_f = jnp.array(0.0, dtype=phi.dtype)
            sv_f = jnp.array(1.0, dtype=phi.dtype)
            failed = jnp.array(False)
            return phi_f, res_f, sv_f, failed

    stepper = NanStepper()
    phis, _, flags = stepper.run(phi, v_init, delta_0=1e-3, delta_1=1e-3)

    assert "NAN" in flags
    assert np.isnan(phis).any()


def test_run_sets_angle_fail_flag(phi, v_init):
    class AngleFailStepper(FlagStepperBase):
        CYCLE_MIN_STEPS = 10_000  # don't let cycle interfere

        def __init__(self, phi0):
            self._phi0 = jnp.asarray(phi0)
            super().__init__()

        def _projection_loop(self, phi):
            # We want:
            # - step 0 (trace start): move along e0
            # - step 1: move along e1 (orthogonal to previous step direction) -> ANGLE_FAIL
            #
            # We can distinguish the first call by checking whether we're still at phi0.
            dist = jnp.linalg.norm(phi - self._phi0)
            e0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=phi.dtype)
            e1 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=phi.dtype)

            # If we're basically at the start, return phi0 + small e0.
            # Otherwise return current + small e1 (orthogonal to the previous movement).
            phi_f = jnp.where(
                dist < 1e-12,
                self._phi0 + 1e-3 * e0,
                phi + 1e-3 * e1,
            )
            res_f = jnp.array(0.0, dtype=phi.dtype)
            sv_f = jnp.array(1.0, dtype=phi.dtype)
            failed = jnp.array(False)
            return phi_f, res_f, sv_f, failed

        def is_terminate(self, phi):
            return phi[1] > self._phi0[1]

    stepper = AngleFailStepper(phi)
    phis, _, flags = stepper.run(phi, v_init, delta_0=1e-3, delta_1=1e-3)

    assert "ANGLE_FAIL" in flags
