import numpy as np
import pytest
import jax.numpy as jnp

from pyphasediagram.stepper.flory_stepper import Stepper

# TODO: unit test where _projection_loop WILL fail


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


@pytest.fixture()
def stepper(chis):
    obj = Stepper(chis)
    # keep tests fast
    obj.MAX_STEPS = 50
    return obj


def test_residual_shape_and_finite(stepper, phi):
    r = stepper.residual(phi)
    assert r.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_is_terminate_false_for_valid_far_apart(stepper, phi):
    # with phi fixture, phases are not too close and composition is valid
    assert bool(stepper.is_terminate(phi)) is False


def test_is_terminate_true_when_phases_close(stepper):
    # phases nearly equal
    phi_close = jnp.array([0.2, 0.3, 0.2001, 0.3001], dtype=jnp.float64)
    assert bool(stepper.is_terminate(phi_close)) is True


def test_is_terminate_true_when_invalid(stepper):
    # negative component
    phi_neg = jnp.array([-0.1, 0.2, 0.3, 0.2], dtype=jnp.float64)
    assert bool(stepper.is_terminate(phi_neg)) is True

    # sum > 1 in a phase
    phi_sum = jnp.array([0.8, 0.4, 0.2, 0.2], dtype=jnp.float64)
    assert bool(stepper.is_terminate(phi_sum)) is True


def test_binary_state_returns_none_for_small_chi(stepper, capsys):
    x = stepper.binary_state(1.5)
    assert x is None
    out = capsys.readouterr().out
    assert "too low for phase separation" in out


def test_binary_state_solves_equation_for_large_chi(stepper):
    chi = 3.0
    x = stepper.binary_state(chi)
    assert x is not None
    assert 0.0 < x < 0.5  # with initial guess 1e-4, should converge to dilute solution

    # Verify the defining equation is close to zero
    val = np.log(x / (1.0 - x)) + chi * (1.0 - 2.0 * x)
    assert abs(val) < 1e-10


@pytest.mark.parametrize(
    "which_comp, expected_v_init",
    [
        (0, np.array([-1, -1, -1, -1])),
        (1, np.array([1, 0, 1, 0])),
        (2, np.array([0, 1, 0, 1])),
    ],
)
def test_binary_init_returns_valid_initial_point(stepper, which_comp, expected_v_init):
    phi_init, v_init = stepper.binary_init(which_comp)

    assert phi_init is not None
    assert v_init is not None

    phi_init = np.asarray(phi_init, dtype=float)
    v_init = np.asarray(v_init)

    assert phi_init.shape == (4,)
    assert v_init.shape == (4,)
    assert np.all(v_init == expected_v_init)

    # Valid composition constraints
    assert np.all(phi_init >= 1e-12)
    assert (phi_init[0] + phi_init[1]) <= 1.0 - 1e-12
    assert (phi_init[2] + phi_init[3]) <= 1.0 - 1e-12

    # Check symmetry properties given symmetric chis
    if which_comp == 0:
        assert np.isclose(phi_init[0], phi_init[3], atol=1e-6)
        assert np.isclose(phi_init[1], phi_init[2], atol=1e-6)

    # Should be on/very near the manifold after projection in binary_init
    r = np.asarray(stepper._residual_jit(jnp.asarray(phi_init)))
    assert np.linalg.norm(r) < 1e-7


def test_run_executes_from_binary_init(stepper):
    # Use a binary_init that should exist for the chis fixture
    phi_init, v_init = stepper.binary_init(0)
    assert phi_init is not None and v_init is not None

    phis, svs, flags = stepper.run(
        np.asarray(phi_init), np.asarray(v_init), delta_0=2e-4, delta_1=1e-3
    )

    assert isinstance(phis, np.ndarray)
    assert isinstance(svs, np.ndarray)

    # phis is (2 phases, 2 comps, K)
    assert phis.ndim == 3
    assert phis.shape[0] == 2
    assert phis.shape[1] == 2
    assert phis.shape[2] == svs.shape[0]
    assert phis.shape[2] >= 1

    # sanity: values finite and roughly valid compositions
    assert np.all(np.isfinite(phis))
    assert np.all(phis >= 1e-12)
    sums = phis.sum(axis=1)  # (2, K)
    assert np.all(sums <= 1.0 - 1e-12)


def test_run_points_are_near_manifold(stepper):
    """
    Check that a handful of produced points satisfy residual ~ 0.
    Keep it light: only check first few points.
    """
    phi_init, v_init = stepper.binary_init(0)
    phis, _, flags = stepper.run(
        np.asarray(phi_init), np.asarray(v_init), delta_0=2e-4, delta_1=1e-3
    )

    phis_T = np.transpose(phis, axes=(2, 0, 1)).reshape(-1, 4)
    for phi in phis_T:
        r = stepper._residual_jit(jnp.asarray(phi))
        assert np.linalg.norm(r) < 1e-7
