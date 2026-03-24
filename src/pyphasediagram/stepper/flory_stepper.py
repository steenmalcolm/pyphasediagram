import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

from pyphasediagram.stepper.base import BaseStepper


class Stepper(BaseStepper):
    """Coexistence curve stepper for a ternary mixture."""

    def __init__(self, chis):
        self.chis = jnp.asarray(chis)  # (2,2)
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
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
        # IMPORTANT: This must be JAX-traceable for the fully-jitted run loop.

        close = jnp.linalg.norm(phi[2:] - phi[:2]) < 1e-3

        invalid = jnp.logical_or(jnp.any(phi < 0.0), (phi[0] + phi[1]) > 1.0)
        invalid = jnp.logical_or(invalid, (phi[2] + phi[3]) > 1.0)

        return jnp.logical_or(close, invalid)

    def binary_state(self, chi) -> float:
        """Find the binary coexistence point for the given Flory parameter."""
        # This is initialization (SciPy); it does not need to be jitted to run the main stepper loop.
        if chi <= 2:
            print(f"Warning: Chi value {chi:.2f}<=2 too low for phase separation")
            return None

        eq_func = lambda x: np.log(x / (1 - x)) + chi * (1 - 2 * x)
        sol = root(eq_func, 1e-4)
        if not sol.success:
            raise ValueError(f"Root finding failed: {sol.message}")
        return float(sol.x.item())

    def binary_init(self, which_comp: int = 0) -> jnp.ndarray:
        """Find an initial coexistence point for the dilute limit of one component"""

        phi_init: np.ndarray = None
        v_init: np.ndarray = None
        phi_bin: float = None

        if (
            which_comp == 0
        ):  # Phase separation between components 1 and 2, component 0 is dilute
            chi = self.chis[0, 1] - 0.5 * (self.chis[0, 0] + self.chis[1, 1])
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            v_init = np.array([-1, -1, -1, -1])

        # Phase separation between components 0 and 2, component 1 is dilute
        elif which_comp == 1:
            chi = -0.5 * self.chis[1, 1]
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[0] = 1 - phi_init[0] - phi_init[1]
            phi_init[2] = 1 - phi_init[2] - phi_init[3]
            v_init = np.array([1, 0, 1, 0])

        # Phase separation between components 0 and 1, component 2 is dilute
        elif which_comp == 2:
            chi = -0.5 * self.chis[0, 0]
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[1] = 1 - phi_init[0] - phi_init[1]
            phi_init[3] = 1 - phi_init[2] - phi_init[3]
            v_init = np.array([0, 1, 0, 1])

        # No phase separation in this binary limit, return None to indicate failure to initialize
        if phi_bin is None:
            return None, None

        # Keep initialization in NumPy/Python; convert to JAX arrays at the end
        res = np.asarray(self._residual_jit(jnp.asarray(phi_init)))
        max_iter = 10000
        iteration = 0

        while np.linalg.norm(res) > 1e-8:
            alpha = 1.0 if np.linalg.norm(res) < 1.0 else 0.1
            v, res_jax, _ = self._projection(jnp.asarray(phi_init))
            v = np.asarray(v)
            res = np.asarray(res_jax)
            # Use JAX-compatible termination check (works both in and out of jit)
            while bool(self.is_terminate(jnp.asarray(phi_init + v * alpha))):
                alpha *= 0.5
            phi_init += v * alpha

            if iteration > max_iter:
                # TODO: remove this stamement
                return None, None
                raise RuntimeError("In binary_init: Projection not converging")
            iteration += 1

        return np.asarray(phi_init), np.asarray(v_init)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import flory

    np.random.seed(42)
    for i in range(100):
        # Test binary_init and compare it to flory
        chi_12, chi_01 = np.random.uniform(-30, -10, 2)
        chi_02 = np.random.random() + 2.5

        chis = np.array(
            [
                [-2 * chi_01, chi_12 - chi_01 - chi_02],
                [chi_12 - chi_01 - chi_02, -2 * chi_02],
            ]
        )
        chis = np.array([[21.97142774, -14.75547769], [-14.75547769, -6.46398788]])
        obj = Stepper(chis)
        phi_init, v_init = obj.binary_init(1)

        chi_01, chi_02, chi_12 = (
            -chis[0, 0] / 2,
            -chis[1, 1] / 2,
            chis[0, 1] - (chis[0, 0] + chis[1, 1]) / 2,
        )
        chis_3 = np.array(
            [[0, chi_01, chi_02], [chi_01, 0, chi_12], [chi_02, chi_12, 0]]
        )
        if phi_init is not None:
            phi_means = [0, phi_initw[::2].mean(), phi_init[1::2].mean()]
        else:
            phi_means = [0, 1e-3, 0.5]
        phi_means[0] = 1 - phi_means[1] - phi_means[2]
        phases = flory.find_coexisting_phases(3, chis_3, phi_means)
        fracs = phases.fractions[:, 1:].flatten()
        plt.plot(fracs[::2], fracs[1::2], color="blue", alpha=0.5)
        if phi_init is not None:
            plt.scatter(phi_init[::2], phi_init[1::2], color="red", alpha=0.5)
            print(
                min(
                    np.linalg.norm(phi_init - fracs),
                    np.linalg.norm(np.roll(phi_init, 2) - fracs),
                )
            )

    import matplotlib.pyplot as plt
    plt.plot(phi_init[::2], phi_init[1::2])
