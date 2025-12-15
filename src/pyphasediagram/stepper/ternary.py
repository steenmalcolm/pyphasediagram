import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from pyphasediagram.stepper.base import BaseStepper


class TernaryStepper(BaseStepper):
    """Coexistence curve stepper for a ternary mixture."""

    def __init__(self, chi_matrix):
        self.chi_mat = chi_matrix  # (2,2)
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
        """
        phis_phase = phi.reshape(2, -1)  # (2, 2)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ self.chi_mat.T  # (2, 2)
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2,2)
        mu_diff = mu[0] - mu[1]  # (2,)

        pi = -jnp.log(phi0_phase) + 0.5 * (phis_phase * chi_prod).sum(axis=1)  # (2,)
        pi_diff = pi[0] - pi[1]  # ()

        return jnp.hstack((mu_diff, pi_diff))  # (3,)

    def is_terminate(self, phi: np.ndarray) -> bool:
        """Stop when phases become too similar or invalid composition."""

        close = np.linalg.norm(phi[2:] - phi[:2]) < 1e-3
        invalid = (phi < 0).any() or (phi[0] + phi[1]) > 1 or (phi[2] + phi[3]) > 1
        return close or invalid

    def binary_state(self) -> float:
        """Find the binary coexistence point for the given Flory parameter."""
        chi = self.chi_mat
        chi_12 = float(chi[0, 1] - 0.5 * (chi[0, 0] + chi[1, 1]))
        if chi_12 < 2:
            raise RuntimeError("Flory parameter too low for phase separation")

        eq_func = lambda x: np.log(x / (1 - x)) + chi_12 * (1 - 2 * x)
        sol = root(eq_func, 1e-4)
        if not sol.success:
            raise ValueError(f"Root finding failed: {sol.message}")
        return float(sol.x.item())

    def phi_init(self) -> jnp.ndarray:
        """Find an initial coexistence point near the binary limit."""

        # nearly-binary start
        x0 = self.binary_state()
        phi = np.array([x0, 1 - x0, 1 - x0, x0]) - 1e-4

        res = self._residual_jit(phi)
        max_iter = 1000
        iteration = 0

        while np.linalg.norm(res) > 1e-8:
            v, res = self.projection(phi)
            if not self.is_terminate(phi + v):
                phi = phi + v
            else:
                vv = v
                while self.is_terminate(phi - vv):
                    vv = vv * 0.5
                phi = phi - vv

            if iteration > max_iter:
                raise RuntimeError("In init_point: Projection not converging")
            iteration += 1

        return phi
