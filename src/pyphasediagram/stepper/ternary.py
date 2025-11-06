import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from pyphasediagram.stepper import BaseStepper


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


if __name__ == "__main__":
    import numpy as np
    import pyphasediagram as diag
    import matplotlib.pyplot as plt

    for i in range(10):
        chi_ds, chi_dr, chi_rs = (
            np.random.uniform(2.1, 4),
            np.random.uniform(0.0, 1.5),
            np.random.uniform(1.5, 2.0),
        )
        # chi_ds, chi_dr, chi_rs = (2.868522346358936, 0.08408085346877525, 1.9871917531982581)
        chi_matrix = np.array(
            [
                [-2 * chi_dr, chi_ds - chi_dr - chi_rs],
                [chi_ds - chi_dr - chi_rs, -2 * chi_rs],
            ]
        )

        stepper = TernaryStepper(chi_matrix)
        st_old = diag.BinodalStepperOld(chi_matrix)
        phi_list, _, _ = st_old.run()
        n = time.perf_counter()
        phi_list = stepper.run()
        print(f"Took {time.perf_counter()-n:.2f} seconds for JAX stepper")

        phi_arr_jax = np.array(phi_list)
        phi_r_den, phi_d_den, phi_r_dil, phi_d_dil = phi_arr_jax.T
        plt.subplot(121)
        plt.plot(phi_r_den, phi_d_den)
        plt.plot(phi_r_dil, phi_d_dil)
        plt.yticks([])
        plt.xticks([])

        n = time.perf_counter()
        phi_arr, _, _ = st_old.run()
        print(f"Took {time.perf_counter()-n:.2f} seconds for old stepper")

        phi_r_den, phi_d_den, phi_r_dil, phi_d_dil = (
            phi_arr[0, 0],
            phi_arr[0, 1],
            phi_arr[1, 0],
            phi_arr[1, 1],
        )
        plt.subplot(122)
        plt.plot(phi_r_den, phi_d_den)
        plt.plot(phi_r_dil, phi_d_dil)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f"delete_me_ternary_{i}.png")
        plt.close()

    # %%

    N = 10000
    phis = []
    for i in range(N):
        phi = np.random.random(4)
        while phi[0] + phi[1] > 1 or phi[2] + phi[3] > 1:
            phi = np.random.random(4)
        phis.append(phi)

    n = time.perf_counter()
    for i in range(N):

        J1 = stepper._jac_fn(phis[i])
        J2 = st_old.jacobian(phis[i])
        assert np.allclose(J1, J2), f"Jacobian mismatch for phi={phis[i]}"
        v_n1 = stepper.projection(phis[i])[0]
        v_n2 = st_old.projection(phis[i])
        assert np.allclose(v_n1, v_n2), f"Projection mismatch for phi={phis[i]}"
