import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from pyphasediagram.stepper.base import BaseStepper
from pyphasediagram.stepper.point import BinodalPoint


class TernaryStepper(BaseStepper):
    """Coexistence curve stepper for a ternary mixture."""

    def __init__(self, chis):
        self.chis = chis  # (2,2)
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

    def is_terminate(self, phi: np.ndarray) -> bool:
        """Stop when phases become too similar or invalid composition."""

        close = np.linalg.norm(phi[2:] - phi[:2]) < 1e-5
        invalid = (phi < 0).any() or (phi[0] + phi[1]) > 1 or (phi[2] + phi[3]) > 1
        return close or invalid

    def binary_state(self, chi) -> float:
        """Find the binary coexistence point for the given Flory parameter."""
        # chi_12 = float(chi[0, 1] - 0.5 * (chi[0, 0] + chi[1, 1]))
        if chi < 2:
            raise RuntimeError(f"Flory parameter must be > 2 but is {chi:.2f}")

        eq_func = lambda x: np.log(x / (1 - x)) + chi * (1 - 2 * x)
        sol = root(eq_func, 1e-4)
        if not sol.success:
            raise ValueError(f"Root finding failed: {sol.message}")
        return float(sol.x.item())

    def binary_init(self, which_comp: int = 0) -> jnp.ndarray:
        """Find an initial coexistence point for the dilute limit of one component"""
        # chi_12, chi_13, chi_23 = 2.9, 1.5, 1.5
        # chi_matrix = np.array(
        #     [
        #         [-2 * chi_13, chi_12 - chi_13 - chi_23],
        #         [chi_12 - chi_13 - chi_23, -2 * chi_23],
        #     ]

        # nearly-binary start
        if which_comp == 0:
            chi = self.chis[0, 1] - 0.5 * (self.chis[0, 0] + self.chis[1, 1])
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4

        elif which_comp == 1:
            chi = -0.5 * self.chis[1, 1]
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[0] = 1 - phi_init[0] - phi_init[1]
            phi_init[2] = 1 - phi_init[2] - phi_init[3]

        elif which_comp == 2:
            chi = -0.5 * self.chis[0, 0]
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[1] = 1 - phi_init[0] - phi_init[1]
            phi_init[3] = 1 - phi_init[2] - phi_init[3]

        res = self._residual_jit(phi_init)
        max_iter = 1000
        iteration = 0

        while np.linalg.norm(res) > 1e-8:
            v, res, _ = self._projection(phi_init)
            if not self.is_terminate(phi_init + v):
                phi_init = phi_init + v
            else:
                vv = v
                while self.is_terminate(phi_init - vv):
                    vv = vv * 0.5
                phi_init = phi_init - vv

            if iteration > max_iter:
                raise RuntimeError("In binary_init: Projection not converging")
            iteration += 1

        return phi_init


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    chi = 2.7
    for _ in range(10):
        chi_12, chi_13, chi_23 = np.random.random(3) + 2
        chi_matrix = np.array(
            [
                [-2 * chi_13, chi_12 - chi_13 - chi_23],
                [chi_12 - chi_13 - chi_23, -2 * chi_23],
            ]
        )
        obj = TernaryStepper(chi_matrix)
        n = time.perf_counter()
        phi_init = obj.binary_init(0)
        assert phi_init[:2].sum() < 1 and phi_init[2:].sum() < 1
        plt.scatter(
            phi_init[::2], phi_init[1::2], color="r", s=5, label="solvent dilute limit"
        )
        phi_init = obj.binary_init(1)
        assert (phi_init[::2] > 0).all()
        plt.scatter(
            phi_init[::2], phi_init[1::2], color="b", s=5, label="droplet dilute limit"
        )
        phi_init = obj.binary_init(2)
        assert (phi_init[1::2] > 0).all()
        plt.scatter(
            phi_init[::2],
            phi_init[1::2],
            color="g",
            s=5,
            label="regulator dilute limit",
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [1, 0], "k--")
    plt.show()
    print("end")

    # v_t_init = np.array([-1, -1, -1, -1], dtype=float)
    # obj.run(phi_init, v_t_init)
    # print(f"took {time.perf_counter() - n:.2f} seconds")

    # print(len(obj.bins_list))
    # plt.figure(figsize=(10, 10))
    # colormap = plt.cm.viridis
    # for i, b in enumerate(obj.bins_list):
    #     print(b.shape)
    #     plt.plot(b[0, 0], b[0, 1], "r")
    #     plt.plot(b[1, 0], b[1, 1], "b")
    #     color = colormap(i / len(obj.bins_list))
    #     plt.scatter(b[0, 0, 0], b[0, 1, 0], color=color, s=10, alpha=0.5)
    #     plt.scatter(b[1, 0, 0], b[1, 1, 0], color=color, s=10, alpha=0.5)
    #     plt.scatter(b[0, 0, -1], b[0, 1, -1], color=color, s=10, alpha=0.5)
    #     plt.scatter(b[1, 0, -1], b[1, 1, -1], color=color, s=10, alpha=0.5)
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
