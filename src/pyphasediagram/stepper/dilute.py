# %%
import time
import jax.numpy as jnp
import numpy as np
from pyphasediagram.stepper.base import BaseStepper
import flory


class DiluteStepper(BaseStepper):
    """Coexistence curve stepper for the limit of small droplet volume."""

    def __init__(self, chi_comp: np.ndarray, phi_dils: np.ndarray, N: int = 0):

        self.chi_comp = chi_comp  # (N+1,N+1)
        self.phi_dils = phi_dils  # (N-2,)
        self.N = N
        if isinstance(chi_comp, np.ndarray):
            self.chi_mat = self.chi_reduced(chi_comp)  # (N,N)
            assert (
                self.chi_mat.shape[0] - 2 == phi_dils.shape[0]
            ), f"Need to set {self.chi_mat.shape[0]-2} dilute compositions not {phi_dils.shape[0]}"
        else:
            assert N > 0, "Must provide N>0 to generate random chi matrix"
        super().__init__()

    def chi_reduced(self, chi):
        """Interaction matrix for incompressible system."""
        N = len(chi) - 1
        chi_r = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                chi_r[i, j] = chi[i + 1, j + 1] - chi[i + 1, 0] - chi[j + 1, 0]
        return chi_r

    def residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
        """
        phis_phase = jnp.stack([jnp.r_[self.phi_dils, x[:2]], x[2:]])  # (2, N)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ self.chi_mat.T  # (2, N)
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2, N)
        mu_diff = mu[0] - mu[1]  # (N,)

        pi = -jnp.log(phi0_phase) + 0.5 * (phis_phase * chi_prod).sum(axis=1)  # (2,)
        pi_diff = pi[0] - pi[1]  # ()

        return jnp.r_[mu_diff, pi_diff]  # (N+1,)

    def is_terminate(self, x: np.ndarray) -> bool:
        """Stop when phases become too similar or invalid composition."""

        phis_phase = np.stack([np.r_[self.phi_dils, x[:2]], x[2:]])  # (2, N)
        close = np.linalg.norm(phis_phase[1] - phis_phase[0]) < 1e-3
        invalid = (phis_phase < 0).any() or (phis_phase.sum(axis=1) > 1).any()
        return close or invalid

    def phi_init(self) -> jnp.ndarray:
        """Find an initial coexistence point near the binary limit."""
        # if isinstance(self.chi_comp, np.ndarray):
        #     phases = flory.find_coexisting_phases(
        #         self.chi_comp.shape[0],
        #         self.chi_comp,
        #         np.ones(self.N + 1) / (self.N + 1),
        #         progress=False,
        #     )
        #     fracs = phases.fractions
        #     assert (
        #         len(fracs) == 2
        #     ), "Need to find two coexisting phases for initialization"

        # else:
        #     fracs = []
        #     while len(fracs) != 2:
        #         # Compressible case N->N+1
        #         chi_comp = np.random.random((self.N + 1, self.N + 1)) + 3
        #         chi_comp = (chi_comp + chi_comp.T) / 2
        #         chi_comp -= np.diag(np.diag(self.chi_mat))
        #         phases = flory.find_coexisting_phases(
        #             self.N + 1,
        #             chi_comp,
        #             np.ones(self.N + 1) / (self.N + 1),
        #             progress=False,
        #         )
        #         fracs = phases.fractions

        #     self.phi_dils = fracs[0, 1:-2]
        #     self.chi_mat = np.zeros((self.N, self.N))
        #     for i in range(self.N):
        #         for j in range(self.N):
        #             self.chi_mat[i, j] = (
        #                 chi_comp[i + 1, j + 1] - chi_comp[i + 1, 0] - chi_comp[j + 1, 0]
        #             )

        # x = np.r_[fracs[0, -2:], fracs[1, 1:]]
        self.chi_comp = np.array(
            [
                [0.0, 3.4764248, 3.42753118, 3.18283335, 3.8735968],
                [3.4764248, 0.0, 3.24135875, 3.38022499, 3.66563028],
                [3.42753118, 3.24135875, 0.0, 3.43717238, 3.43450685],
                [3.18283335, 3.38022499, 3.43717238, 0.0, 3.72282364],
                [3.8735968, 3.66563028, 3.43450685, 3.72282364, 0.0],
            ]
        )
        self.chi_mat = self.chi_reduced(self.chi_comp)

        x = np.array(
            [0.20873904, 0.16561139, 0.04246615, 0.05092392, 0.03970867, 0.83087196]
        )

        # res = self._residual_jit(x)

        # i = 0
        # while np.linalg.norm(res) > 1e-8:
        #     v_n, res = self.projection(x)
        #     x += v_n
        #     print(f"{i}: {np.linalg.norm(res):.2e}")
        #     if i > 100:
        #         print("muddafukka you suck")
        #         break

        #     i += 1

        return x


# %%
# import matplotlib.pyplot as plt
# import numpy as np
# import flory

# # Compressible system
# chi_mat = np.array(
#     [
#         [0.0, 3.4764248, 3.42753118, 3.18283335, 3.8735968],
#         [3.4764248, 0.0, 3.24135875, 3.38022499, 3.66563028],
#         [3.42753118, 3.24135875, 0.0, 3.43717238, 3.43450685],
#         [3.18283335, 3.38022499, 3.43717238, 0.0, 3.72282364],
#         [3.8735968, 3.66563028, 3.43450685, 3.72282364, 0.0],
#     ]
# )

# # Sum of mean volume fractions = 1

# N = 4
# # fracs = []
# # while len(fracs) != 2:
# #     break
# #     print("Trying to find phases...")
# # chi_mat = np.random.random((N, N)) + 3
# # chi_mat = (chi_mat + chi_mat.T) / 2
# # chi_mat -= np.diag(np.diag(chi_mat))
# # phi_means = np.ones(N + 1) / (N + 1)
# # phases = flory.find_coexisting_phases(N + 1, chi_mat, phi_means, progress=False)
# # fracs = phases.fractions
# # assert len(phases.fractions) == 2
# # %%
# # x = np.r_[fracs[0, -2:], fracs[1, 1:]]
# x = np.array([0.20873904, 0.16561139, 0.04246615, 0.05092392, 0.03970867, 0.83087196])


# phi_dils = np.array([0.20858883, 0.20812768])


# def chi_reduced(chi):

#     N = len(chi) - 1
#     chi_r = np.zeros((N, N))

#     for i in range(N):
#         for j in range(N):
#             chi_r[i, j] = chi[i + 1, j + 1] - chi[i + 1, 0] - chi[j + 1, 0]

#     return chi_r


# chi_r = chi_reduced(chi_mat)
# chi_r
# # %%
# obj = SingleCompositionVaraition(chi_mat, phi_dils)
# # %%
# res = obj._residual_jit(x)
# while np.linalg.norm(res) > 1e-8:
#     v, res = obj.projection(x)
#     x += v


# # %%
# pos_T = obj.run(delta=1e-3)
# pos = pos_T.T
# # %%
# plt.figure(figsize=(6, 12))
# plt.subplot(3, 1, 1)
# plt.plot(pos[0], pos[1], label=r"$\phi^{(1)}$")
# plt.plot(pos[-2], pos[-1], label=r"$\phi^{(2)}$")
# plt.scatter(pos[0, 0], pos[1, 0], color="red", label="start")
# plt.scatter(pos[-2, 0], pos[-1, 0], color="red")
# plt.xlabel(r"$\phi_{1}$", fontsize=12)
# plt.ylabel(r"$\phi_{2}$", fontsize=12)
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(pos[2], pos[3], label=r"$\phi^{(2)}$")
# plt.scatter(pos[2, 0], pos[3, 0], c="r")
# plt.scatter(phi_dils[0], phi_dils[1], c="orange", label=r"$\phi^{(1)}$")
# plt.scatter(pos[2, 0], pos[3, 0], color="red", label="start")
# plt.xlabel(r"$\phi_{3}$", fontsize=12)
# plt.ylabel(r"$\phi_{4}$", fontsize=12)
# plt.legend()
# plt.subplot(3, 1, 3)
# phi0dil = 1 - phi_dils.sum() - pos[:2].sum(axis=0)
# phi0den = 1 - pos[2:].sum(axis=0)
# plt.plot(phi0dil, phi0den)
# plt.scatter(
#     phi0dil[0],
#     phi0den[0],
#     c="r",
# )
# plt.xlabel(r"$\phi_{0}^{(1)}$", fontsize=12)
# plt.ylabel(r"$\phi_{0}^{(2)}$", fontsize=12)
# plt.legend()
# plt.tight_layout()

# # %%
# res = [obj._residual_jit(p) for p in pos.T]
# # res = res[:-1]
# res = np.array(res).T
# for r in res:
#     plt.plot(abs(r))
# plt.yscale("log")

# # %%
# plt.plot(lmd)
# # %%
# phi1dil = pos[1]
# phi1den = pos[N]
# # plt.plot(phi1dil)
# plt.plot(phi1den[:100])

# # %%
# a = jnp.r_[obj.phi_dils, x[2:]]
# b = x[:2]
# a.shape, b.shape
# phis_phase = jnp.r_[
#     jnp.r_[obj.phi_dils, x[:2]][None, :], x[2:][None, :]
# ]  # (2, Nphis_phase = jnp.r_[jnp.r_[obj.phi_dils, x[:2]][None,:], x[2:][None,:]]  # (2, N))
# # phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
# # chi_prod = phis_phase @ self.chi_mat.T  # (2, N)
# # mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2, N)
# # mu_diff = mu[0] - mu[1]  # (N,)


# # %%

# a = np.arange(4)
# b = np.arange(4)
# np.r_[a[None, :], b[None, :]]
