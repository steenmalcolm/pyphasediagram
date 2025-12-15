# %%
import time
import jax.numpy as jnp
import numpy as np
from pyphasediagram.stepper import BaseStepper
import flory


class SingleCompositionVaraition(BaseStepper):
    """Coexistence curve stepper for a ternary mixture."""

    def __init__(self, chi_comp: np.ndarray, phi_means: np.ndarray, N: int = 0):

        self.chi_comp = chi_comp  # (N+1,N+1)
        self.phi_means = phi_means  # (N,)
        self.N = N
        if chi_comp == None:
            assert N > 0, "Must provide N>0 to generate random chi matrix"
        else:
            self.chi_mat = self.chi_reduced(chi_comp)
            assert (
                self.chi_mat.shape[0] == phi_means.shape[0]
            ), f"Need to set {self.chi_mat.shape[0]} mean compositions not {phi_means.shape[0]}"
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
        phis, lmd = x[:-1], x[-1]
        phis_phase = phis.reshape(2, -1)  # (2, N)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ self.chi_mat.T  # (2, N)
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2, N)
        mu_diff = mu[0] - mu[1]  # (N,)

        pi = -jnp.log(phi0_phase) + 0.5 * (phis_phase * chi_prod).sum(axis=1)  # (2,)
        pi_diff = pi[0] - pi[1]  # ()

        coexistance_cond = jnp.r_[mu_diff, pi_diff]  # (N+1,)

        # Add constraints for mass conservation except for first component
        mass_cond = (
            lmd * phis_phase[0, 1:] + (1 - lmd) * phis_phase[1, 1:] - self.phi_means[1:]
        )
        return jnp.r_[coexistance_cond, mass_cond]

    def is_terminate(self, x: np.ndarray) -> bool:
        """Stop when phases become too similar or invalid composition."""

        phis_phase, lmd = x[:-1].reshape(2, -1), x[-1]
        close = np.linalg.norm(phis_phase[1] - phis_phase[0]) < 1e-3
        invalid = (
            (phis_phase < 0).any()
            or (phis_phase.sum(axis=1) > 1).any()
            or (lmd < 0)
            or (lmd > 1)
        )
        return close or invalid

    def phi_init(self) -> jnp.ndarray:
        """Find an initial coexistence point near the binary limit."""

        if self.chi_comp == None:
            fracs = []
            while len(fracs) != 2:
                # Compressible case N->N+1
                chi_comp = np.random.random((self.N + 1, self.N + 1)) + 3
                chi_comp = (chi_comp + chi_comp.T) / 2
                chi_comp -= np.diag(np.diag(self.chi_mat))
                phi_means = np.ones(self.N + 1) / (self.N + 1)
                phases = flory.find_coexisting_phases(
                    self.N + 1, chi_comp, phi_means, progress=False
                )
                fracs = phases.fractions

            self.phi_means = phi_means
            self.chi_mat = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    self.chi_mat[i, j] = (
                        chi_comp[i + 1, j + 1] - chi_comp[i + 1, 0] - chi_comp[j + 1, 0]
                    )

        else:
            phases = flory.find_coexisting_phases(
                self.chi_comp.shape[0], self.chi_comp, self.phi_means, progress=False
            )
            fracs = phases.fractions
            assert (
                len(fracs) == 2
            ), "Need to find two coexisting phases for initialization"

        x = fracs[:, 1:].ravel()
        x = np.r_[x, phases.volumes[0]]

        res = self._residual_jit(x)

        i = 0
        while np.linalg.norm(res) > 1e-8:
            v_n, res = self.projection(x)
            x += v_n
            print(f"{i}: {np.linalg.norm(res):.2e}")
            if i > 100:
                print("muddafukka you suck")
                break

            i += 1

        return x


# %%
# import matplotlib.pyplot as plt
# import numpy as np
# import flory

# # Compressible system
# chi_comp = np.array(
#     [
#         [0.0, 3.4764248, 3.42753118, 3.18283335, 3.8735968],
#         [3.4764248, 0.0, 3.24135875, 3.38022499, 3.66563028],
#         [3.42753118, 3.24135875, 0.0, 3.43717238, 3.43450685],
#         [3.18283335, 3.38022499, 3.43717238, 0.0, 3.72282364],
#         [3.8735968, 3.66563028, 3.43450685, 3.72282364, 0.0],
#     ]
# )

# # Sum of mean volume fractions = 1

# # Number of independent components in incompressible system
# N = 4
# fracs = []
# while len(fracs) != 2:
#     print("Trying to find phases...")

#     # Flory calculation for compressible system
#     chi_comp = np.random.random((N + 1, N + 1)) + 3
#     chi_comp = (chi_comp + chi_comp.T) / 2
#     chi_comp -= np.diag(np.diag(chi_comp))
#     phi_means = np.ones(N + 1) / (N + 1)
#     phases = flory.find_coexisting_phases(N + 1, chi_comp, phi_means, progress=False)
#     fracs = phases.fractions
# assert len(phases.fractions) == 2
# # %%
# phis = phases.fractions[:, 1:].ravel()
# lmd = phases.volumes[0]
# x = np.r_[phis, lmd]

# # %%
# obj = SingleCompositionVaraition(chi_comp, phi_means[1:])
# pos_T = obj.run()
# # %%
# pos = pos_T.T
# # %%
# plt.figure(figsize=(8, 8))
# for i in range(N):
#     plt.subplot(2, 2, i + 1)
#     phidil = pos[i]
#     phiden = pos[N + i]
#     lmd = pos[-1]
#     phi_mean = phidil * lmd + phiden * (1 - lmd)
#     # plt.plot(lmd)
#     plt.plot(phidil, phiden)
#     plt.scatter(phidil[0], phiden[0], color="red", label="start")
#     if i == 0:
#         plt.legend()

#     # plt.title(f"phi_{i+1} mean {phi_mean.mean():.3f}+-{phi_mean.std():.3f}")
#     plt.text(
#         0.5,
#         0.5,
#         r"$\bar{\phi}_%d = %.3f \pm %.3f$" % (i + 1, phi_mean.mean(), phi_mean.std()),
#         transform=plt.gca().transAxes,
#         fontsize=12,
#         verticalalignment="center",
#         horizontalalignment="center",
#     )
#     plt.xlabel(r"$\phi_{%d}^{(1)}$" % (i + 1), fontsize=12)
#     plt.ylabel(r"$\phi_{%d}^{(2)}$" % (i + 1), fontsize=12)
#     # plt.plot(phidil*lmd + phiden*(1-lmd))

# # %%

# # %%
# res = [obj._residual_jit(p) for p in pos.T]
# res = res[:-1]
# res = np.array(res).T
# for r in res:
#     plt.plot(r)

# # %%
# plt.plot(lmd)
# # %%
# phi1dil = pos[1]
# phi1den = pos[N + 1]
# # plt.plot(phi1dil)
# plt.plot(phi1den[:100])

# # %%
# plt.plot(pos[0] * lmd + pos[4] * (1 - lmd))
