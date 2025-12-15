# %%
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from pyphasediagram.stepper.base import BaseStepper


class ChiFitter(BaseStepper):
    """Coexistence curve stepper for a ternary mixture."""

    def __init__(self, phis_phase: np.ndarray = None):
        # Convert to jnp array
        self.phis_phase = jnp.array(phis_phase)
        super().__init__()

    def residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Return chemical potential and osmotic pressure differences between phases.
        """
        phis, chis = x[:4], x[4:]
        chi_mat = jnp.array([[chis[0], chis[1]], [chis[1], chis[2]]])
        phis_phase = phis.reshape(2, -1)  # (2, 2)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ chi_mat  # (2, 2)
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


# %%
# from pyphasediagram.stepper import TernaryStepper
# import matplotlib.pyplot as plt
# from scipy.linalg import null_space

# lmd = 10
# lmd_max = 0.2

# del_chi = -np.array([0.57020835, 0.57729161, 0.58446286])
# del_chi1 = np.r_[del_chi[2], 0, -del_chi[0]]
# del_chi1 *= np.linalg.norm(del_chi) / np.linalg.norm(del_chi1)
# del_chi2 = np.cross(del_chi, del_chi1)
# del_chi2 *= np.linalg.norm(del_chi) / np.linalg.norm(del_chi2)
# chis = np.array([-3.5, -0.8, -3.0])
# chi_matrix = np.array([[chis[0], chis[1]], [chis[1], chis[2]]])
# del_chi_matrix = np.array([[del_chi[0], del_chi[1]], [del_chi[1], del_chi[2]]])
# t_obj = TernaryStepper(chi_matrix)
# binodal = t_obj.run()
# phi_r_dil, phi_d_dil, phi_r_den, phi_d_den = binodal.T
# ev, vec = np.linalg.eig(del_chi_matrix)
# tie_lines = np.c_[phi_r_den - phi_r_dil, phi_d_den - phi_d_dil]
# # %%

# chis = np.array([-3, -0.8, -3.0])
# chi_matrix = np.array([[chis[0], chis[1]], [chis[1], chis[2]]])
# t_obj = TernaryStepper(chi_matrix)
# binodal = t_obj.run()
# # %%
# alignment = []
# obj = ChiFitter()
# no_null = []
# for idx in np.arange(0, len(binodal)):
#     # print(f"Step {idx+1}/{len(binodal)}", end=", ")
#     b = binodal[idx]
#     x = jnp.r_[b, chis]
#     res = obj._residual_jit(x)
#     J = obj._jac_fn(x)
#     V = null_space(J)
#     S = V[:4]
#     if null_space(S).size > 0:
#         del_x = V @ null_space(S).ravel()
#         c = del_x[4:]
#         # print("Delta chi:", c, end=", ")
#         del_chi_matrix = np.array([[c[0], c[1]], [c[1], c[2]]])
#         alignment.append(del_chi_matrix @ np.r_[b[0] - b[2], b[1] - b[3]])
#         # print("alignment:", alignment[-1])

#         assert (
#             np.linalg.norm(obj._residual_jit(x + del_x)) < 1e-8
#         ), "Failed to find nullspace step"
#         assert np.linalg.norm(del_x[:4]) < 1e-10, "Change in compositions"
#     else:
#         eigvals, eigvecs = np.linalg.eig(S)
#         if (np.abs(eigvals) < 1e-8).any():
#             del_x = V @ eigvecs[:, np.argmin(np.abs(eigvals))]
#             del_chi_matrix = np.array([[del_x[4], del_x[5]], [del_x[5], del_x[6]]])
#             alignment.append(del_chi_matrix @ np.r_[b[0] - b[2], b[1] - b[3]])
#         else:
#             no_null.append(idx)
#             # print("No null space found", end=", ")
# # %%
# no_null = np.array(no_null)
# phi_r_dil, phi_d_dil, phi_r_den, phi_d_den = binodal.T
# plt.plot(phi_r_dil, phi_d_dil, label="dilute", color="blue")
# plt.plot(phi_r_den, phi_d_den, label="dense", color="blue")
# plt.plot(phi_r_dil[no_null], phi_d_dil[no_null], "x", label="no null", color="red")
# plt.plot(phi_r_den[no_null], phi_d_den[no_null], "x", label="no null", color="red")

# # %%
# # U, sig, Vt = np.linalg.svd(J, full_matrices=False)
# # S_inv = Vt.T @ np.diag([1 / s if s > 1e-8 else 0 for s in sig]) @ U.T
# # del_x = -S_inv @ obj._residual_jit(x)
# print(
#     f"before: {np.linalg.norm(obj._residual_jit(x)):.2e}\n after: {np.linalg.norm(obj._residual_jit(x + del_x)):.2e}"
# )
# # %%

# x_list.append(x + del_x)
# # %%
# phi_p = (S @ a) * np.linalg.norm(del_phi) / np.linalg.norm(S @ a)
# plt.plot([0, phi_p[0]], [0, phi_p[1]], label="tangent", c="red")
# plt.plot([0, phi_p[2]], [0, phi_p[3]], label="tangent", c="blue")
# plt.plot([0, del_phi[0]], [0, del_phi[1]], label="tangent", c="red")
# plt.plot([0, del_phi[2]], [0, del_phi[3]], label="tangent", c="blue")
# plt.axis("equal")


# # %%
# pos = np.array(x_list).T[:4]
# pos.shape

# # %%
# plt.plot(pos[0], pos[1], label=r"$\phi^{(1)}$")
# plt.plot(pos[2], pos[3], label=r"$\phi^{(2)}$")
# plt.scatter(phi_final[0], phi_final[1], color="red", label="target")
# plt.scatter(phi_final[2], phi_final[3], color="red")
# # %%
# N = 3
# v1 = np.ones(N) / np.sqrt(N)
# v2 = np.random.random(N)
# v2 -= v1 @ v2 * v1
# v2 /= np.linalg.norm(v2)
# v3 = np.cross(v1, v2)
# V = np.c_[v1, v2, v3]

# lmds = -np.random.randint(1, 10, N)
# chi = V @ np.diag(lmds) @ V.T
# chi


# # %%
# def expand_chi(chi):
#     n = len(chi)
#     chi_e = -0.5 * np.diag(chi)
#     for i in range(0, n):
#         for j in range(i + 1, n):
#             chi_e = np.append(chi_e, chi[i, j] - 0.5 * (chi[i, i] + chi[j, j]))
#     return chi_e


# chis = expand_chi(chi)

# # %%
# chi_matrix = np.array(
#     [
#         [0, chis[0], chis[1], chis[2]],
#         [chis[0], 0, chis[3], chis[4]],
#         [chis[1], chis[3], 0, chis[5]],
#         [chis[2], chis[4], chis[5], 0],
#     ]
# )
# chi_matrix

# # %%
# import flory

# phi_means = np.ones(4) * 0.25
# phases = flory.find_coexisting_phases(4, chi_matrix, phi_means)

# # %%
