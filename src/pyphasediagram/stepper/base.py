from abc import ABC, abstractmethod
from scipy.linalg import null_space

import jax
import jax.numpy as jnp
import numpy as np

# Always use float64 with jax
jax.config.update("jax_enable_x64", True)


class BaseStepper(ABC):
    """
    Base class for numerical tracing of a coexistence curve E(phi)=0 (M=N-1).
    Subclasses must implement: residual(phi), phi_init(), is_terminate(phi).
    """

    # Store points in phasespace where binodals cross
    co_arr = np.empty((0, 4))
    bins_list = []

    max_steps = 1e6

    def __init__(self):
        self._residual_jit = jax.jit(self.residual)
        self._jac_fn = jax.jit(jax.jacobian(self.residual))
        self._svd_smallest_fn = jax.jit(self._smallest_singular_values)

    def _smallest_singular_values(self, phi):
        J = jax.jacobian(self.residual)(phi)  # shape (N-1, N)
        _, S, _ = jnp.linalg.svd(J, full_matrices=False)

        return S

    @abstractmethod
    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Return chemical potential and osmotic pressure differences between phases."""

    @abstractmethod
    def phi_init(self) -> jnp.ndarray:
        """Return initial feasible point on coexistence curve, shape (N,)."""

    @abstractmethod
    def is_terminate(self, phi: jnp.ndarray) -> bool:
        """Return True if stepping should terminate at given phi."""

    def tangent_vec(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Vector in the nullspace of jacobian"""
        J = self._jac_fn(phi)  # (3, 4)
        ns = null_space(self._jac_fn(phi))

        return ns.ravel()

    def projection(self, phi: jnp.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Projects composition back onto the coexistence manifold."""

        E = self._residual_jit(phi)

        # Adaptive tolerance
        tol = min(max(1e-12, np.linalg.norm(E) * 1e-3), 1e-6)

        J = self._jac_fn(phi)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)

        # Regularization for small singular values to increase stability
        S_inv = 1 / S
        S_inv[S / S.max() < tol] = 0
        J_pinv = (Vt.T * S_inv) @ U.T
        return -J_pinv @ E, E, abs(S[-1])

    def run(self, phi0=None, dir0=None, delta0=1e-3):
        """Executes the stepping procedure and returns the coexistance curve"""

        # Step in both directions

        for branch_dir in [1, -1]:

            # Need special care when starting from a branch split
            is_start = True
            if phi0 is not None:
                dir0 *= branch_dir
                phi_new = phi0.copy()

            # Start at boundary
            else:
                # Don't go in both directions if starting at boundary
                if branch_dir == -1:
                    break
                phi_new = self.phi_init()

            # Branch dir can change after branching whereas dir0 is fixed at start
            dir_branch = dir0
            res_new = self._residual_jit(phi_new)
            phi_list = [phi_new.copy()]

            delta = delta0
            s_min = self.projection(phi_new)[2]

            s_list = [s_min]
            steps = 0
            while not self.is_terminate(np.array(phi_new)) and (
                phi0 is None or np.linalg.norm(phi_new - phi0) > 1e-3 or is_start
            ):

                if steps > self.max_steps:
                    print("Reached maximum number of steps.")
                    break

                # check for crossover of binodal branches
                if (
                    s_min < 1e-1
                    and np.linalg.norm(phi_new[:2] - phi_new[len(phi_new) // 2 :])
                    > 5e-2
                    and not is_start
                ):
                    # Decrease step size near branching point
                    delta = max(min(s_min, delta0), delta0**2 * 1e-2)
                    # Branching detected

                    if s_min < 1e-4:

                        if len(self.co_arr) == 0 or (
                            # new branching point sufficiently different from previous ones
                            np.linalg.norm(
                                # system invariant under phase swap
                                np.r_[
                                    self.co_arr,
                                    np.roll(self.co_arr, len(phi_new) // 2, axis=1),
                                ]
                                - phi_new,
                                axis=1,
                            ).min()
                            > 1e-2
                        ):

                            self.co_arr = np.r_[self.co_arr, phi_new[None, :]]
                            U, S, Vt = np.linalg.svd(
                                self._jac_fn(phi_new), full_matrices=False
                            )
                            self.run(phi0=phi_new, dir0=Vt[-1], delta0=delta0)
                            is_start = True
                            dir_branch = v_t
                else:
                    delta = delta0

                # tangent step
                v_t = self.tangent_vec(phi_new)  # (N,)

                # keep direction consistent
                if is_start:
                    direction = (
                        dir_branch if dir_branch is not None else -np.ones_like(v_t)
                    )
                else:
                    direction = phi_list[-1] - phi_list[-2]

                direction /= np.linalg.norm(direction)

                a = np.dot(v_t, direction)

                # If the new direction deviates too much from previous the system likely wants to branch
                if a**2 < 0.9:
                    v_t = direction
                else:
                    v_t *= np.sign(a)

                phi_new = phi_new + delta * v_t
                res_new = self._residual_jit(phi_new)

                # project back to manifold
                proj_counter = 0
                while np.linalg.norm(res_new) > 1e-8:
                    v_n, res_new, s_min = self.projection(phi_new)
                    phi_new = phi_new + v_n
                    if proj_counter > 100:
                        raise RuntimeError(
                            f"Projection did not converge after {proj_counter} iterations"
                        )
                    proj_counter += 1

                if np.isnan(phi_new).any():
                    raise RuntimeError(f"Binodal NaN encountered after {steps} steps")

                if s_min > 1e-3:
                    is_start = False

                if steps % 100 == 0 and steps:
                    del_phi_arr = np.linalg.norm(np.diff(phi_list[-100:],axis=0), axis=1)
                    if del_phi_arr.max() < 1e-9:
                        break

                phi_list.append(phi_new.copy())
                s_list.append(s_min)
                Vt, S, Vt = np.linalg.svd(self._jac_fn(phi_new), full_matrices=False)

                steps += 1

            phi_arr = np.transpose(
                np.array(phi_list).reshape(-1, 2, len(phi_new) // 2), axes=(1, 2, 0)
            )
            self.bins_list.append(phi_arr)


# %%
