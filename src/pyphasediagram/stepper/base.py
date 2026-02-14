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

    MAX_STEPS = 1e5

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

    def run(
        self,
        phi_init: np.ndarray,
        v_init: np.ndarray,
        delta_0=1e-5,
        delta_1=1e-3,
    ):
        """Executes the stepping procedure and returns the coexistance curve"""

        phi_new = phi_init.copy()

        res_new = self._residual_jit(phi_new)
        phi_list = [phi_new.copy()]

        delta = delta_1

        # Track second smallest singular value to detect branching points
        sv = self.projection(phi_new)[2]
        sv_list = [sv]
        steps = 0
        is_trace_start = True
        while not self.is_terminate(np.array(phi_new)):
            is_trace_start = len(phi_list) < 2

            if steps > self.MAX_STEPS:
                print("Reached maximum number of steps.")
                break

            # Decrease step size near branching point
            delta = max(min(sv, delta_1), delta_0)

            # tangent step
            v_t = self.tangent_vec(phi_new)  # (N,)

            # keep sign of direction consistent with previous step
            v_prev = v_init if is_trace_start else phi_list[-1] - phi_list[-2]
            v_t *= np.sign(np.dot(v_t, v_prev))

            phi_new = phi_new + delta * v_t
            res_new = self._residual_jit(phi_new)

            # project back to manifold
            proj_counter = 0
            # TODO: move this clause to _project method and add a max_iter argument
            while np.linalg.norm(res_new) > 1e-8:
                v_n, res_new, sv = self.projection(phi_new)
                phi_new = phi_new + v_n
                if proj_counter > 100:
                    raise RuntimeError(
                        f"Projection did not converge after {proj_counter} iterations"
                    )
                #                 proj_counter += 1

            if np.isnan(phi_new).any():
                raise RuntimeError(f"Binodal NaN encountered after {steps} steps")

            # TODO: Once BinodalPoint instances replace phi_list, create `is_colinear` method to check if the direction of new point is colinear with previous direction.
            if not is_trace_start:
                # If the direction of new point deviates too much from previous direction the system likely wants to branch
                v_current = phi_new - phi_list[-1]

                angle = (
                    np.dot(v_t, v_prev) / np.linalg.norm(v_t) / np.linalg.norm(v_prev)
                )
                if angle**2 < 0.9:
                    raise RuntimeError(
                        f"Large deviation between tangent direction and previous step, a={abs(angle):.2f}. Might be cause by branching with singular value = {sv:.2e}"
                    )

            phi_list.append(phi_new.copy())
            sv_list.append(sv)

            steps += 1

        phi_arr = np.transpose(
            np.array(phi_list).reshape(-1, 2, len(phi_new) // 2), axes=(1, 2, 0)
        )
        self.bins_list.append(phi_arr)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    phi_arr = np.transpose(
        np.array(phi_list).reshape(-1, 2, len(phi_new) // 2), axes=(1, 2, 0)
    )
    plt.scatter(phi_arr[0, 0], phi_arr[0, 1])
    plt.scatter(phi_arr[1, 0], phi_arr[1, 1])
    plt.plot([0, 1], [1, 0], "k--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    import matplotlib.pyplot as plt

    plt.scatter(phi_init[::2], phi_init[1::2], color="red")
    plt.scatter(phi_new[::2], phi_new[1::2], color="blue")
    plt.plot([0, 1], [1, 0], "k--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    import matplotlib.pyplot as plt

    colors = ["red", "blue"]
    for i, d in enumerate([v_t, v_init]):
        d /= np.linalg.norm(d)
        plt.plot([0, d[0]], [0, d[1]], color=colors[i], label=f"{i}")
        plt.plot([0, d[2]], [0, d[3]], color=colors[i])

    sv = np.array(sv_list)
    plt.plot(sv)
    plt.yscale("log")
