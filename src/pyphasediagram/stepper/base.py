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

    max_steps = 1e6

    def __init__(self):
        self._residual_jit = jax.jit(self.residual)
        self._jac_fn = jax.jit(jax.jacobian(self.residual))

    @abstractmethod
    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Return chemical potential and osmotic pressure differences between phases."""
        pass

    @abstractmethod
    def phi_init(self) -> jnp.ndarray:
        """Return initial feasible point on coexistence curve, shape (N,)."""
        pass

    @abstractmethod
    def is_terminate(self, phi: jnp.ndarray) -> bool:
        """Return True to stop tracing."""
        pass

    def tangent_vec(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Vector in the nullspace of jacobian"""
        J = self._jac_fn(phi)  # (3, 4)
        ns = null_space(self._jac_fn(phi))
        return ns.ravel()

    def projection(self, phi: jnp.ndarray) -> np.ndarray:
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
        return -J_pinv @ E, E

    def run(self, delta=1e-3):
        """Executes the stepping procedure and returns the coexistance curve"""

        phi_new = self.phi_init()

        res_new = self._residual_jit(phi_new)
        phi_list = [phi_new.copy()]

        steps = 0
        while not self.is_terminate(np.array(phi_new)):  # Cast to numpy for speed

            if steps > self.max_steps:
                break

            # tangent step
            v_t = self.tangent_vec(phi_new)  # (N,)

            # keep direction consistent
            if steps < 2:
                v_t *= np.sign(np.dot(v_t, -np.ones_like(v_t)))
            else:
                v_t *= np.sign(np.dot(v_t, phi_list[-1] - phi_list[-2]))

            phi_new = phi_new + delta * v_t
            res_new = self._residual_jit(phi_new)

            # project back to manifold
            proj_counter = 0
            while np.linalg.norm(res_new) > 1e-8:
                v_n, res_new = self.projection(phi_new)
                phi_new = phi_new + v_n
                if proj_counter > 100:
                    raise RuntimeError(
                        f"Projection did not converge after {proj_counter} iterations"
                    )
                proj_counter += 1

            if np.isnan(phi_new).any():
                raise RuntimeError(f"Binodal NaN encountered after {steps} steps")

            phi_list.append(phi_new.copy())

            steps += 1

        phi_arr = np.transpose(
            np.array(phi_list).reshape(-1, 2, len(phi_new) // 2), axes=(1, 2, 0)
        )
        return phi_arr
