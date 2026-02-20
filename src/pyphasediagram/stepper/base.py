from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp

# Always use float64 with jax
jax.config.update("jax_enable_x64", True)


class BaseStepper(ABC):
    """
    Base class for numerical tracing of a coexistence curve res(phi)=0 (M=N-1).
    Subclasses must implement: residual(phi), phi_init(), is_terminate(phi).
    """

    MAX_STEPS = 1e5

    # --- cycle termination knobs ---
    CYCLE_MIN_STEPS = 10  # don't trigger immediately
    CYCLE_INIT_TOL = 1e-3  # "back at start" tolerance

    def __init__(self):
        # JIT all computational kernels
        self._residual_jit = jax.jit(self.residual)
        self._jac_fn = jax.jit(jax.jacobian(self.residual))
        self._svd_fn = jax.jit(self._svd, static_argnames=["full_matrices"])

        # JIT the whole run loop
        self._run_jit = jax.jit(
            self._run_impl,
            static_argnames=["max_steps"],
        )

    @abstractmethod
    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Return chemical potential and osmotic pressure differences between phases."""

    @abstractmethod
    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Return True if stepping should terminate at given phi."""
        # IMPORTANT: to be fully jittable, subclasses must implement this using JAX ops
        # and return a scalar boolean-like jnp.ndarray (dtype=bool).

    def _svd(self, J, full_matrices=False):
        U, S, Vt = jnp.linalg.svd(J, full_matrices=full_matrices)
        return U, S, Vt

    def _tangent_vec(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Vector in the nullspace of jacobian"""
        J = self._jac_fn(phi)
        _, _, Vt = self._svd_fn(J, full_matrices=True)
        ns = Vt[-1]
        return ns

    def _projection(self, phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Projects composition back onto the coexistence manifold."""
        res = self._residual_jit(phi)

        # Adaptive tolerance
        tol = jnp.clip(jnp.linalg.norm(res) * 1e-3, 1e-12, 1e-6)

        J = self._jac_fn(phi)
        U, S, Vt = self._svd_fn(J, full_matrices=False)

        # Regularization for small singular values to increase stability
        S_inv = 1.0 / S
        mask = (S / jnp.max(S)) < tol
        S_inv = jnp.where(mask, 0.0, S_inv)

        # Pseudoinverse from SVD: J^+ = V * diag(S_inv) * U^T
        J_pinv = (Vt.T * S_inv) @ U.T
        return -J_pinv @ res, res, jnp.min(jnp.abs(S))

    def _hit_initial_cycle(
        self, step: jnp.ndarray, phi: jnp.ndarray, phi_hist: jnp.ndarray
    ):
        phi0 = phi_hist[0]
        dist0 = jnp.linalg.norm(phi - phi0)
        return jnp.logical_and(
            step >= self.CYCLE_MIN_STEPS, dist0 < self.CYCLE_INIT_TOL
        )

    def _decode_status(self, status):
        flags = []
        if status & 1:
            flags.append("MAX_STEPS")
        if status & 2:
            flags.append("PROJECTION_FAIL")
        if status & 4:
            flags.append("NAN")
        if status & 8:
            flags.append("ANGLE_FAIL")
        if status & 16:
            flags.append("CYCLED")
        return flags

    def _projection_loop(
        self, phi: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Newton-style projection loop, capped at 100 iters.
        Returns: (projected_phi, final_residual_norm, final_sv, failed_flag)
        """

        # Compute initial residual norm and sv from a *single* projection eval
        _, res_vec0, sv0 = self._projection(phi)
        res0 = jnp.linalg.norm(res_vec0)

        def proj_cond(state):
            k, phi_k, res_k, sv_k, failed = state
            # Only iterate if the initial residual was nonzero and we're still above tolerance
            return jnp.logical_and(
                res0 > 0.0, jnp.logical_and(res_k > 1e-10, jnp.logical_not(failed))
            )

        def proj_body(state):
            k, phi_k, res_k, sv_k, failed = state

            v_n, res_vec, sv_new = self._projection(phi_k)
            phi_next = phi_k + v_n
            res_next = jnp.linalg.norm(res_vec)

            k_next = k + 1
            failed_next = jnp.logical_or(failed, k_next > 100)
            return k_next, phi_next, res_next, sv_new, failed_next

        k0 = jnp.array(0, dtype=jnp.int32)
        state0 = (k0, phi, res0, sv0, jnp.array(False))
        _, phif, resf, svf, failed = jax.lax.while_loop(proj_cond, proj_body, state0)
        return phif, resf, svf, failed

    def _step_once(
        self,
        step: jnp.ndarray,
        active: jnp.ndarray,
        phi_new: jnp.ndarray,
        sv: jnp.ndarray,
        phi_hist: jnp.ndarray,
        sv_hist: jnp.ndarray,
        status: jnp.ndarray,
        v_init: jnp.ndarray,
        delta_0: float,
        delta_1: float,
        max_steps: int,
        phi_dtype,
    ):
        """
        Perform exactly one step of the outer tracing loop.
        Returns updated (step, active, phi_new, sv, phi_hist, sv_hist, status).
        """

        # Stop if already terminated
        term = self.is_terminate(phi_new)
        active = jnp.logical_and(active, jnp.logical_not(term))

        def do_step(args):
            step, phi_new, sv, phi_hist, sv_hist, status = args

            # Decrease step size near branching point
            delta = jnp.maximum(jnp.minimum(sv, delta_1), delta_0)

            # tangent step
            v_t = self._tangent_vec(phi_new)

            # keep sign of direction consistent with previous step
            is_trace_start = step < 1
            phi_prev = jax.lax.select(is_trace_start, phi_new, phi_hist[step - 1])
            v_prev = jax.lax.select(is_trace_start, v_init, phi_new - phi_prev)

            dot = jnp.dot(v_t, v_prev)
            sgn = jnp.where(dot >= 0, 1.0, -1.0).astype(phi_dtype)
            v_t = v_t * sgn

            # predictor
            phi_pred = phi_new + delta * v_t

            # project back to manifold (capped loop)
            phi_proj, res_norm, sv_new, proj_failed = self._projection_loop(phi_pred)

            # NaN check
            nan_failed = jnp.any(jnp.isnan(phi_proj))

            # Angle deviation check (only if not trace start)
            v_current = phi_proj - phi_new
            denom = jnp.linalg.norm(v_current) * jnp.linalg.norm(v_prev)
            angle = jnp.where(denom > 0, jnp.dot(v_current, v_prev) / denom, 1.0)
            angle_failed = jnp.logical_and(
                jnp.logical_not(is_trace_start), angle * angle < 0.9
            )

            # Update status flags
            status = jax.lax.select(proj_failed, status | jnp.int32(1 << 1), status)
            status = jax.lax.select(nan_failed, status | jnp.int32(1 << 2), status)
            status = jax.lax.select(angle_failed, status | jnp.int32(1 << 3), status)

            # commit new point
            step_next = step + 1
            phi_hist = phi_hist.at[step_next].set(phi_proj)
            sv_hist = sv_hist.at[step_next].set(sv_new)

            return step_next, phi_proj, sv_new, phi_hist, sv_hist, status

        def skip_step(args):
            # if inactive, just keep everything unchanged
            return args

        step2, phi2, sv2, phi_hist2, sv_hist2, status2 = jax.lax.cond(
            active,
            do_step,
            skip_step,
            operand=(step, phi_new, sv, phi_hist, sv_hist, status),
        )

        # Enforce max_steps
        hit_max = step2 >= max_steps
        active2 = jnp.logical_and(active, jnp.logical_not(hit_max))
        status2 = jax.lax.select(hit_max, status2 | jnp.int32(1 << 0), status2)

        # Cycle check
        cycled = self._hit_initial_cycle(step2, phi2, phi_hist2)
        active2 = jnp.logical_and(active2, jnp.logical_not(cycled))
        status2 = jax.lax.select(cycled, status2 | jnp.int32(1 << 4), status2)

        return step2, active2, phi2, sv2, phi_hist2, sv_hist2, status2

    def _loop_cond(self, carry):
        step, active, phi_new, sv, phi_hist, sv_hist, status = carry
        return active

    def _loop_body(self, carry, v_init, delta_0, delta_1, max_steps, phi_dtype):
        step, active, phi_new, sv, phi_hist, sv_hist, status = carry
        return self._step_once(
            step=step,
            active=active,
            phi_new=phi_new,
            sv=sv,
            phi_hist=phi_hist,
            sv_hist=sv_hist,
            status=status,
            v_init=v_init,
            delta_0=delta_0,
            delta_1=delta_1,
            max_steps=max_steps,
            phi_dtype=phi_dtype,
        )

    def _run_impl(
        self,
        phi_init: jnp.ndarray,
        v_init: jnp.ndarray,
        delta_0: float,
        delta_1: float,
        max_steps: int,
    ):
        """Executes the stepping procedure and returns the coexistance curve"""

        n = phi_init.shape[0]
        max_steps = int(max_steps)
        phi_dtype = phi_init.dtype

        # history buffers (fixed size, JIT-friendly)
        phi_hist = jnp.zeros((max_steps + 1, n), dtype=phi_dtype)
        sv_hist = jnp.zeros((max_steps + 1,), dtype=phi_dtype)

        phi_hist = phi_hist.at[0].set(phi_init)

        # initial singular value tracker
        sv0 = self._projection(phi_init)[2]
        sv_hist = sv_hist.at[0].set(sv0)

        # carry: (step, active, phi_new, sv, phi_hist, sv_hist, status_flags)
        # status_flags bits:
        # 0: hit MAX_STEPS
        # 1: projection did not converge within 100 iters
        # 2: NaN encountered
        # 3: large deviation angle check triggered
        # 4: cycled back to initial point
        status0 = jnp.array(0, dtype=jnp.int32)

        step0 = jnp.array(0, dtype=jnp.int32)
        active0 = jnp.array(True)
        carry0 = (step0, active0, phi_init, sv0, phi_hist, sv_hist, status0)

        def body(carry):
            return self._loop_body(
                carry, v_init, delta_0, delta_1, max_steps, phi_dtype
            )

        stepf, _, phif, svf, phi_hist_f, sv_hist_f, statusf = jax.lax.while_loop(
            self._loop_cond, body, carry0
        )
        return phi_hist_f, sv_hist_f, stepf, statusf

    def run(
        self,
        phi_init,
        v_init,
        delta_0=2e-4,
        delta_1=1e-3,
    ):
        """Executes the stepping procedure and returns the coexistance curve"""
        max_steps = int(self.MAX_STEPS)
        phi_jnp = jnp.asarray(phi_init)
        v_jnp = jnp.asarray(v_init, dtype=phi_init.dtype)

        phi_hist, sv_hist, stepf, statusf = self._run_jit(
            phi_jnp,
            v_jnp,
            float(delta_0),
            float(delta_1),
            max_steps=max_steps,
        )

        flags = self._decode_status(statusf)

        # remove empty preallocated tail from history buffers
        k = int(np.asarray(stepf))
        phi_hist = np.asarray(phi_hist)[: k + 1]
        svs = np.asarray(sv_hist)[: k + 1]

        n = phi_hist.shape[1]
        phis = np.transpose(phi_hist.reshape(-1, 2, n // 2), axes=(1, 2, 0))

        return phis, svs, flags
