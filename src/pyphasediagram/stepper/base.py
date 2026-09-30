"""JAX-based predictor-corrector tracing for coexistence curves.

This module defines the abstract numerical machinery used to trace a
one-dimensional solution manifold ``residual(phi) = 0``.  It also defines the
bit flags returned when tracing reaches a numerical or geometric stopping
condition.

Importing the module enables 64-bit floating-point calculations in JAX.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

# TODO: need a better way of checking if we are at the beginning of the run to avoid cycle termination too early.

# Always use float64 with jax
jax.config.update("jax_enable_x64", True)

# Status flag constants for error reporting
STATUS_MAX_STEPS = jnp.int32(1 << 0)  # Hit maximum number of steps
STATUS_PROJ_FAILED = jnp.int32(1 << 1)  # Projection failed to converge
STATUS_NAN = jnp.int32(1 << 2)  # NaN encountered
STATUS_ANGLE_FAILED = jnp.int32(1 << 3)  # Angle deviation check failed
STATUS_CYCLED = jnp.int32(1 << 4)  # Cycled back to initial point


class BaseStepper(ABC):
    """Trace a one-dimensional implicit curve with predictor--corrector steps.

    For a state vector with ``n`` entries, subclasses define an ``n - 1``
    dimensional residual.  The residual Jacobian therefore has a
    one-dimensional null space whose direction is used as the tangent to
    the curve.  Each predicted point is projected back onto the zero-residual
    manifold with a regularized Newton update.

    Subclasses must implement :meth:`residual` and :meth:`is_terminate` using
    JAX-compatible operations.

    Attributes
    ----------
    MAX_STEPS : int
        Maximum number of predictor--corrector steps in one trace.

    CYCLE_MIN_STEPS : int
        Minimum number of steps before returning to the initial point is
        treated as a completed cycle. This prevents premature cycle termination.
    CYCLE_INIT_TOL : float
        Euclidean-distance tolerance used to detect a return to the initial
        point.

    Notes
    -----
    Construction creates JIT-compiled residual, Jacobian, singular-value
    decomposition, and tracing-loop callables.  Implementations of abstract
    methods must consequently avoid Python-side control flow that depends on
    traced values.
    """

    MAX_STEPS: int = int(1e50)

    # terminate once we are back at the initial point within tolerance
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
        """Evaluate the constraints defining the implicit curve.

        Parameters
        ----------
        phi : jax.Array
            State vector with shape ``(n,)``.

        Returns
        -------
        jax.Array
            Residual vector with shape ``(n - 1,)``. The target manifold is
            defined by all entries being zero.

        Notes
        -----
        Implementations must be differentiable by JAX and compatible with
        :func:`jax.jit`.
        """

    @abstractmethod
    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Determine whether tracing should stop at a state.

        Parameters
        ----------
        phi : jax.Array
            State vector with shape ``(n,)``.

        Returns
        -------
        jax.Array
            Scalar Boolean array that is ``True`` when tracing should stop.

        Notes
        -----
        Implementations must use JAX-compatible operations because this method
        is evaluated inside the JIT-compiled tracing loop.
        """
        # IMPORTANT: to be fully jittable, subclasses must implement this using JAX ops
        # and return a scalar boolean-like jnp.ndarray (dtype=bool).

    def _svd(self, J, full_matrices=False):
        """Compute a singular-value decomposition with JAX.

        Parameters
        ----------
        J : jax.Array
            Matrix with shape ``(m, n)``.
        full_matrices : bool, optional
            Whether to compute full-sized left and right singular-vector
            matrices.

        Returns
        -------
        U : jax.Array
            Left singular vectors.
        S : jax.Array
            Singular values with shape ``(min(m, n),)``.
        Vt : jax.Array
            Transposed right singular vectors.
        """
        U, S, Vt = jnp.linalg.svd(J, full_matrices=full_matrices)
        return U, S, Vt

    def _tangent_vec(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Calculate a tangent vector to the residual manifold.

        Parameters
        ----------
        phi : jax.Array
            State vector with shape ``(n,)``.

        Returns
        -------
        jax.Array
            Unit vector with shape ``(n,)`` from the right null space of the
            residual Jacobian.

        Notes
        -----
        The singular-value decomposition does not define the sign of the
        tangent. Directional consistency is imposed later by
        :meth:`_step_once`.
        """
        J = self._jac_fn(phi)
        _, _, Vt = self._svd_fn(J, full_matrices=True)
        ns = Vt[-1]
        return ns

    def _projection(self, phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Calculate one regularized Newton projection update.

        Parameters
        ----------
        phi : jax.Array
            State vector with shape ``(n,)``.

        Returns
        -------
        correction : jax.Array
            Newton correction with shape ``(n,)``. Adding it to ``phi`` moves
            the state toward the zero-residual manifold.
        residual : jax.Array
            Residual evaluated at ``phi``, with shape ``(n - 1,)``.
        min_singular_value : jax.Array
            Scalar smallest absolute singular value of the residual Jacobian.

        Notes
        -----
        Singular values that are small relative to the largest singular value
        are excluded from the pseudoinverse. The relative threshold adapts to
        the residual norm and is clipped to the interval ``[1e-12, 1e-6]``.
        """
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
        """Check whether a trace has returned to its initial state.

        Parameters
        ----------
        step : jax.Array
            Scalar index of the current tracing step.
        phi : jax.Array
            Current state vector with shape ``(n,)``.
        phi_hist : jax.Array
            Preallocated state history with shape ``(max_steps + 1, n)``.

        Returns
        -------
        jax.Array
            Scalar Boolean array. It is ``True`` when at least
            :attr:`CYCLE_MIN_STEPS` have been taken and the state is within
            :attr:`CYCLE_INIT_TOL` of the initial state.
        """
        phi0 = phi_hist[0]
        dist0 = jnp.linalg.norm(phi - phi0)
        return jnp.logical_and(
            step >= self.CYCLE_MIN_STEPS, dist0 < self.CYCLE_INIT_TOL
        )

    def _decode_status(self, status):
        """Convert a status bit mask to human-readable flag names.

        Parameters
        ----------
        status : int or jax.Array
            Scalar integer whose bits encode tracing conditions.

        Returns
        -------
        list of str
            Zero or more of ``"MAX_STEPS"``, ``"PROJECTION_FAIL"``,
            ``"NAN"``, ``"ANGLE_FAIL"``, and ``"CYCLED"``, in that order.
        """
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
        """Iteratively project a state onto the zero-residual manifold.

        Parameters
        ----------
        phi : jax.Array
            Predicted state vector with shape ``(n,)``.

        Returns
        -------
        projected_phi : jax.Array
            Projected state vector with shape ``(n,)``.
        residual_norm : jax.Array
            Scalar residual norm from the final projection iteration.
        min_singular_value : jax.Array
            Scalar smallest absolute Jacobian singular value from the final
            projection iteration.
        failed : jax.Array
            Scalar Boolean array indicating that the iteration limit was
            exceeded before convergence.

        Notes
        -----
        Iteration continues until the residual norm is at most ``1e-10`` or
        the projection has taken more than 100 updates. The loop uses
        :func:`jax.lax.while_loop` so that it remains JIT-compatible.
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
        """Perform one iteration of the outer tracing loop.

        Parameters
        ----------
        step : jax.Array
            Scalar index of the current tracing step.
        active : jax.Array
            Scalar Boolean array indicating whether tracing is active.
        phi_new : jax.Array
            Current state vector with shape ``(n,)``.
        sv : jax.Array
            Scalar smallest Jacobian singular value at the current state.
        phi_hist : jax.Array
            Preallocated state history with shape ``(max_steps + 1, n)``.
        sv_hist : jax.Array
            Preallocated singular-value history with shape
            ``(max_steps + 1,)``.
        status : jax.Array
            Scalar integer status bit mask.
        v_init : jax.Array
            Initial tangent direction with shape ``(n,)``.
        delta_0 : float
            Minimum predictor step size.
        delta_1 : float
            Maximum predictor step size.
        max_steps : int
            Maximum number of predictor--corrector steps.
        phi_dtype : numpy.dtype or jax.numpy.dtype
            Data type used when choosing the tangent orientation.

        Returns
        -------
        step : jax.Array
            Updated scalar step index.
        active : jax.Array
            Updated scalar active flag.
        phi_new : jax.Array
            Updated state vector with shape ``(n,)``.
        sv : jax.Array
            Updated scalar smallest singular value.
        phi_hist : jax.Array
            Updated state-history buffer.
        sv_hist : jax.Array
            Updated singular-value-history buffer.
        status : jax.Array
            Updated scalar status bit mask.

        Notes
        -----
        The predictor step size is the current singular value clipped to
        ``[delta_0, delta_1]``. The method records projection, NaN, angle,
        maximum-step, and cycle conditions in ``status``.
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
            status = jax.lax.select(proj_failed, status | STATUS_PROJ_FAILED, status)
            status = jax.lax.select(nan_failed, status | STATUS_NAN, status)
            status = jax.lax.select(angle_failed, status | STATUS_ANGLE_FAILED, status)

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
        status2 = jax.lax.select(hit_max, status2 | STATUS_MAX_STEPS, status2)

        # Cycle check
        cycled = self._hit_initial_cycle(step2, phi2, phi_hist2)
        active2 = jnp.logical_and(active2, jnp.logical_not(cycled))
        status2 = jax.lax.select(cycled, status2 | STATUS_CYCLED, status2)

        return step2, active2, phi2, sv2, phi_hist2, sv_hist2, status2

    def _loop_cond(self, carry):
        """Return the active flag from a tracing-loop state.

        Parameters
        ----------
        carry : tuple
            Loop state ``(step, active, phi, sv, phi_hist, sv_hist, status)``.

        Returns
        -------
        jax.Array
            Scalar Boolean array controlling the outer JAX loop.
        """
        step, active, phi_new, sv, phi_hist, sv_hist, status = carry
        return active

    def _loop_body(self, carry, v_init, delta_0, delta_1, max_steps, phi_dtype):
        """Advance the tracing-loop state by one iteration.

        Parameters
        ----------
        carry : tuple
            Loop state ``(step, active, phi, sv, phi_hist, sv_hist, status)``.
        v_init : jax.Array
            Initial tangent direction with shape ``(n,)``.
        delta_0 : float
            Minimum predictor step size.
        delta_1 : float
            Maximum predictor step size.
        max_steps : int
            Maximum number of predictor--corrector steps.
        phi_dtype : numpy.dtype or jax.numpy.dtype
            State-vector data type.

        Returns
        -------
        tuple
            Updated loop state in the same order as ``carry``.
        """
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
        """Execute the JIT-compatible core tracing loop.

        Parameters
        ----------
        phi_init : jax.Array
            Initial state vector with shape ``(n,)``.
        v_init : jax.Array
            Initial tangent direction with shape ``(n,)``.
        delta_0 : float
            Minimum predictor step size.
        delta_1 : float
            Maximum predictor step size.
        max_steps : int
            Maximum number of predictor--corrector steps. This argument is
            static when the method is JIT-compiled.

        Returns
        -------
        phi_hist : jax.Array
            Preallocated state-history buffer with shape
            ``(max_steps + 1, n)``. Entries after the final step remain zero.
        sv_hist : jax.Array
            Preallocated singular-value-history buffer with shape
            ``(max_steps + 1,)``.
        final_step : jax.Array
            Scalar index of the final populated history entry.
        status : jax.Array
            Scalar integer status bit mask.
        """

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
        """Trace a coexistence curve from an initial state and direction.

        Parameters
        ----------
        phi_init : numpy.ndarray or jax.Array
            Initial state vector with shape ``(n,)``. Entries are ordered by
            phase, with the independent component coordinates of phase A
            followed by those of phase B. Consequently, ``n`` must be even.
        v_init : numpy.ndarray or jax.Array
            Initial tangent direction with shape ``(n,)`` and the same
            coordinate ordering and data type as ``phi_init``.
        delta_0 : float, optional
            Minimum predictor step size.
        delta_1 : float, optional
            Maximum predictor step size.

        Returns
        -------
        phis : numpy.ndarray
            Traced compositions with shape ``(2, n // 2, k)``. The axes
            correspond to phase, independent component, and trace point.
        svs : numpy.ndarray
            Smallest Jacobian singular value at each trace point, with shape
            ``(k,)``.
        flags : list of str
            Status names encountered during tracing. Possible entries are
            ``"MAX_STEPS"``, ``"PROJECTION_FAIL"``, ``"NAN"``,
            ``"ANGLE_FAIL"``, and ``"CYCLED"``.

        Notes
        -----
        ``MAX_STEPS`` determines the fixed buffer size compiled by JAX. The
        unused part of each buffer is removed before the NumPy results are
        returned.
        """
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
