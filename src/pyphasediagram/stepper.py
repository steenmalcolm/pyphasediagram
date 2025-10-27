import numpy as np
from scipy.linalg import null_space, cho_factor, cho_solve
from scipy.optimize import root


class BinodalStepper:
    """
    Numerically traces the binodal curve for a ternary mixture.

    The class integrates along the coexistence curve in composition space by starting
    from the binary limit and following the tangent direction given by the null space of
    the equilibrium conditions and projecting back onto the binodal after each step.
    It is based on the Flory–Huggins free energy model.

    Parameters
    ----------
    chi_matrix : np.ndarray
        2×2 reduced matrix of Flory–Huggins interaction parameters χ_ij.

    """

    # Max number of steps for run
    max_steps = 40000
    eps = 1e-12

    def __init__(self, chi_matrix: np.ndarray):
        self.chi_mat = chi_matrix
        # Step size for moving along the binodal

    @staticmethod
    def phase_distance(phi):
        return np.linalg.norm(phi[:2] - phi[2:])

    def free_en(self, phi: np.ndarray) -> np.ndarray:
        """Return the Flory–Huggins free energy density for given compositions."""
        x, y = phi[[0, 2]], phi[[1, 3]]
        z = 1 - x - y
        f = x * np.log(x) + y * np.log(y) + z * np.log(z)
        f += 0.5 * (
            self.chi_mat[0, 0] * x**2
            + 2 * self.chi_mat[0, 1] * x * y
            + self.chi_mat[1, 1] * y**2
        )
        return f

    def residual(self, phi: np.ndarray) -> np.ndarray:
        """Return chemical potential and osmotic pressure differences between phases."""
        x, y = phi[[0, 2]], phi[[1, 3]]
        z = 1 - x - y
        logx, logy, logz = np.log(x), np.log(y), np.log(z)

        dx = x[0] - x[1]
        dy = y[0] - y[1]

        mu = (
            np.array([logx[0] - logx[1], logy[0] - logy[1]])
            - logz[0]
            + logz[1]
            + self.chi_mat @ np.array([dx, dy])
        )
        pi = (
            -logz[0]
            + logz[1]
            + 0.5
            * (
                self.chi_mat[0, 0] * (x[0] ** 2 - x[1] ** 2)
                + 2 * self.chi_mat[0, 1] * (x[0] * y[0] - x[1] * y[1])
                + self.chi_mat[1, 1] * (y[0] ** 2 - y[1] ** 2)
            )
        )
        return np.hstack((mu, pi))

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        """Jacobian of the residual vector w.r.t. the four phase compositions"""
        x, y = phi[[0, 2]], phi[[1, 3]]
        z = 1 - x - y
        c = self.chi_mat

        inv_x, inv_y, inv_z = 1 / x, 1 / y, 1 / z
        # sign to compute difference between phases
        s = np.array([1.0, -1.0])

        # Three conditions and four variables
        J = np.zeros((3, 4))
        # Row 0: d(mu0)/d(...)
        J[0, ::2] = s * (inv_x + inv_z + c[0, 0])  # dmu0/dx0, dmu0/dx1
        J[0, 1::2] = s * (inv_z + c[0, 1])  # dmu0/dy0, dmu0/dy1

        # Row 1: d(mu1)/d(...)
        J[1, ::2] = s * (inv_z + c[1, 0])  # dmu1/dx0, dmu1/dx1
        J[1, 1::2] = s * (inv_y + inv_z + c[1, 1])  # dmu1/dy0, dmu1/dy1

        # Row 2: d(pi)/d(...)
        J[2, ::2] = s * (inv_z + c[0, 0] * x + c[0, 1] * y)  # dpi/dx0, dpi/dx1
        J[2, 1::2] = s * (inv_z + c[1, 0] * x + c[1, 1] * y)  # dpi/dy0, dpi/dy1
        return J

    def tangent_vec(self, phi: np.ndarray) -> np.ndarray:
        """Computes the tangent vector (null-space direction) at a coexistence point."""
        ns = null_space(self.jacobian(phi))
        if ns.shape != (4, 1):
            raise RuntimeError("Null space not found")
        return ns.ravel()

    def solve_least_squares(self, J: np.ndarray, E: np.ndarray) -> np.ndarray:
        try:
            L, lower = cho_factor(J @ J.T)
            return J.T @ cho_solve((L, lower), E)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "Singular Jacobian encountered in projection step; perhaps near critical point"
            )

    def projection(self, phi: np.ndarray) -> np.ndarray:
        """Projects a trial composition back onto the coexistence manifold."""
        return self.solve_least_squares(self.jacobian(phi), -self.residual(phi))

    def binary_state(self):

        chi_12 = self.chi_mat[0, 1] - 1 / 2 * (self.chi_mat[0, 0] + self.chi_mat[1, 1])
        if chi_12 < 2:
            raise RuntimeError("Flory parameter too low for phase separation")
        eq_func = lambda x: np.log(x / (1 - x)) + chi_12 * (1 - 2 * x)
        # Initial guess small to find dilute phase
        sol = root(eq_func, 1e-4)
        if not sol.success:
            raise ValueError(f"Root finding failed: {sol.message}")
        return sol.x.item()

    def is_valid_state(self, phi: np.ndarray) -> bool:
        return not (
            np.any(phi < self.eps)
            or (phi[0] + phi[1]) > (1 - self.eps)
            or (phi[2] + phi[3]) > (1 - self.eps)
        )

    def init_point(self) -> np.ndarray:
        """Finds an initial coexistence point near the binary limit."""

        # Start from a nearly binary case
        x0 = self.binary_state()
        phi = np.array([x0, 1 - x0, 1 - x0, x0])
        phi -= 1e-4
        res = self.residual(phi)
        max_iter = 50
        iteration = 0
        phi_list = [phi.copy()]
        res_list = [np.linalg.norm(res)]

        while np.linalg.norm(res) > 1e-10:
            v = self.projection(phi)
            if self.is_valid_state(phi + v):
                phi += v
            else:
                while not self.is_valid_state(phi - v):
                    v *= 0.5
                phi -= v
            res = self.residual(phi)
            phi_list.append(phi.copy())
            res_list.append(np.linalg.norm(res))
            if not self.is_valid_state(phi):
                raise RuntimeError("Initial state outside of physical range")
            if iteration > max_iter:
                raise RuntimeError("In init_point: Projection not converging")

            iteration += 1

        if np.sqrt((phi[2] - phi[0]) ** 2 + (phi[3] - phi[1]) ** 2) < 1e-3:
            raise RuntimeError("Homogeneous state as initial state")

        return phi

    def run(self, delta=1e-3) -> None:
        """Executes the stepping procedure and returns the binodal compositions."""

        phi_new = self.init_point()
        diff_new = self.phase_distance(phi_new)
        res_new = self.residual(phi_new)

        phi_list = [phi_new.copy()]
        res_list = [res_new.copy()]
        diff_list = [diff_new.copy()]

        steps = 0
        while diff_new > 1e-2 and self.is_valid_state(phi_new):
            if steps > self.max_steps:
                break

            # v_t: tangent direction along binodal
            v_t = self.tangent_vec(phi_new)

            # Keep direction of nullspace vector consistent
            if steps < 2:
                v_t *= np.sign(np.dot(v_t, -np.ones_like(v_t)))
            else:
                v_t *= np.sign(np.dot(v_t, phi_list[-1] - phi_list[-2]))

            phi_new += v_t * delta
            res_new = self.residual(phi_new)

            # Project back to coexistence line
            proj_counter = 0
            while np.linalg.norm(res_new) > 1e-10:
                if proj_counter > 100:
                    raise RuntimeError(
                        f"Projection not converging after {steps} steps; residual {res_new[0]:.2e}, {res_new[1]:.2e}, {res_new[2]:.2e}"
                    )
                # v_n: normal correction back to coexistence curve
                try:
                    v_n = self.projection(phi_new)
                except RuntimeError as e:
                    if self.phase_distance(phi_new) < 3e-2:
                        break
                    print(
                        f"Debug: error for distance {self.phase_distance(phi_new):.2e} at step {steps}"
                    )
                    raise e
                phi_new += v_n
                res_new = self.residual(phi_new)
                proj_counter += 1

            diff_new = self.phase_distance(phi_new)
            if np.any(np.isnan(phi_new)):
                raise RuntimeError(f"Binodal NaN encountered after {steps} steps")

            phi_list.append(phi_new.copy())
            res_list.append(res_new.copy())
            diff_list.append(diff_new.copy())

            steps += 1

        # shape (N_phase, N_comp, N_steps) = (2, 2, N_steps)
        phi_arr = np.transpose(np.array(phi_list).reshape(-1, 2, 2), axes=(1, 2, 0))

        return phi_arr, np.array(res_list), np.array(diff_list)
