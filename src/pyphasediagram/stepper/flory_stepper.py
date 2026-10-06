"""Flory--Huggins coexistence-curve tracing for ternary mixtures.

The module specializes :class:`~pyphasediagram.stepper.base.BaseStepper` with
the chemical-potential and osmotic-pressure constraints of an incompressible
ternary mixture.  It also provides initial states near each binary edge of the
composition triangle.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root

from pyphasediagram.stepper.base import BaseStepper


class Stepper(BaseStepper):
    r"""Trace two-phase coexistence curves for a ternary mixture.

    Parameters
    ----------
    chis : array-like
        Reduced interaction matrix with shape ``(2, 2)`` for the two
        independent composition coordinates.

    Attributes
    ----------
    chis : jax.Array
        Reduced interaction matrix with shape ``(2, 2)``.

    Notes
    -----
    A state is ordered as
    :math:`\boldsymbol{\phi} = (\phi_1^{(a)}, \phi_2^{(a)},
    \phi_1^{(b)}, \phi_2^{(b)})`, where :math:`a` and :math:`b` label the
    two phases. The solvent composition in phase :math:`\alpha` is given by the incompressibility condition

    .. math::

       \phi_0^{(\alpha)} = 1 - \phi_1^{(\alpha)} - \phi_2^{(\alpha)}.
    """

    def __init__(self, chis: ArrayLike) -> None:
        self.chis = jnp.asarray(chis)  # (2,2)
        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        r"""Evaluate the two-phase coexistence constraints.

        Parameters
        ----------
        phi : jax.Array
            Phase compositions with shape ``(4,)`` and ordering
            ``[phi1_a, phi2_a, phi1_b, phi2_b]``.

        Returns
        -------
        jax.Array
            Residual with shape ``(3,)`` containing the phase-A minus phase-B
            differences in the two independent chemical potentials followed
            by the osmotic-pressure difference.

        Notes
        -----
        For each phase :math:`\alpha \in \{a, b\}`, the independent chemical
        potentials are

        .. math::

           \mu_i^{(\alpha)} =
           \ln \phi_i^{(\alpha)} - \ln \phi_0^{(\alpha)}
           + \sum_{j=1}^{2} \chi_{ij}\phi_j^{(\alpha)},
           \qquad i \in \{1, 2\},

        and the osmotic pressure is

        .. math::

           \Pi^{(\alpha)} = -\ln \phi_0^{(\alpha)}
           + \frac{1}{2}\sum_{i,j=1}^{2}
           \chi_{ij}\phi_i^{(\alpha)}\phi_j^{(\alpha)}.

        Here,
        :math:`\phi_0^{(\alpha)} = 1 - \phi_1^{(\alpha)} -
        \phi_2^{(\alpha)}`. and :math:`\chi_{ij}` are the reduced interaction parameters.

        The logarithms require positive independent compositions and positive
        :math:`\phi_0^{(\alpha)}`. Invalid states are handled by
        :meth:`is_terminate` during curve tracing rather than validated here.
        """
        phis_phase = phi.reshape(2, -1)  # (2, 2)
        phi0_phase = 1.0 - phis_phase.sum(axis=1)  # (2,)
        chi_prod = phis_phase @ self.chis.T  # (2, 2)
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod  # (2,2)
        mu_diff = mu[0] - mu[1]  # (2,)

        pi = -jnp.log(phi0_phase) + 0.5 * (phis_phase * chi_prod).sum(axis=1)  # (2,)
        pi_diff = pi[0] - pi[1]  # ()

        return jnp.hstack((mu_diff, pi_diff))  # (3,)

    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Determine whether curve tracing should stop.

        Parameters
        ----------
        phi : jax.Array
            Phase compositions with shape ``(4,)`` and ordering
            ``[phi1_a, phi2_a, phi1_b, phi2_b]``.

        Returns
        -------
        jax.Array
            Scalar Boolean array. It is ``True`` if the phase compositions
            differ by less than ``1e-3``, if an independent composition is
            negative, or if the independent compositions in either phase sum
            to more than one.

        Notes
        -----
        This method uses only JAX operations because it is called from the
        JIT-compiled tracing loop.
        """
        # IMPORTANT: This must be JAX-traceable for the fully-jitted run loop.

        close = jnp.linalg.norm(phi[2:] - phi[:2]) < 1e-3

        invalid = jnp.logical_or(jnp.any(phi < 0.0), (phi[0] + phi[1]) > 1.0)
        invalid = jnp.logical_or(invalid, (phi[2] + phi[3]) > 1.0)

        return jnp.logical_or(close, invalid)

    def binary_state(self, chi: float) -> Optional[float]:
        r"""Calculate the dilute fraction of a symmetric binary coexistence.

        The dense volume fraction follows from the incompressibility
        condition.

        Parameters
        ----------
        chi : float
            Binary Flory--Huggins interaction parameter.

        Returns
        -------
        float or None
            Dilute-phase composition :math:`x \in (0, 1/2)`. ``None`` is
            returned when :math:`\chi \leq 2`, for which this model has no
            demixed binary state.

        Raises
        ------
        ValueError
            If the SciPy root solver does not converge.

        Notes
        -----
        The method solves the implicit equation

        .. math::

           \ln\!\left(\frac{x}{1-x}\right) + \chi(1-2x) = 0

        from an initial guess of ``1e-4``. It runs only during initialization
        and is not JIT-compiled.
        """
        # This is initialization (SciPy); it does not need to be jitted to run the main stepper loop.
        if chi <= 2:
            print(f"Warning: Chi value {chi:.2f}<=2 too low for phase separation")
            return None

        eq_func = lambda x: np.log(x / (1 - x)) + chi * (1 - 2 * x)
        sol = root(eq_func, 1e-4)
        if not sol.success:
            raise ValueError(f"Root finding failed: {sol.message}")
        return float(sol.x.item())

    def binary_init(
        self, which_comp: int = 0
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Construct an initial state near a binary limit.

        Parameters
        ----------
        which_comp : {0, 1, 2}, optional
            Component selected to be dilute. The corresponding demixing
            pair and effective binary interaction are listed below, where
            :math:`\chi_{ij}` denotes the reduced interaction matrix ``chis``.

            * ``0``: components 1 and 2, with
              :math:`\chi = \chi_{01}
              - (\chi_{00} + \chi_{11})/2`;
            * ``1``: components 0 and 2, with
              :math:`\chi = -\chi_{11}/2`;
            * ``2``: components 0 and 1, with
              :math:`\chi = -\chi_{00}/2`.


        Returns
        -------
        phi_init : numpy.ndarray or None
            Initial phase compositions with shape ``(4,)`` and ordering
            ``[phi1_a, phi2_a, phi1_b, phi2_b]``. ``None`` indicates that an
            initial state could not be constructed.
        v_init : numpy.ndarray or None
            Initial tracing direction with shape ``(4,)`` and the same
            coordinate ordering. ``None`` is returned together with a missing
            initial state.

        Notes
        -----
        ``(None, None)`` is returned when the selected effective interaction
        is at most 2, when ``which_comp`` is outside ``{0, 1, 2}``, or when
        projection has not converged after :math:`10\,000` iterations.
        Otherwise, the initial composition is projected until
        :math:`\lVert\boldsymbol{r}\rVert_2 \leq 10^{-8}`. Candidate updates
        are shortened to remain inside the valid composition domain.
        """

        phi_init: np.ndarray = None
        v_init: np.ndarray = None
        phi_bin: float = None

        if (
            which_comp == 0
        ):  # Phase separation between components 1 and 2, component 0 is dilute
            chi = self.chis[0, 1] - 0.5 * (self.chis[0, 0] + self.chis[1, 1])
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            v_init = np.array([-1, -1, -1, -1])

        # Phase separation between components 0 and 2, component 1 is dilute
        elif which_comp == 1:
            chi = -0.5 * self.chis[1, 1]
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[0] = 1 - phi_init[0] - phi_init[1]
            phi_init[2] = 1 - phi_init[2] - phi_init[3]
            v_init = np.array([1, 0, 1, 0])

        # Phase separation between components 0 and 1, component 2 is dilute
        elif which_comp == 2:
            chi = -0.5 * self.chis[0, 0]
            if chi <= 2:
                return None, None
            phi_bin = self.binary_state(chi)
            phi_init = np.array([phi_bin, 1 - phi_bin, 1 - phi_bin, phi_bin]) - 1e-4
            phi_init[1] = 1 - phi_init[0] - phi_init[1]
            phi_init[3] = 1 - phi_init[2] - phi_init[3]
            v_init = np.array([0, 1, 0, 1])

        # No phase separation in this binary limit, return None to indicate failure to initialize
        if phi_bin is None:
            return None, None

        # Keep initialization in NumPy/Python; convert to JAX arrays at the end
        res = np.asarray(self._residual_jit(jnp.asarray(phi_init)))
        max_iter = 10000
        iteration = 0

        while np.linalg.norm(res) > 1e-8:
            alpha = 1.0 if np.linalg.norm(res) < 1.0 else 0.1
            v, res_jax, _ = self._projection(jnp.asarray(phi_init))
            v = np.asarray(v)
            res = np.asarray(res_jax)
            # Use JAX-compatible termination check (works both in and out of jit)
            while bool(self.is_terminate(jnp.asarray(phi_init + v * alpha))):
                alpha *= 0.5
            phi_init += v * alpha

            if iteration > max_iter:
                # TODO: remove this stamement
                return None, None
                raise RuntimeError("In binary_init: Projection not converging")
            iteration += 1

        return np.asarray(phi_init), np.asarray(v_init)


if __name__ == "__main__":
    import time

    import flory
    import matplotlib.pyplot as plt

    np.random.seed(42)
    for i in range(100):
        # Test binary_init and compare it to flory
        chi_12, chi_01 = np.random.uniform(-30, -10, 2)
        chi_02 = np.random.random() + 2.5

        chis = np.array(
            [
                [-2 * chi_01, chi_12 - chi_01 - chi_02],
                [chi_12 - chi_01 - chi_02, -2 * chi_02],
            ]
        )
        chis = np.array([[21.97142774, -14.75547769], [-14.75547769, -6.46398788]])
        obj = Stepper(chis)
        phi_init, v_init = obj.binary_init(1)

        chi_01, chi_02, chi_12 = (
            -chis[0, 0] / 2,
            -chis[1, 1] / 2,
            chis[0, 1] - (chis[0, 0] + chis[1, 1]) / 2,
        )
        chis_3 = np.array(
            [[0, chi_01, chi_02], [chi_01, 0, chi_12], [chi_02, chi_12, 0]]
        )
        if phi_init is not None:
            phi_means = [0, phi_init[::2].mean(), phi_init[1::2].mean()]
        else:
            phi_means = [0, 1e-3, 0.5]
        phi_means[0] = 1 - phi_means[1] - phi_means[2]
        phases = flory.find_coexisting_phases(3, chis_3, phi_means)
        fracs = phases.fractions[:, 1:].flatten()
        plt.plot(fracs[::2], fracs[1::2], color="blue", alpha=0.5)
        if phi_init is not None:
            plt.scatter(phi_init[::2], phi_init[1::2], color="red", alpha=0.5)
            print(
                min(
                    np.linalg.norm(phi_init - fracs),
                    np.linalg.norm(np.roll(phi_init, 2) - fracs),
                )
            )

    import matplotlib.pyplot as plt

    plt.plot(phi_init[::2], phi_init[1::2])
