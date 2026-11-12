"""
Coexistence-curve tracing in a four-component mixture with the exchange chemical potential
of component 3 held fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

import pyphasediagram as ppd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class FourCompStepper(ppd.stepper.base.BaseStepper):
    r"""Trace constrained coexistence curves in a four-component mixture.

    Parameters
    ----------
    chis : array-like
        Full symmetric Flory--Huggins interaction matrix with shape ``(4, 4)``.
        Component 0 is eliminated by incompressibility.

    Attributes
    ----------
    chis : jax.Array
        Full interaction matrix with shape ``(4, 4)``.
    chis_reduced : jax.Array
        Interaction matrix after eliminating component 0, with shape ``(3, 3)``.
    mu3_star : float
        Fixed chemical potential of component 3.

    Notes
    -----
    The compositions are ordered as

    .. math::

       \boldsymbol{\phi} =
       (\phi_1^{(a)}, \phi_2^{(a)},
       \phi_1^{(b)}, \phi_2^{(b)},
       \phi_3^{(a)}, \phi_3^{(b)})

    where :math:`a` and :math:`b` label the two phases. In each phase,

    .. math::

       \phi_0 = 1 -  \phi_1 - \phi_2 - \phi_3

    """

    _BINARY_INIT_MAX_ITERATIONS = 10_000

    def __init__(self, chis: ArrayLike) -> None:
        chis_array = np.asarray(chis, dtype=float)
        if chis_array.shape != (4, 4):
            msg = f"chis must have shape (4, 4), but got {chis_array.shape}"
            raise ValueError(msg)

        self.chis = jnp.asarray(chis_array)
        self.chis_reduced = jnp.asarray(ppd.utils.reduce_chis(chis_array, ncomp=4))
        self.mu3_star: float = None

        super().__init__()

    def residual(self, phi: jnp.ndarray) -> jnp.ndarray:
        r"""Evaluate the constrained two-phase coexistence conditions.

        Parameters
        ----------
        phi : jax.Array
            Phase compositions with shape ``(6,)`` and ordering
            ``[phi1_a, phi2_a, phi3_a, phi1_b, phi2_b, phi3_b]``.

        Returns
        -------
        jax.Array
            Five residuals: equal exchange chemical potentials of components 1 and 2,
            fixed chemical potential of component 3 in both phases and equal pressure
        Notes
        -----
        The equal chemical potentials of components 1 and 2 are:
        .. math::

           \mu_1^{(a)} = \mu_1^{(b)}, \quad
           \mu_2^{(a)} = \mu_2^{(b)}.

        Fixed chemical potential of component 3 is
        .. math::

           \mu_3^{(a)} = \mu_3^{(b)} = \mu_3^*.

        Equal pressure in both phases is
        .. math::

           \Pi_c^{(a)} = \Pi_c^{(b)}.

        """
        phis_phase = phi.reshape(2, 3)
        phi0_phase = 1 - phis_phase.sum(axis=1)
        chi_prod = phis_phase @ self.chis_reduced.T

        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod
        mu12_diff = mu[0, :2] - mu[1, :2]

        mu3_fixed = jnp.array([mu[0, 2] - self.mu3_star, mu[1, 2] - self.mu3_star])

        pressure = -jnp.log(phi0_phase)
        pressure += 0.5 * (phis_phase * chi_prod).sum(axis=1)
        pressure_diff = pressure[0] - pressure[1]

        return jnp.hstack((mu12_diff, mu3_fixed, pressure_diff))

    def is_terminate(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Determine whether curve tracing should stop.

        Parameters
        ----------
        phi : jax.Array
            Phase compositions with shape ``(6,)`` and ordering
            ``[phi1_a, phi2_a, phi3_a, phi1_b, phi2_b, phi3_b]``.

        Returns
        -------
        jax.Array
            Scalar Boolean array. It is true when the phases differ by less
            than ``1e-3`` or any phase has a nonpositive fraction.
        """
        close = jnp.linalg.norm(phi[3:] - phi[:3]) < 1e-3
        nonpositive = jnp.any(phi <= 0.0)
        nonpositive = jnp.logical_or(nonpositive, phi[:3].sum() >= 1)
        nonpositive = jnp.logical_or(nonpositive, phi[3:].sum() >= 1)
        return jnp.logical_or(close, nonpositive)

    def run(
        self,
        phi_init: np.ndarray | jnp.ndarray,
        v_init: np.ndarray | jnp.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:

        if self.mu3_star is not None:
            msg = (
                "Create a new FourCompStepper instance for each run to avoid "
                "reusing a JIT-cached chemical potential with the wrong fixed chemical potential."
            )
            raise RuntimeError(msg)
        # Calculate mu3_star from the initial composition
        phis_phase = phi_init.reshape(2, 3)
        phi0_phase = 1 - phis_phase.sum(axis=1)
        chi_prod = phis_phase @ self.chis_reduced.T
        mu = jnp.log(phis_phase) - jnp.log(phi0_phase)[:, None] + chi_prod
        self.mu3_star = mu[0, 2]
        return super().run(phi_init, v_init, **kwargs)


if __name__ == "__main__":
    import flory
    import matplotlib.pyplot as plt

    num_comp = 4  # Set number of components
    chis = [
        [0.0, 1.9, 0.5, 5.0],
        [1.9, 0.0, 0.0, -1.6],
        [0.5, 0.0, 0.0, 0.9],
        [5.0, -1.6, 0.9, 0.0],
    ]  # Set the \chi matrix
    chis = [
        [0.0, -1.6, 0.9, 5.0],
        [-1.6, 0.0, 0.0, 1.9],
        [0.9, 0.0, 0.0, 0.5],
        [5.0, 1.9, 0.5, 0.0],
    ]
    # Keep phi1 dilute
    phi2_bar = 0.01

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        color="gray",
        alpha=0.2,
        label=r"$\phi_0 = 0$",
    )

    for i, phi1_bar in enumerate(np.linspace(0.01, 0.99, 10)):
        phii_bar = (1 - phi2_bar - phi1_bar) / 2
        phi_means = [phii_bar, phi1_bar, phi2_bar, phii_bar]
        if sum(phi_means) != 1 and any(phi < 0 for phi in phi_means):
            msg = "Invalid composition"
            raise ValueError(msg)

        # obtain coexisting phases
        # create a free energy
        fh = flory.FloryHuggins(num_comp, chis)
        # create a ensemble
        ensemble = flory.CanonicalEnsemble(num_comp, phi_means)
        # construct a finder from interaction, entropy and ensemble
        finder = flory.CoexistingPhasesFinder(fh.interaction, fh.entropy, ensemble)
        # obtain phases by clustering compartments
        phases = finder.run(progress=False).get_clusters()

        phis_init = np.array(phases.fractions[:, 1:]).flatten()
        if len(phis_init) != 6:
            continue
        # Initialize the direction away from the physically admissible boundary
        v_init = -np.sign(phis_init - 0.5)
        stepper = FourCompStepper(chis)
        phis, svs, flags = stepper.run(phis_init, v_init)

        for phase in [0, 1]:
            if i == 0 and phase == 0:
                ax.plot(*phis[phase], color="C0", label="binodal")
            ax.plot(*phis[phase], color="C0")
        if i == 0:
            ax.scatter(*phis[0, :, -1], color="red", s=25, label="critical point")
        ax.scatter(*phis[0, :, -1], color="red", s=25)

    ax.set(xlabel=r"$\phi_1$", ylabel=r"$\phi_2$", zlabel=r"$\phi_3$")
    ax.view_init(elev=64, azim=147, roll=-125)
    ax.legend()
    plt.show()
