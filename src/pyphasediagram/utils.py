"""Thermodynamic helper functions for incompressible ternary mixtures.

Components are indexed ``0, 1, 2`` with component 0 acting as the solvent.
The independent composition coordinates are :math:`(\\phi_1, \\phi_2)` and the
solvent fraction follows from incompressibility,
:math:`\\phi_0 = 1 - \\phi_1 - \\phi_2`.
"""

import numpy as np
from numpy.typing import ArrayLike


def reduce_chis(chis: ArrayLike, ncomp: int = 3) -> np.ndarray:
    r"""Convert a full Flory--Huggins interaction matrix to reduced form.

    The concentration of species 0 is eliminated using incompressibility,
    leaving ``ncomp - 1`` independent concentrations.

    Parameters
    ----------
    chis : array-like
        Symmetric Flory--Huggins interaction matrix with shape
        ``(ncomp, ncomp)``. Only off-diagonal entries are used.
        A matrix that is already reduced, with shape
        ``(ncomp - 1, ncomp - 1)``, is returned unchanged.
    ncomp : int, optional
        Total number of components, including the eliminated species 0.
        Defaults to 3.

    Returns
    -------
    numpy.ndarray
        Reduced interaction matrix with shape
        ``(ncomp - 1, ncomp - 1)``, acting on the independent
        compositions :math:`(\phi_1, \ldots, \phi_{N-1})`.

        Its elements are given by

        .. math::

            \tilde{\chi}_{ij}
            = \chi_{ij} - \chi_{0i} - \chi_{0j},

        for :math:`i,j \in \{1,\ldots,N-1\}`, with
        :math:`\chi_{ii}=0`.

    Raises
    ------
    ValueError
        If ``ncomp`` is smaller than 2 or ``chis`` does not have
        shape ``(ncomp, ncomp)`` or ``(ncomp - 1, ncomp - 1)``.

    Notes
    -----
    The full Flory--Huggins interaction free-energy density is

    .. math::

        f_{\mathrm{int}}
        = \sum_{0 \leq i < j < N} \chi_{ij} \phi_i \phi_j.

    Eliminating species 0 through the incompressibility constraint

    .. math::

        \phi_0 = 1 - \sum_{i=1}^{N-1} \phi_i

    gives, up to terms linear in the independent concentrations,

    .. math::

        f_{\mathrm{int}}
        = \frac{1}{2}
          \sum_{i,j=1}^{N-1}
          \tilde{\chi}_{ij} \phi_i \phi_j
          + \text{linear terms}.
    """
    if ncomp < 2:
        msg = "ncomp must be at least 2"
        raise ValueError(msg)

    chis = np.asarray(chis, dtype=float)

    if chis.shape == (ncomp - 1, ncomp - 1):
        return chis

    if chis.shape != (ncomp, ncomp):
        msg = (
            f"chis must have shape ({ncomp}, {ncomp}) or "
            f"({ncomp - 1}, {ncomp - 1}), but got {chis.shape}"
        )
        raise ValueError(msg)

    chi_0 = chis[0, 1:]
    reduced = chis[1:, 1:] - chi_0[:, None] - chi_0[None, :]

    # Ignore the diagonal entries of the original interaction matrix.
    np.fill_diagonal(reduced, -2 * chi_0)

    return reduced


def free_energy(phi: ArrayLike, chis: ArrayLike) -> np.ndarray:
    r"""Calculate the Flory--Huggins free-energy density of mixing.

    Parameters
    ----------
    phi : array-like
        Independent compositions :math:`(\phi_1, \phi_2)` with shape ``(2,)``
        or ``(2, n_points)``. All of :math:`\phi_1`, :math:`\phi_2`, and
        :math:`\phi_0` must be positive.
    chis : array-like
        Flory--Huggins interaction matrix with shape ``(3, 3)``. An already
        reduced matrix with shape ``(2, 2)`` is also accepted.

    Returns
    -------
    numpy.ndarray
        Dimensionless free-energy density per lattice site with shape
        ``()`` for a single composition or ``(n_points,)``.

    Raises
    ------
    ValueError
        If ``chis`` does not have shape ``(3, 3)`` or ``(2, 2)``.

    Notes
    -----
    With the reduced interaction matrix :math:`C` from :func:`reduce_chis`
    and :math:`\phi_0 = 1 - \phi_1 - \phi_2`,

    .. math::

       f(\boldsymbol{\phi}) = \sum_{i=0}^{2}\phi_i\ln\phi_i
       + \tfrac{1}{2}\boldsymbol{\phi}^{\mathsf{T}} C\boldsymbol{\phi}.
    """
    phi = np.asarray(phi, dtype=float)
    reduced_chis = reduce_chis(chis)
    phi_0 = 1.0 - phi.sum(axis=0)
    return (
        np.sum(phi * np.log(phi), axis=0)
        + phi_0 * np.log(phi_0)
        + np.sum(phi * (reduced_chis @ phi), axis=0) / 2
    )


def exchange_chemical_potentials(phi: ArrayLike, chis: ArrayLike) -> np.ndarray:
    r"""Calculate the exchange chemical potentials relative to the solvent.

    Parameters
    ----------
    phi : array-like
        Independent compositions :math:`(\phi_1, \phi_2)` with shape ``(2,)``
        or ``(2, n_points)``. All of :math:`\phi_1`, :math:`\phi_2`, and
        :math:`\phi_0` must be positive.
    chis : array-like
        Flory--Huggins interaction matrix with shape ``(3, 3)``. An already
        reduced matrix with shape ``(2, 2)`` is also accepted.

    Returns
    -------
    numpy.ndarray
        Dimensionless exchange chemical potentials
        :math:`\mu_i - \mu_0` for :math:`i \in \{1, 2\}`, with the same shape
        as ``phi``.

    Raises
    ------
    ValueError
        If ``chis`` does not have shape ``(3, 3)`` or ``(2, 2)``.

    Notes
    -----
    With the reduced interaction matrix :math:`C` from :func:`reduce_chis`,

    .. math::

       \mu_i - \mu_0 = \ln\phi_i - \ln\phi_0 + (C\boldsymbol{\phi})_i,
       \qquad i \in \{1, 2\},

    which is the derivative :math:`\partial f / \partial \phi_i` of the
    free-energy density
    :math:`f = \sum_{i=0}^{2}\phi_i\ln\phi_i
    + \tfrac{1}{2}\boldsymbol{\phi}^{\mathsf{T}} C\boldsymbol{\phi}`.
    """
    phi = np.asarray(phi, dtype=float)
    reduced_chis = reduce_chis(chis)
    phi_0 = 1.0 - phi.sum(axis=0)
    return np.log(phi) - np.log(phi_0) + reduced_chis @ phi
