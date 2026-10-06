"""Thermodynamic helper functions for incompressible ternary mixtures.

Components are indexed ``0, 1, 2`` with component 0 acting as the solvent.
The independent composition coordinates are :math:`(\\phi_1, \\phi_2)` and the
solvent fraction follows from incompressibility,
:math:`\\phi_0 = 1 - \\phi_1 - \\phi_2`.
"""

import numpy as np
from numpy.typing import ArrayLike


def reduce_chis(chis: ArrayLike) -> np.ndarray:
    r"""Convert a full interaction matrix to the reduced ``(2, 2)`` form.

    Parameters
    ----------
    chis : array-like
        Symmetric Flory--Huggins interaction matrix with shape ``(3, 3)``.
        Only the off-diagonal entries :math:`\chi_{01}`, :math:`\chi_{02}`,
        and :math:`\chi_{12}` are used. A matrix that is already reduced,
        with shape ``(2, 2)``, is returned unchanged.

    Returns
    -------
    numpy.ndarray
        Reduced interaction matrix :math:`C` with shape ``(2, 2)`` acting on
        the independent compositions :math:`(\phi_1, \phi_2)`:

        .. math::

           C = \begin{pmatrix}
           -2\chi_{01} & \chi_{12} - \chi_{01} - \chi_{02} \\
           \chi_{12} - \chi_{01} - \chi_{02} & -2\chi_{02}
           \end{pmatrix}.

    Raises
    ------
    ValueError
        If ``chis`` does not have shape ``(3, 3)`` or ``(2, 2)``.
    """
    chis = np.asarray(chis, dtype=float)
    if chis.shape == (2, 2):
        return chis
    if chis.shape != (3, 3):
        raise ValueError(f"chis must be a 2x2 or a 3x3 array, but got {chis.shape}")

    chi_01, chi_02, chi_12 = chis[0, 1], chis[0, 2], chis[1, 2]
    off_diagonal = chi_12 - chi_01 - chi_02
    return np.array([[-2 * chi_01, off_diagonal], [off_diagonal, -2 * chi_02]])


def _as_compositions(phi: ArrayLike) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 0 or phi.shape[0] != 2:
        raise ValueError(
            f"phi must have a leading axis of length 2, but got shape {phi.shape}"
        )
    return phi


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
        If ``chis`` does not have shape ``(3, 3)`` or ``(2, 2)``, or the
        first axis of ``phi`` does not have length 2.

    Notes
    -----
    With the reduced interaction matrix :math:`C` from :func:`reduce_chis`
    and :math:`\phi_0 = 1 - \phi_1 - \phi_2`,

    .. math::

       f(\boldsymbol{\phi}) = \sum_{i=0}^{2}\phi_i\ln\phi_i
       + \tfrac{1}{2}\boldsymbol{\phi}^{\mathsf{T}} C\boldsymbol{\phi}.
    """
    phi = _as_compositions(phi)
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
        If ``chis`` does not have shape ``(3, 3)`` or ``(2, 2)``, or the
        first axis of ``phi`` does not have length 2.

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
    phi = _as_compositions(phi)
    reduced_chis = reduce_chis(chis)
    phi_0 = 1.0 - phi.sum(axis=0)
    return np.log(phi) - np.log(phi_0) + reduced_chis @ phi
