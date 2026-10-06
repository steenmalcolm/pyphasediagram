r"""High-level construction and visualization of ternary phase diagrams.

The :class:`PhaseDiagram` facade combines spinodal and binodal calculations
for an incompressible ternary mixture. It also provides convenience plots for
the resulting coexistence curves, critical points, and phase regions on
ternary axes.
"""

from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpltern import TernaryAxes
from numpy.typing import ArrayLike
from shapely.geometry.base import BaseGeometry

from pyphasediagram.binodal import Binodal
from pyphasediagram.spinodal import Spinodal
from pyphasediagram.utils import reduce_chis


def _polygon_exteriors(
    geometries: Iterable[BaseGeometry],
) -> Iterator[np.ndarray]:
    """Yield exterior coordinates from the polygonal parts of geometries.

    Parameters
    ----------
    geometries : iterable of shapely.geometry.base.BaseGeometry
        Polygonal geometries or collections that may contain polygons.

    Yields
    ------
    numpy.ndarray
        Exterior coordinates of one polygon with shape ``(n, 2)``.

    Notes
    -----
    Non-polygonal parts of geometry collections are ignored. Interior rings
    are not returned because the phase-region plotting code currently draws
    polygon exteriors only.
    """
    for geometry in geometries:
        if geometry.is_empty:
            continue
        if geometry.geom_type == "Polygon":
            yield np.asarray(geometry.exterior.coords)
        elif hasattr(geometry, "geoms"):
            yield from _polygon_exteriors(geometry.geoms)


class PhaseDiagram:
    r"""Build and visualize a ternary phase diagram.

    The supplied interaction matrix is converted to the reduced form used by
    :class:`~pyphasediagram.spinodal.Spinodal` and
    :class:`~pyphasediagram.binodal.Binodal`. Calling :meth:`build` constructs
    both objects and derives the locally stable one-, two-, and three-phase
    regions.

    Parameters
    ----------
    chis : array-like
        Full symmetric Flory--Huggins interaction matrix with shape ``(3, 3)``.
        Its off-diagonal entry ``chis[i, j]`` is the interaction parameter
        between components ``i`` and ``j``. Diagonal entries are not used.

    Attributes
    ----------
    chis : numpy.ndarray
        Reduced interaction matrix with shape ``(2, 2)``.
    spinodal : Spinodal
        Calculated spinodal curve and locally stable polygons. Available after
        :meth:`build` has completed.
    binodal : Binodal
        Calculated coexistence sections and phase-region polygons. Available
        after :meth:`build` has completed.

    Raises
    ------
    ValueError
        If ``chis`` does not have shape ``(3, 3)``.

    Notes
    -----
    Compositions are represented by the independent coordinates
    :math:`(\phi_1, \phi_2)`. Incompressibility fixes the remaining fraction
    as :math:`\phi_0 = 1 - \phi_1 - \phi_2`.

    Plotting methods place :math:`\phi_0`, :math:`\phi_1`, and :math:`\phi_2`
    on the top, left, and right ternary axes, respectively.

    The interaction matrix has the form

    .. math::

       \boldsymbol{\chi} =
       \begin{pmatrix}
       0 & \chi_{01} & \chi_{02} \\
       \chi_{01} & 0 & \chi_{12} \\
       \chi_{02} & \chi_{12} & 0
       \end{pmatrix},

    where the interaction parameters contribute to the dimensionless
    free-energy density as

    .. math::

       f(\phi_0, \phi_1, \phi_2)
       = \sum_{i=0}^{2}\phi_i\ln\phi_i
       + \phi_0\chi_{01}\phi_1
       + \phi_0\chi_{02}\phi_2
       + \phi_1\chi_{12}\phi_2.
    """

    @staticmethod
    def _to_ternary_coordinates(
        phi1: ArrayLike, phi2: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Convert independent compositions to ``mpltern`` coordinates.

        Parameters
        ----------
        phi1 : array-like
            Fraction of component 1.
        phi2 : array-like
            Fraction of component 2.

        Returns
        -------
        phi0 : numpy.ndarray
            Fraction of component 0, calculated as ``1 - phi1 - phi2``.
        phi1 : numpy.ndarray
            Fraction of component 1.
        phi2 : numpy.ndarray
            Fraction of component 2.

        Notes
        -----
        ``mpltern`` expects the fractions on the top, left, and right axes.
        This project assigns those axes to :math:`\phi_0`, :math:`\phi_1`,
        and :math:`\phi_2`, respectively.
        """
        phi1 = np.asarray(phi1)
        phi2 = np.asarray(phi2)
        phi0 = 1.0 - phi1 - phi2
        return phi0, phi1, phi2

    def __init__(self, chis: np.ndarray) -> None:
        chis = np.asarray(chis, dtype=float)
        if chis.shape != (3, 3):
            raise ValueError(
                f"PhaseDiagram requires a 3x3 interaction matrix, got shape "
                f"{chis.shape}"
            )
        self.chis = reduce_chis(chis)

    def build(self) -> None:
        """Build the spinodal and binodal parts of the phase diagram.

        Returns
        -------
        None

        Notes
        -----
        Each call creates new :class:`~pyphasediagram.spinodal.Spinodal` and
        :class:`~pyphasediagram.binodal.Binodal` instances, replacing results
        from any previous build.
        """
        self.spinodal = Spinodal(self.chis)
        self.spinodal.build()
        cps = self.spinodal.critical_points
        self.binodal = Binodal(self.chis, cps)
        self.binodal.build(unstable_manifold=self.spinodal.get_unstable_manifold())

    def get_compositions(self, phi_m: np.ndarray) -> np.ndarray:
        """Return coexisting phases for a mean composition.

        Parameters
        ----------
        phi_m : numpy.ndarray
            Mean independent composition ``[phi1, phi2]`` with shape ``(2,)``.

        Returns
        -------
        None
            This method is currently a placeholder and does not yet calculate
            coexistence compositions.

        Notes
        -----
        The return annotation describes the intended API; the current
        implementation returns ``None``.
        """
        pass

    def plot_phase_counts(self) -> tuple[Figure, TernaryAxes]:
        """Plot the one-, two-, and three-phase regions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the phase-region plot.
        ax : mpltern.TernaryAxes
            Ternary axes containing the phase-region plot.

        Notes
        -----
        :meth:`build` must be called before this method. The full composition
        simplex is drawn as the one-phase background, with calculated two- and
        three-phase polygons overlaid on it. A new figure is created on every
        call.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "ternary"})
        self._draw_phase_counts(ax)
        return fig, ax

    def _draw_phase_counts(self, ax: TernaryAxes) -> None:
        """Draw the phase-count regions on ternary axes.

        Parameters
        ----------
        ax : mpltern.TernaryAxes
            Ternary axes on which to draw.

        Returns
        -------
        None
        """

        ax.set_title("Phase regions")
        colors = {
            "1 phase": "#2ca02c",
            "2 phases": "#1f77b4",
            "3 phases": "#d62728",
        }
        alphas = {"1 phase": 1.0, "2 phases": 1.0, "3 phases": 1.0}

        # One-phase region: full domain triangle
        ax.fill(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            facecolor=colors["1 phase"],
            edgecolor="none",
            alpha=alphas["1 phase"],
        )

        # Two-phase regions
        for coords in _polygon_exteriors(self.binodal.two_phase_polygons):
            ax.fill(
                *self._to_ternary_coordinates(coords[:, 0], coords[:, 1]),
                facecolor=colors["2 phases"],
                edgecolor="none",
                alpha=alphas["2 phases"],
            )

        # Three-phase regions
        for coords in _polygon_exteriors(self.binodal.three_phase_polygons):
            ax.fill(
                *self._to_ternary_coordinates(coords[:, 0], coords[:, 1]),
                facecolor=colors["3 phases"],
                edgecolor="none",
                alpha=alphas["3 phases"],
            )

        ax.set_tlabel(r"$\phi_0$")
        ax.set_llabel(r"$\phi_1$")
        ax.set_rlabel(r"$\phi_2$")
        ax.legend(
            handles=[
                Patch(facecolor=c, alpha=alphas[l], label=l) for l, c in colors.items()
            ]
        )

    def plot(self) -> tuple[Figure, TernaryAxes]:
        """Plot coexistence curves and special points.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the phase-diagram plot.
        ax : mpltern.TernaryAxes
            Ternary axes containing the spinodal curves, binodal sections,
            three-phase polygon vertices, and critical points.

        Notes
        -----
        :meth:`build` must be called before this method. Spinodal curves are
        dashed black lines, binodal branches are red, critical points are gold
        stars, and three-phase polygon vertices are shown as markers. A new
        figure is created on every call.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "ternary"})
        self._draw_phase_diagram(ax)
        return fig, ax

    def _draw_phase_diagram(self, ax: TernaryAxes) -> None:
        """Draw coexistence curves and special points on ternary axes.

        Parameters
        ----------
        ax : mpltern.TernaryAxes
            Ternary axes on which to draw.

        Returns
        -------
        None
        """

        ax.set_title("Binodal & Spinodal")
        # Spinodal
        import networkx as nx

        for i, comp in enumerate(nx.connected_components(self.spinodal.spinodal_graph)):
            sg = self.spinodal.spinodal_graph.subgraph(comp)
            phi1s, phi2s = self.spinodal._coords_from_subgraph(sg)
            label = "Spinodal" if i == 0 else None
            ax.plot(
                *self._to_ternary_coordinates(phi1s, phi2s),
                "k--",
                label=label,
            )

        # Binodal sections
        for i, section in enumerate(self.binodal.binodal_sections):
            label = "Binodal" if i == 0 else None
            ax.plot(
                *self._to_ternary_coordinates(section.phis[0, 0], section.phis[0, 1]),
                color="red",
                label=label,
            )
            ax.plot(
                *self._to_ternary_coordinates(section.phis[1, 0], section.phis[1, 1]),
                color="red",
            )

        # Three-phase points
        # for i, coords in enumerate(
        #     _polygon_exteriors(self.binodal.three_phase_polygons)
        # ):
        #     label = "3-phase point" if i == 0 else None
        #     ax.scatter(
        #         *self._to_ternary_coordinates(coords[:-1, 0], coords[:-1, 1]),
        #         zorder=5,
        #         s=40,
        #         label=label,
        #     )

        # Critical points
        for i, cp in enumerate(self.spinodal.critical_points):
            label = "Critical point" if i == 0 else None
            ax.scatter(
                *self._to_ternary_coordinates(cp.phi1, cp.phi2),
                marker="*",
                color="gold",
                edgecolors="k",
                s=150,
                zorder=6,
                label=label,
            )

        ax.set_tlabel(r"$\phi_0$")
        ax.set_llabel(r"$\phi_1$")
        ax.set_rlabel(r"$\phi_2$")
        ax.legend()

    def plot_summary(self) -> tuple[Figure, tuple[TernaryAxes, TernaryAxes]]:
        """Plot curves and phase regions in a two-panel summary.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing both summary panels.
        axes : tuple of mpltern.TernaryAxes
            Pair ``(curve_ax, region_ax)`` containing the ternary
            coexistence-curve plot and phase-region plot, respectively.

        Notes
        -----
        :meth:`build` must be called before this method. The figure title lists
        the three independent entries of the reduced interaction matrix. A
        new figure is created on every call.
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 5), subplot_kw={"projection": "ternary"}
        )
        fig.suptitle(
            r"$(\chi_{01}, \chi_{02}, \chi_{12}) = "
            + f"({-self.chis[0, 0]/2:.4f}, {-self.chis[1, 1]/2:.4f}, {self.chis[0, 1]-self.chis[0, 0]/2-self.chis[1, 1]/2:.4f})$"
        )
        self._draw_phase_diagram(ax1)
        self._draw_phase_counts(ax2)
        fig.tight_layout()
        return fig, (ax1, ax2)


if __name__ == "__main__":
    import signal

    from tqdm import tqdm

    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    i = 0
    pbar = tqdm(total=100)
    while i < 100:
        chi = 2.75
        chi_dr, chi_rs, chi_ds = (np.random.random(3) - 0.5) * 0.4 + chi
        chi_11, chi_22, chi_12 = -2 * chi_ds, -2 * chi_rs, chi_dr - chi_rs - chi_ds
        chis = np.array([[chi_11, chi_12], [chi_12, chi_22]])
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            diagram = PhaseDiagram(chis)
            diagram.build()
            signal.alarm(0)
        except TimeoutError:
            print(f"\nIteration timed out (>60s) for chis =\n{chis}")
            continue
        except Exception as e:
            print(f"\nError for chis =\n{chis}\n{e}")
            continue
        if len(diagram.binodal.binodal_sections) == 0:
            continue
        diagram.plot_summary()
        plt.savefig(f"delete/{i+200}.png")
        plt.close()
        pbar.update(1)
        i += 1
    pbar.close()
