import numpy as np
import matplotlib.pyplot as plt
import shapely

from pyphasediagram.spinodal import Spinodal
from pyphasediagram.binodal import Binodal


class PhaseDiagram:
    """
    Builds and analyzes a ternary phase diagram from binodal segments.

    Runs `BinodalStepper` for each binary limit
    (droplet-regulator, regulator-solvent, droplet-solvent),
    maps the resulting binodals into (phi_d, phi_r) space, detects three-phase
    coexistence points from binodal intersections, and finds the compositions of
    the coexisting phases given a mean composition.

    Parameters
    ----------
    chi_dr : float
        Flory–Huggins interaction parameter droplet–regulator
    chi_rs : float
        Flory–Huggins interaction parameter regulator–solvent
    chi_ds : float
        Flory–Huggins interaction parameter droplet–solvent

    """

    def __init__(self, chis):
        if len(chis) == 3:
            chis = np.array(
                [
                    [-2 * chis[0, 1], chis[1, 2] - chis[0, 1] - chis[0, 2]],
                    [chis[1, 2] - chis[0, 1] - chis[0, 2], -2 * chis[0, 2]],
                ]
            )
        elif len(chis) != 2:
            raise ValueError("chis must be a 2x2 or a 3x3 array")

        self.chis = chis

    def _free_energy(self, phi: np.ndarray) -> np.ndarray:
        """Compute the dimensionless free energy density for given compositions."""
        phi_1, phi_2 = phi[0], phi[1]
        phi_0 = 1 - phi_1 - phi_2

        f = (
            phi_1 * np.log(phi_1)
            + phi_2 * np.log(phi_2)
            + phi_0 * np.log(phi_0)
            + np.dot(phi, self.chis @ phi)
        )
        return f

    def build(self, delta: float = 1e-3) -> None:
        """Build the phase diagram: compute binodals, map to (phi_d, phi_r), and find 3-phase points."""
        self.spinodal = Spinodal(self.chis)
        self.spinodal.build()
        cps = self.spinodal.critical_points
        self.binodal = Binodal(self.chis, cps)
        self.binodal.build(unstable_manifold=self.spinodal.get_unstable_manifold())

    def get_compositions(self, phi_m: np.ndarray) -> np.ndarray:
        """Given a mean composition, return the compositions of coexisting phases."""
        pass

    def barycentric(self, phi_m: np.ndarray):
        """
        Compute (α, β, γ) such that phi_m = α phi^1 + β phi^2 + γ phi^3 and α+β+γ = 1.
        """

        if phi_m.ndim == 1:
            alpha, beta, gamma = np.zeros((3, len(self._d)))
        else:
            alpha, beta, gamma = np.zeros((3, len(self._d), *phi_m.shape[1:]))

        for i, ((p1, p2, p3), d) in enumerate(zip(self.three_phase_points, self._d)):
            alpha[i] = (phi_m[0] - p3[0]) * (p2[1] - p3[1]) - (phi_m[1] - p3[1]) * (
                p2[0] - p3[0]
            )
            alpha[i] /= d
            beta[i] = (p1[0] - p3[0]) * (phi_m[1] - p3[1]) - (p1[1] - p3[1]) * (
                phi_m[0] - p3[0]
            )
            beta[i] /= d
            gamma[i] = 1 - alpha[i] - beta[i]

        return alpha, beta, gamma

    def plot_phase_counts(self, ax=None):
        """Visualize the number of coexisting phases across the composition space."""
        from matplotlib.patches import Polygon as MplPolygon, Patch
        from matplotlib.collections import PatchCollection

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.set_title("Phase regions")
        colors = {"1 phase": "#2ca02c", "2 phases": "#1f77b4", "3 phases": "#d62728"}
        alphas = {"1 phase": 0.5, "2 phases": 0.6, "3 phases": 0.7}

        # One-phase region: full domain triangle
        domain = MplPolygon([(0, 0), (0, 1), (1, 0)], closed=True)
        ax.add_collection(
            PatchCollection(
                [domain],
                facecolors="#2ca02c",
                edgecolors="none",
                alpha=0.5,
                label="1 phase",
            )
        )

        # Two-phase regions
        two_patches = []
        for poly in self.binodal.two_phase_polygons:
            if poly.geom_type == "MultiPolygon":
                for p in poly.geoms:
                    two_patches.append(
                        MplPolygon(np.array(p.exterior.coords), closed=True)
                    )
            else:
                two_patches.append(
                    MplPolygon(np.array(poly.exterior.coords), closed=True)
                )
        if two_patches:
            ax.add_collection(
                PatchCollection(
                    two_patches,
                    facecolors="#1f77b4",
                    edgecolors="none",
                    alpha=0.6,
                    label="2 phases",
                )
            )

        # Three-phase regions
        three_patches = []
        for poly in self.binodal.three_phase_polygons:
            if poly.geom_type == "MultiPolygon":
                for p in poly.geoms:
                    three_patches.append(
                        MplPolygon(np.array(p.exterior.coords), closed=True)
                    )
            else:
                three_patches.append(
                    MplPolygon(np.array(poly.exterior.coords), closed=True)
                )
        if three_patches:
            ax.add_collection(
                PatchCollection(
                    three_patches,
                    facecolors="#d62728",
                    edgecolors="none",
                    alpha=0.7,
                    label="3 phases",
                )
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        ax.set_aspect("equal")
        ax.legend(
            handles=[
                Patch(facecolor=c, alpha=alphas[l], label=l) for l, c in colors.items()
            ]
        )
        return fig, ax

    def plot(self, ax=None):
        """Quick visualization of spinodal, binodal sections, and three-phase points."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.set_title("Binodal & Spinodal")
        # Spinodal
        import networkx as nx

        for i, comp in enumerate(nx.connected_components(self.spinodal.spinodal_graph)):
            sg = self.spinodal.spinodal_graph.subgraph(comp)
            phi1s, phi2s = self.spinodal._coords_from_subgraph(sg)
            label = "Spinodal" if i == 0 else None
            ax.plot(phi1s, phi2s, "k--", label=label)

        # Binodal sections
        for i, section in enumerate(self.binodal.binodal_sections):
            label = "Binodal" if i == 0 else None
            ax.plot(section.phis[0, 0], section.phis[0, 1], color="red", label=label)
            ax.plot(section.phis[1, 0], section.phis[1, 1], color="red")

        # Three-phase points
        for i, poly in enumerate(self.binodal.three_phase_polygons):
            coords = np.array(poly.exterior.coords)
            label = "3-phase point" if i == 0 else None
            ax.scatter(coords[:-1, 0], coords[:-1, 1], zorder=5, s=40, label=label)

        # Critical points
        for i, cp in enumerate(self.spinodal.critical_points):
            label = "Critical point" if i == 0 else None
            ax.scatter(
                cp.phi1,
                cp.phi2,
                marker="*",
                color="gold",
                edgecolors="k",
                s=150,
                zorder=6,
                label=label,
            )

        ax.plot([0, 1], [1, 0], "k-", linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        ax.set_aspect("equal")
        ax.legend()
        return fig, ax

    def plot_summary(self):
        """Side-by-side subplots: binodal/spinodal (left) and phase regions (right)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            r"$(\chi_{11}, \chi_{12}, \chi_{22}) = "
            + f"({self.chis[0, 0]:.5f}, {self.chis[0, 1]:.5f}, {self.chis[1, 1]:.5f})$"
        )
        self.plot(ax=ax1)
        self.plot_phase_counts(ax=ax2)
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
