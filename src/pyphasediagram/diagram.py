import numpy as np
import itertools
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from pyphasediagram.stepper import BinodalStepper


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

    bin_cases = ["dr", "rs", "ds"]
    cl_cases = ["red", "green", "blue"]
    N_phases = 2
    N_components = 2

    def __init__(self, chi_dr: float, chi_rs: float, chi_ds: float):
        self.chi_dr, self.chi_rs, self.chi_ds = chi_dr, chi_rs, chi_ds
        self.binodals = []
        self.three_phase_points = []
        self.tie_lines = []

    def find_three_phase(self) -> None:
        """Finds intersection points between different binodals (three-phase coexistence)."""
        self.three_phase_points = []

        for (i1, bin1), (i2, bin2) in itertools.combinations(
            enumerate(self.binodals), 2
        ):
            for pi1 in range(2):
                phase11, phase12 = bin1[pi1].T, bin1[(pi1 + 1) % 2].T

                for pi2 in range(2):
                    phase21, phase22 = bin2[pi2].T, bin2[(pi2 + 1) % 2].T
                    indices_list = self.intercept_indices(phase11, phase21)

                    if len(indices_list) > 100:
                        raise RuntimeWarning("Overlapping binodals detected")

                    for i, j in indices_list:
                        self.three_phase_points.append(
                            (phase11[i], phase12[i], phase22[j])
                        )

    def intercept_indices(self, bin1: np.ndarray, bin2: np.ndarray):
        """Find approximate intersection indices between two binodal lines."""
        inter = LineString(bin1).intersection(LineString(bin2))
        if inter.is_empty:
            return []

        if inter.geom_type == "Point":
            intersections = [inter]
        elif inter.geom_type == "MultiPoint":
            intersections = list(inter.geoms)
            if len(intersections) > 10:
                return intersections  # likely overlap
        else:
            intersections = [
                LineString(bin1).interpolate(t, normalized=True)
                for t in np.linspace(0, 1, 10)
            ]

        segs1 = np.stack([bin1[:-1], bin1[1:]], axis=1)
        segs2 = np.stack([bin2[:-1], bin2[1:]], axis=1)
        centers1, centers2 = segs1.mean(axis=1), segs2.mean(axis=1)

        results = []
        for pt in intersections:
            p = np.array([pt.x, pt.y])
            i = np.argmin(np.linalg.norm(centers1 - p, axis=1))
            j = np.argmin(np.linalg.norm(centers2 - p, axis=1))
            results.append((i, j))

        return results

    def characterize(self):
        """Build the phase diagram: compute binodals, map to (phi_d, phi_r), and find 3-phase points."""

        chis = np.array([self.chi_dr, self.chi_rs, self.chi_ds])

        for bin_i, bin_case in enumerate(self.bin_cases):
            chi_12, chi_13, chi_23 = np.roll(chis, -bin_i)
            if chi_12 <= 2:
                print("Skipping binodal calculation for case", bin_case)
                continue

            chi_matrix = np.array(
                [
                    [-2 * chi_13, chi_12 - chi_13 - chi_23],
                    [chi_12 - chi_13 - chi_23, -2 * chi_23],
                ]
            )
            stepper = BinodalStepper(chi_matrix)

            pos, _, _ = stepper.run()
            # Map to phi_d-phi_r space
            for i in range(bin_i):
                pos = np.einsum(
                    "ij,mjn->min", np.array([[0, 1], [-1, -1]]), pos
                ) + np.array([[[0], [1]]])

            # Check if end point overlaps with starting point of previously found binodals binodal
            if bin_i and np.any(
                [
                    (np.linalg.norm(pos[:, :, -1] - b[::-1, :, 0], axis=1) < 0.01).all()
                    for b in self.binodals
                ]
            ):
                print(f"This binodal {self.bin_cases[bin_i]} has already been found")
                continue

            self.binodals.append(pos)
        self.find_three_phase()

    def get_composition(self, phi_m: np.ndarray) -> np.ndarray:
        """Given a mean composition, return the compositions of coexisting phases."""
        comp = []
        for i, binodal in enumerate(self.binodals):
            phi_displ = phi_m[:, None] - binodal[0, :, :]
            if len(self.tie_lines) <= i:
                tie_line = binodal[1, :, :] - binodal[0, :, :]
                tie_line /= np.linalg.norm(tie_line, axis=0)
                self.tie_lines.append(tie_line)
            else:
                tie_line = self.tie_lines[i]

            dist_to_line = np.abs(
                phi_displ[0] * tie_line[1] - phi_displ[1] * tie_line[0]
            )
            phi_comp = binodal[:, :, np.argmin(dist_to_line)]
            comp_angles = phi_comp[:, 1] / phi_comp[:, 0]

            if (  # Outside of binodal
                phi_m[1] / phi_m[0] < comp_angles.min()
                or phi_m[1] / phi_m[0] > comp_angles.max()
            ):
                comp.append(phi_m)

            else:  # Inside
                comp.append(phi_comp)

        return comp

    def plot_binodals(self):
        """Quick visualization of binodals, tie-lines at triple points."""

        plt.figure(figsize=(6, 6))
        for i, binodal in enumerate(self.binodals):
            bin_case = self.bin_cases[i]
            plt.plot(binodal[0, 0], binodal[0, 1], color=self.cl_cases[i])
            plt.plot(
                binodal[1, 0],
                binodal[1, 1],
                label=f"{bin_case}",
                color=self.cl_cases[i],
            )
        for points in self.three_phase_points:
            for p1, p2 in itertools.combinations(points, 2):
                plt.plot(
                    [
                        p1[0],
                        p2[0],
                    ],
                    [p1[1], p2[1]],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                )
            # Plot a start
            plt.scatter(
                points[0][0],
                points[0][1],
                marker="*",
                color="gold",
                s=150,
                edgecolors="k",
                zorder=5,
            )
        for i in range(20):
            phi_m = np.random.random(2)
            while phi_m.sum() >= 1:
                phi_m = np.random.random(2)
            phi_comps = self.get_composition(phi_m)

            plt.scatter(phi_m[0], phi_m[1], color="black", s=15)
            for c in phi_comps:
                if c.shape == (2, 2):
                    plt.plot(c[:, 0], c[:, 1], "k-.", alpha=0.5, lw=0.5)
                    plt.scatter(phi_m[0], phi_m[1], color="orange", s=15)
        plt.plot([0, 1], [1, 0], "k--")
        plt.text(
            0.3,
            0.8,
            r"$\chi_{ds}=%.2f, \chi_{dr}=%.2f, \chi_{rs}=%.2f$"
            % (self.chi_ds, self.chi_dr, self.chi_rs),
            fontsize=12,
        )
        plt.xlabel(r"$\phi_r$")
        plt.ylabel(r"$\phi_d$")
        plt.legend()

        plt.xlim(0, 1)
        plt.ylim(0, 1)


if __name__ == "__main__":
    chi_ds, chi_dr, chi_rs = 2.66, 2.61, 2.62
    diagram = PhaseDiagram(chi_dr, chi_rs, chi_ds)
    diagram.characterize()
    diagram.plot_binodals()
