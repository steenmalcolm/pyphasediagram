# %%
import numpy as np
import itertools
from shapely.geometry import LineString, Polygon
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
        self._d = []  # Denominators for barycentric coords

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
                        p1, p2, p3 = (phase11[i], phase12[i], phase22[j])
                        self.three_phase_points.append((p1, p2, p3))

        # Remove duplicates and calculate denominators for barycentric coords
        self.three_phase_points = self.remove_duplicate_points(self.three_phase_points)
        for p1, p2, p3 in self.three_phase_points:
            self._d.append(
                (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
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

    def free_energy(self, phi: np.ndarray) -> np.ndarray:
        """Compute the dimensionless free energy density for given compositions."""
        phi_d, phi_r = phi[1], phi[0]
        phi_s = 1 - phi_d - phi_r

        f = (
            phi_d * np.log(phi_d)
            + phi_r * np.log(phi_r)
            + phi_s * np.log(phi_s)
            + self.chi_dr * phi_d * phi_r
            + self.chi_ds * phi_d * phi_s
            + self.chi_rs * phi_r * phi_s
        )
        return f

    def characterize(self, delta: float = 1e-3) -> None:
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

            pos, _, _ = stepper.run(delta=delta)
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

    def get_compositions(self, phi_m: np.ndarray) -> np.ndarray:
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

    def remove_duplicate_points(self, lst, tol=1e-2):
        """Removes tuples of points that are too similar to others in the list."""
        unique = []
        canonical_reps = []  # for fast comparison

        for points in lst:
            arr = np.vstack(points)
            # For triangle each point is uniquely defined by its angle around the c.o.m.
            arr_m = arr.mean(axis=0)
            arr -= arr_m
            arr = arr[np.argsort(np.arctan2(arr[:, 1], arr[:, 0]))]
            arr += arr_m
            canonical = arr.flatten()

            # Check against existing ones
            is_similar = False
            for c in canonical_reps:
                if np.allclose(c, canonical, atol=tol):
                    is_similar = True
                    break

            if not is_similar:
                unique.append(points)
                canonical_reps.append(canonical)

        return unique

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

    def phase_map(self) -> list[np.ndarray]:
        """
        Determine the number of coexisting phases across the phase diagram.
        A value of zero indicates two-phase coexistence and one indicates three-phase coexistence.
        Anything in between indicates a transition along the tie line.
        The shape of the output list is `N_binodals` arrays with shape `(N_triple_points, N_points, 2)`
        """
        phase_counts_list = []
        for bin_idx, binodal_T in enumerate(self.binodals):

            phase_counts = np.zeros(
                (len(self.three_phase_points), binodal_T.shape[-1], 2)
            )  # shape (N_triple_points, N_points)

            # Transpose such that phase diagram coords are on axis zero
            binodal = np.transpose(binodal_T, axes=(1, 0, 2))
            free_en_bin = self.free_energy(binodal)  # shape (N_phases, N_points)
            alpha, beta, gamma = self.barycentric(
                binodal
            )  # shape (N_triple_points, N_phases, N_points)
            for tp_idx, tp_points in enumerate(self.three_phase_points):
                print("bin_case: ", self.bin_cases[bin_idx])

                free_en_tp = self.free_energy(
                    np.array(tp_points).T
                )  # shape (N_phases, )
                a, b, c = (
                    alpha[tp_idx],
                    beta[tp_idx],
                    gamma[tp_idx],
                )
                coeff1 = (
                    free_en_bin[0]
                    - a[0] * free_en_tp[0]
                    - b[0] * free_en_tp[1]
                    - c[0] * free_en_tp[2]
                )
                coeff2 = (
                    free_en_bin[1]
                    - a[1] * free_en_tp[0]
                    - b[1] * free_en_tp[1]
                    - c[1] * free_en_tp[2]
                )
                # Edge case when both coefficients are zero
                coeff1[coeff1 == 0] = -1e-10
                coeff2[coeff2 == 0] = -1e-10
                phase_count = np.zeros(
                    (binodal.shape[-1], 2)
                )  # shape (N_points, N_phases)
                # The following cases should be mutually exclusive
                # 1) The transition from 2 phase to 3 phase is along the tie line
                lmd = coeff2 / (coeff2 - coeff1)
                phase_count = (
                    -1
                    / 2
                    * (np.sign(coeff1 - coeff2) - 1)
                    * np.vstack((lmd, np.ones_like(lmd)))
                    + 1
                    / 2
                    * (np.sign(coeff1 - coeff2) + 1)
                    * np.vstack((np.zeros_like(lmd), lmd))
                ).T
                # 2) 2 phase free energy is always higher than 3 phase if binodal points
                # themselves have a lower free energy than the weighted sum of the three phases
                phase_count[((coeff1 > 0) & (coeff2 > 0))] = 1
                # 3) 3 phase free energy is always higher than 2 phase always for opposite case
                phase_count[((coeff1 < 0) & (coeff2 < 0))] = 0

                phase_counts[tp_idx] = phase_count

            phase_counts_list.append(phase_counts)
        return phase_counts_list

    def plot_phase_counts(self, tp_idx: int = 0):
        # Only consider one tripple point
        if len(self.three_phase_points) == 0:
            print("This system has no three-phase coexistence points.")
            self.plot_binodals()
            return

        tp_poly = Polygon(np.array(self.three_phase_points[tp_idx]))
        all_poly = Polygon(np.array([[0, 0], [1, 0], [0, 1]]))
        diff_all = all_poly.difference(tp_poly)
        x, y = diff_all.exterior.xy
        plt.fill(x, y, color="red", zorder=0)
        x, y = tp_poly.exterior.xy
        plt.fill(x, y, color="blue", zorder=2)
        phase_counts_list = [pc[tp_idx] for pc in self.phase_map()]

        for i, (phase_counts, binodal) in enumerate(
            zip(phase_counts_list, self.binodals)
        ):
            two_phase = phase_counts.sum(axis=1) == 0.0
            three_phase = phase_counts.sum(axis=1) == 2.0

            # Tie lines within two-phase region
            two_edges = np.diff(two_phase.astype(int))
            starts = np.where(two_edges == 1)[0] + 1
            ends = np.where(two_edges == -1)[0] + 1
            # for start, end in zip(starts, ends):
            # plt.fill(
            #     np.append(binodal[0, 0, start:end], binodal[1, 0, start:end][::-1]),
            #     np.append(binodal[0, 1, start:end], binodal[1, 1, start:end][::-1]),
            #     color="orange",
            #     zorder=2,
            # )

            # Transition along the binodal
            two_three_phase = ~(two_phase | three_phase)
            # Find disjoint segments where transition is along the binodal
            ttp_edges = np.diff(two_three_phase.astype(int))
            starts = np.where(ttp_edges == 1)[0] + 1
            ends = np.where(ttp_edges == -1)[0] + 1
            # Handle edge cases (condition true at array boundaries)
            if two_three_phase[0]:
                starts = np.r_[0, starts]
            if two_three_phase[-1]:
                ends = np.r_[ends, len(two_three_phase)]

            for j, (start, end) in enumerate(zip(starts, ends)):
                pc = phase_counts[start:end]
                b = binodal[:, :, start:end]
                x1 = b[0, 0] + (b[1, 0] - b[0, 0]) * pc[:, 0]
                y1 = b[0, 1] + (b[1, 1] - b[0, 1]) * pc[:, 0]
                x2 = b[0, 0] + (b[1, 0] - b[0, 0]) * pc[:, 1]
                y2 = b[0, 1] + (b[1, 1] - b[0, 1]) * pc[:, 1]
                plt.fill(
                    np.append(x1, x2[::-1]),
                    np.append(y1, y2[::-1]),
                    color="orange",
                    zorder=2,
                )
            # Plot binodals outside of three-phase region
            bin_poly = Polygon(np.hstack((binodal[0], binodal[1, :, ::-1])).T)
            if not bin_poly.is_valid:
                bin_poly = bin_poly.buffer(0)
            diff = bin_poly.difference(tp_poly)
            if diff.geom_type == "Polygon":
                x, y = diff.exterior.xy
                plt.fill(x, y, color="orange", zorder=3)
            elif diff.geom_type == "MultiPolygon":
                for p in diff.geoms:
                    x, y = p.exterior.xy
                    plt.fill(x, y, color="orange", zorder=3)
        plt.xlabel(r"$\phi_r$")
        plt.ylabel(r"$\phi_d$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        for p in self.three_phase_points[tp_idx]:
            plt.scatter(p[0], p[1], s=30, color="black", zorder=4)

        plt.text(
            0.3,
            0.8,
            r"$\chi_{ds}=%.2f, \chi_{dr}=%.2f, \chi_{rs}=%.2f$"
            % (self.chi_ds, self.chi_dr, self.chi_rs),
            fontsize=12,
        )
        plt.show()
        plt.close()

    def plot_binodals(self):
        """Quick visualization of binodals, tie-lines at triple points."""

        plt.figure(figsize=(6, 6))
        for i, binodal in enumerate(self.binodals):
            print(f"Plotting binodal for case {self.bin_cases[i]}", binodal.shape)
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
            for p in points:
                plt.scatter(
                    p[0],
                    p[1],
                    marker="*",
                    color="gold",
                    s=150,
                    edgecolors="k",
                    zorder=5,
                )
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
        plt.show()
