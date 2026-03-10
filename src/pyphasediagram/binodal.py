import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import shapely
from pyphasediagram.point import CriticalPoint, BinodalInitialPoint
from pyphasediagram.stepper import Stepper


class BinodalSection:

    def __init__(self, phis: np.ndarray, svs: np.ndarray):
        self.phis = phis
        self.svs = svs
        self.line_a = shapely.LineString(phis[0].T)
        self.line_b = shapely.LineString(phis[1].T)
        self.line_ab = shapely.LineString(phis.reshape(4, -1).T)

    def plot(self, colora="red", colorb="blue", **kwargs):
        plt.plot(self.phis[0, 0], self.phis[0, 1], color=colora, **kwargs)

        plt.plot(self.phis[1, 0], self.phis[1, 1], color=colorb, **kwargs)

    def intersects_with(self, other):
        """Check if this binodal section intersects with another binodal section by checking if their corresponding lines intersect using shapely."""
        if not isinstance(other, BinodalSection):
            raise NotImplementedError(
                "intersects_with can only be computed between BinodalSection instances."
            )

        def add_intersection(l1a, l2a):
            i_points = []
            i = l1a.intersection(l2a)

            if i.geom_type == "Point":
                idxs = []
                for l in [l1a, l2a]:
                    coords = list(l.coords)
                    idx_closest = min(
                        range(len(coords)),
                        key=lambda j: i.distance(shapely.Point(coords[j])),
                    )
                    idxs.append(idx_closest)
                i_points.append(idxs)

            elif i.geom_type == "MultiPoint":
                for p in i.geoms:
                    idxs = []
                    for l in [l1a, l2a]:
                        coords = list(l.coords)
                        idx_closest = min(
                            range(len(coords)),
                            key=lambda j: p.distance(shapely.Point(coords[j])),
                        )
                        idxs.append(idx_closest)
                    i_points.append(idxs)
            return i_points

        tp_points = []
        i_points = add_intersection(self.line_a, other.line_a)
        for idxs in i_points:
            tp_points.append(
                (
                    self.phis[0, :, idxs[0]],
                    other.phis[0, :, idxs[1]],
                    self.phis[1, :, idxs[0]],
                    other.phis[1, :, idxs[1]],
                )
            )

        i_points = add_intersection(self.line_a, other.line_b)
        for idxs in i_points:
            tp_points.append(
                (
                    self.phis[0, :, idxs[0]],
                    other.phis[1, :, idxs[1]],
                    self.phis[1, :, idxs[0]],
                    other.phis[0, :, idxs[1]],
                )
            )

        i_points = add_intersection(self.line_b, other.line_a)
        for idxs in i_points:
            tp_points.append(
                (
                    self.phis[1, :, idxs[0]],
                    other.phis[0, :, idxs[1]],
                    self.phis[0, :, idxs[0]],
                    other.phis[1, :, idxs[1]],
                )
            )

        i_points = add_intersection(self.line_b, other.line_b)
        for idxs in i_points:
            tp_points.append(
                (
                    self.phis[1, :, idxs[0]],
                    other.phis[1, :, idxs[1]],
                    self.phis[0, :, idxs[0]],
                    other.phis[0, :, idxs[1]],
                )
            )

        # Plot self.line_a and self.line_b
        return tp_points

    def contained_in(self, other, n_samples=1000):
        """Check if this binodal section is contained within another binodal section by"""
        if not isinstance(other, BinodalSection):
            raise NotImplementedError(
                "contained_in can only be computed between BinodalSection instances."
            )

        # System invariant under swap of phase labeling
        for la, lb in [(self.line_a, self.line_b), (self.line_b, self.line_a)]:

            sa = np.linspace(0.0, la.length, n_samples)
            da = np.array([la.interpolate(si).distance(other.line_a) for si in sa])

            sb = np.linspace(0.0, lb.length, n_samples)
            db = np.array([lb.interpolate(si).distance(other.line_b) for si in sb])

            if np.mean(da) < 1e-3 and np.mean(db) < 1e-3:
                return True
        return False

    def _degenerate_points(
        self, sv_branch_threshold: float = 1e-2
    ) -> list[BinodalInitialPoint]:
        """Identify points where the null space of the Jacobian has dimension greater than 1, which indicates branching"""
        degenerate_points = []
        extrema_idx = np.where(np.diff(np.sign(np.diff(self.svs))) == 2)[0] + 1
        for idx in extrema_idx:
            if self.svs[idx] < sv_branch_threshold:
                phi_init = self.phis[:, :, idx].flatten()
                v_init = (
                    self.phis[:, :, idx + 1].flatten()
                    - self.phis[:, :, idx - 1].flatten()
                )
                degenerate_points.append(BinodalInitialPoint(phi_init, v_init))
        return degenerate_points

    def __len__(self):
        return self.phis.shape[2]


class Binodal:

    SV_BRANCH_THRESHOLD = 1e-2

    def __init__(self, chis: np.ndarray, critical_points: list[CriticalPoint] = []):
        """Initialize the Binodal class with a 2x2 matrix of chi parameters"""
        self.chis = chis
        self.critical_points = critical_points
        self.binodal_sections: list[BinodalSection] = []
        self._stepper = Stepper(chis)
        self._bipt_task_list: list[BinodalInitialPoint] = []
        self._bipt_hist: list[BinodalInitialPoint] = []
        self._three_phase_points = []

    def _build_section(self, phi_init, v_init):
        """
        Build a section of the binodal curve starting from an initial composition phi_init and initial step v_init. This method runs the stepper to compute the binodal points along the section and adds them as nodes in the binodal graph, connecting consecutive points with edges.
        """
        # Run the stepper to compute the binodal points along the section
        phis, svs, flags = self._stepper.run(phi_init, v_init)
        # if len(flags):
        #     print("\t", end=" ")
        #     for flag in flags:
        #         print(f"{flag}", end=", ")
        #     print()
        bb_new = BinodalSection(phis, svs)

        # Avoid duplicates
        for bb in self.binodal_sections:
            if bb_new.contained_in(bb):
                return False

        # Find branching points and add them to the task list
        self._bipt_task_list.extend(self._find_branching_points(bb_new))

        # Save the new section and its initial point
        self.binodal_sections.append(BinodalSection(phis, svs))
        self._bipt_hist.append(BinodalInitialPoint(phi_init, v_init))
        return True

    def build(self):
        """
        Build the binodal curve as a graph with nodes representing points (phi1a, phi2a, phi1b, phi2b) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.binodal_graph.
        """
        # Sections from binary limits
        for which_comp in range(3):
            phi_init, v_init = self._stepper.binary_init(which_comp)
            if isinstance(phi_init, np.ndarray):
                self._bipt_task_list.append(BinodalInitialPoint(phi_init, v_init))

        # Sections from critical points
        for cpt in self.critical_points:
            phi_init, v_init = cpt.get_phi_and_v_init()
            self._bipt_task_list.append(BinodalInitialPoint(phi_init, v_init))

        while len(self._bipt_task_list):
            bipt = self._bipt_task_list.pop(0)
            if any(bipt.is_similar_to(other) for other in self._bipt_hist):
                continue
            self._build_section(*bipt)

        self._find_three_phase_points()

    def _find_branching_points(
        self, section: BinodalSection
    ) -> list[BinodalInitialPoint]:
        """Identify points where the null space of the Jacobian has dimension greater than 1, which indicates branching"""
        branching_points: list[BinodalInitialPoint] = []
        for bp in section._degenerate_points(self.SV_BRANCH_THRESHOLD):
            J = self._stepper._jac_fn(bp.phi_init)
            U, S, Vt = self._stepper._svd(J, full_matrices=True)
            if np.dot(Vt[-1], bp.v_init) > 0.98 and S[-1] < self.SV_BRANCH_THRESHOLD:
                bp.v_init = np.asarray(Vt[-2])
                bp.phi_init += bp.v_init * 1e-3
                branching_points.append(bp)

        return branching_points

    def _find_three_phase_points(self):
        for i, section_a in enumerate(self.binodal_sections):
            for j, section_b in enumerate(self.binodal_sections):
                if i >= j:
                    continue
                i_points = section_a.intersects_with(section_b)
                for i_point in i_points:
                    self._three_phase_points.append(i_point)

    def plot(self, **kwargs):
        for section in self.binodal_sections:
            section.plot(**kwargs)


if __name__ == "__main__":
    from pyphasediagram.spinodal import Spinodal
    import tqdm

    chi = 2.6
    chi_12, chi_01, chi_02 = (chi, chi, chi)
    chis = np.array(
        [
            [-2 * chi_01, chi_12 - chi_01 - chi_02],
            [chi_12 - chi_01 - chi_02, -2 * chi_02],
        ]
    )

    sp_obj = Spinodal(chis)
    sp_obj.build()
    cps = sp_obj.critical_points

    obj = Binodal(chis, cps)
    obj.build()

    sp_obj.plot()
    for section in obj.binodal_sections:
        section.plot()
    polys = [shapely.Polygon(t) for t in obj._three_phase_points]
    poly = polys[0]
    for poly in polys:
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.5)
    union = shapely.unary_union(polys)
    if union.geom_type == "Polygon":
        x, y = union.exterior.xy

        plt.plot(x, y)
        plt.fill(x, y, alpha=0.3)
    elif union.geom_type == "MultiPolygon":
        for geom in union.geoms:
            x, y = geom.exterior.xy
            plt.plot(x, y)
            plt.fill(x, y, alpha=0.3)
    plt.xlabel(r"$\phi_C$")
    plt.ylabel(r"$\phi_A$")
    # print()
    plt.legend()
    plt.savefig("literaturediscussion.png")
    plt.show()
