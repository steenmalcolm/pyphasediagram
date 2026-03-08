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

    def plot(self, colora="red", colorb="blue", **kwargs):
        plt.plot(self.phis[0, 0], self.phis[0, 1], color=colora, **kwargs)

        plt.plot(self.phis[1, 0], self.phis[1, 1], color=colorb, **kwargs)

    def intersects_with(self, other):
        """Check if this binodal section intersects with another binodal section by checking if their corresponding lines intersect using shapely."""
        if not isinstance(other, BinodalSection):
            raise NotImplementedError(
                "intersects_with can only be computed between BinodalSection instances."
            )
        i_points = []

        def add_intersection(l1, l2):
            i = l1.intersection(l2)
            if i.geom_type == "Point":
                i_points.append(i)

        add_intersection(self.line_a, other.line_a)
        add_intersection(self.line_a, other.line_b)
        add_intersection(self.line_b, other.line_a)
        add_intersection(self.line_b, other.line_b)

        return i_points

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
        self._tracer = Stepper(chis)
        self._bipt_hist: list[BinodalInitialPoint] = []

    def _build_section(self, phi_init, v_init):
        """
        Build a section of the binodal curve starting from an initial composition phi_init and initial step v_init. This method runs the tracer to compute the binodal points along the section and adds them as nodes in the binodal graph, connecting consecutive points with edges.
        """
        # Run the tracer to compute the binodal points along the section
        phis, svs, flags = self._tracer.run(phi_init, v_init)
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
            phi_init, v_init = self._tracer.binary_init(which_comp)
            if isinstance(phi_init, np.ndarray):
                self._build_section(phi_init, v_init)

        # Sections from critical points
        for cpt in self.critical_points:
            phi_init, v_init = cpt.get_phi_and_v_init()
            self._build_section(phi_init, v_init)

        # TODO: Keep iterating until all Branching points have been explored, but for now just do one pass
        for bpt in self._find_branching_points():

            # COmmentar!
            if any([bpt.is_similar_to(bipt) for bipt in self._bipt_hist]):
                continue

            bpt.phi_init += bpt.v_init * 1e-3
            self._build_section(
                *bpt,
            )

    def _find_branching_points(self):
        """Identify points where the null space of the Jacobian has dimension greater than 1, which indicates branching"""
        branching_points: list[BinodalInitialPoint] = []
        for bb in self.binodal_sections:
            for bp in bb._degenerate_points(self.SV_BRANCH_THRESHOLD):
                J = self._tracer._jac_fn(bp.phi_init)
                U, S, Vt = self._tracer._svd(J, full_matrices=True)
                if (
                    np.dot(Vt[-1], bp.v_init) > 0.98
                    and S[-1] < self.SV_BRANCH_THRESHOLD
                ):
                    bp.v_init = np.asarray(Vt[-2])
                    branching_points.append(bp)

        return branching_points

    def plot(self, **kwargs):
        for section in self.binodal_section:
            section.plot(**kwargs)


if __name__ == "__main__":
    from pyphasediagram.spinodal import Spinodal
    import tqdm

    chi_12, chi_01, chi_02 = (
        0.8,
        1.1,
        2.2,
    )
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
    # print("\t", end=" ")
    for i, section in enumerate(obj.binodal_sections):
        if i == 0:
            section.plot(label=f"Binodal")
        else:
            section.plot()
        # print(f"{len(branch)}", end=", ")
    plt.xlabel(r"$\phi_C$")
    plt.ylabel(r"$\phi_A$")
    # print()
    plt.legend()
    plt.savefig("literaturediscussion.png")
    plt.show()
