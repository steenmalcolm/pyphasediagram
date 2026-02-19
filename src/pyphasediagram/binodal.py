import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import shapely
from pyphasediagram.point import CriticalPoint, BinodalPoint
from pyphasediagram.stepper import TernaryStepper


class BinodalBranch:
    def __init__(self, phis: np.ndarray, svs: np.ndarray):
        self.phis = phis
        self.svs = svs
        self.line_a = shapely.LineString(phis[0].T)
        self.line_b = shapely.LineString(phis[1].T)

    def plot(self, **kwargs):
        kwargs["color"] = kwargs.pop("color_a", "blue")
        plt.plot(self.phis[0, 0], self.phis[0, 1], **kwargs)

        kwargs["color"] = kwargs.pop("color_b", "red")
        plt.plot(self.phis[1, 0], self.phis[1, 1], **kwargs)

    def intersects_with(self, other):
        """Check if this binodal branch intersects with another binodal branch by checking if their corresponding lines intersect using shapely."""
        if not isinstance(other, BinodalBranch):
            raise NotImplementedError(
                "intersects_with can only be computed between BinodalBranch instances."
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

    def contained_in(self, other):
        """Check if this binodal branch is contained within another binodal branch by"""
        if not isinstance(other, BinodalBranch):
            raise NotImplementedError(
                "contained_in can only be computed between BinodalBranch instances."
            )

        # System invariant under swap of phase labeling
        for la, lb in [(self.line_a, self.line_b), (self.line_b, self.line_a)]:
            ia = la.intersection(other.line_a)
            ib = lb.intersection(other.line_b)
            if ia.geom_type == "MultiPoint" and ib.geom_type == "MultiPoint":
                if (
                    len(ia.geoms) > len(la.coords) / 2
                    and len(ib.geoms) > len(lb.coords) / 2
                ):
                    return True
        return False

    def _degenerate_points(self) -> list[BinodalPoint]:
        """Identify points where the null space of the Jacobian has dimension greater than 1, which indicates branching"""
        branching_points = []
        extrema_idx = np.where(np.diff(np.sign(np.diff(self.svs))) == 2)[0] + 1
        for idx in extrema_idx:
            if self.svs[idx] < 1e-4:
                branching_points.append(
                    BinodalPoint(
                        *self.phis[0, :, idx], *self.phis[1, :, idx], self.svs[idx]
                    )
                )
        return branching_points


class Binodal:

    def __init__(self, chis: np.ndarray, critical_points: list[CriticalPoint] = []):
        """Initialize the Binodal class with a 2x2 matrix of chi parameters"""
        self.chis = chis
        self.critical_points = critical_points
        self.binodal_branches: list[BinodalBranch] = []
        self._tracer = TernaryStepper(chis)
        self._phi_init_hist: list[BinodalPoint] = []

    def _build_branch(self, phi_init, v_init):
        """
        Build a branch of the binodal curve starting from an initial composition phi_init and initial step v_init. This method runs the tracer to compute the binodal points along the branch and adds them as nodes in the binodal graph, connecting consecutive points with edges.
        """
        # Run the tracer to compute the binodal points along the branch
        phis, svs = self._tracer.run(phi_init, v_init)
        bb_new = BinodalBranch(phis, svs)

        # Avoid duplicates
        for bb in self.binodal_branches:
            if bb_new.contained_in(bb):
                return False

        # Save the new branch and its initial point
        self.binodal_branches.append(BinodalBranch(phis, svs))
        self._phi_init_hist.append(BinodalPoint(*phi_init[:2], *phi_init[2:], svs[0]))
        return True

    def build(self):
        """
        Build the binodal curve as a graph with nodes representing points (phi1a, phi2a, phi1b, phi2b) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.binodal_graph.
        """
        # Branches from binary limits
        for which_comp in range(3):
            phi_init, v_init = self._tracer.binary_init(which_comp)
            if phi_init == None:
                continue
            self._build_branch(phi_init, v_init)

        # Branches from critical points
        for cp in self.critical_points:
            phi_init, v_init = cp.get_phi_and_v_init()
            self._build_branch(phi_init, v_init)

        # TODO: Branches from degenerate points
        # self._find_degenerate_points()

    # def _point_has_been_explored(self, bpt: BinodalPoint, tol=1e-3):
    #     """Check if a given binodal point has already been explored by checking if its composition is close to any of the compositions in self._phi_init_hist."""
    #     for hist_pt in self._phi_init_hist:
    #         if hist_pt.is_similar_to(bpt, tol):
    #             return True
    #     return False

    # def _find_degenerate_points(self):
    #     """Identify points where the null space of the Jacobian has dimension greater than 1, which indicates branching"""
    #     branching_points:list[BinodalPoint] = []
    #     for bb in self.binodal_branches:
    #         for bp in bb._degenerate_points():
    #             if


if __name__ == "__main__":
    from pyphasediagram.spinodal import Spinodal

    chi_12, chi_01, chi_02 = np.random.random(3) + 2
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
    print(f"Found {len(obj.binodal_branches)} binodal branches")

    sp_obj.plot()
    for i, branch in enumerate(obj.binodal_branches):
        if i == 0:
            branch.plot(label=f"Binodal")
        else:
            branch.plot()
    plt.legend()
    plt.show()
