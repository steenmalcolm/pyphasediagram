import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SpinodalPoint:
    def __init__(self, idx: int, phi1: float, phi2: float):
        """A point on the spinodal curve with coordinates (phi1, phi2) and an index for graph node identification."""
        self.idx = int(idx)
        self.phi1 = float(phi1)
        self.phi2 = float(phi2)

    def dist(self, pt) -> float:
        """Euclidean distance between this point and another SpinodalPoint."""
        if not isinstance(pt, SpinodalPoint):
            raise NotImplementedError(
                "Distance can only be computed between SpinodalPoint instances."
            )
        return np.sqrt((self.phi1 - pt.phi1) ** 2 + (self.phi2 - pt.phi2) ** 2)

    def __repr__(self):
        return f"SpinodalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f})"

    def __add__(self, pt):
        if not isinstance(pt, SpinodalPoint):
            raise NotImplementedError(
                "Addition can only be performed between SpinodalPoint instances."
            )
        return np.array([self.phi1 + pt.phi1, self.phi2 + pt.phi2])

    def __sub__(self, pt):
        if not isinstance(pt, SpinodalPoint):
            raise NotImplementedError(
                "Subtraction can only be performed between SpinodalPoint instances."
            )
        return np.array([self.phi1 - pt.phi1, self.phi2 - pt.phi2])

    def plot(self, s=10, **kwargs):
        """Convenience method to plot this point."""
        plt.scatter(self.phi1, self.phi2, s=s, **kwargs)

    def is_between(self, pt1, pt2, tol=np.pi / 8):
        """
        Determine if this point is approximately between pt1 and pt2 by checking if the angle between the vectors (self->pt1) and (self->pt2) is close to 180 degrees within a tolerance in units of radians
        """
        if not (isinstance(pt1, SpinodalPoint) and isinstance(pt2, SpinodalPoint)):
            raise NotImplementedError(
                "is_between can only be computed between SpinodalPoint instances."
            )
        self_to_1 = (pt1 - self) / np.linalg.norm(pt1 - self)
        self_to_2 = (pt2 - self) / np.linalg.norm(pt2 - self)
        cos_angle = np.dot(self_to_1, self_to_2)
        return cos_angle < np.cos(np.pi - tol)


class CriticalPoint(SpinodalPoint):
    def __init__(self, idx: int, phi1: float, phi2: float, dphi1: float, dphi2: float):
        """A critical point on the spinodal curve with coordinates (phi1, phi2) and a normalized direction vector (dphi1, dphi2) indicating the direction of the spinodal curve at this point. The direction vector is normalized to have unit length."""
        super().__init__(idx, phi1, phi2)
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        self.dphi1 = float(dphi1) / dphi_norm
        self.dphi2 = float(dphi2) / dphi_norm

    @classmethod
    def from_points(cls, idx: int, pt1: SpinodalPoint, pt2: SpinodalPoint):
        """Create a CriticalPoint from two SpinodalPoints by taking the midpoint of their coordinates as the critical point's coordinates and the vector from pt2 to pt1 as the direction vector. The direction vector is normalized to have unit length."""
        phi1 = (pt1.phi1 + pt2.phi1) / 2
        phi2 = (pt1.phi2 + pt2.phi2) / 2

        dphi1 = pt1.phi1 - pt2.phi1
        dphi2 = pt1.phi2 - pt2.phi2
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        dphi1 = float(dphi1) / dphi_norm
        dphi2 = float(dphi2) / dphi_norm

        return cls(idx, phi1, phi2, dphi1, dphi2)

    def plot(self, s=10, **kwargs):
        """Convenience method to plot this critical point with a different marker and color than regular SpinodalPoints. By default, the marker is a yellow star."""
        if "marker" not in kwargs:
            kwargs["marker"] = "*"
        if "color" not in kwargs:
            kwargs["color"] = "yellow"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        super().plot(s=s, **kwargs)

    def __repr__(self):
        return f"CriticalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f}, dphi1={self.dphi1:.3f}, dphi2={self.dphi2:.3f})"


class Spinodal:

    def __init__(self, chis: np.ndarray):
        """Initialize the Spinodal class with a 2x2 matrix of chi parameters. The spinodal curve will be computed based on these parameters. The graph structure to store the spinodal curve is initialized as an empty NetworkX graph, and a list to store critical points is also initialized."""
        self.chis = chis
        self.spinodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []

    def build(self):
        """
        Build the spinodal curve as a graph with nodes representing points (phi1, phi2) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.spinodal_graph.
        """
        phi1_domains = self._spinodal_domains()
        for i in range(0, len(phi1_domains), 2):
            self._domain_data(phi1_domains[i], phi1_domains[i + 1])
        self._clip_to_domain()
        self._connect_branches()
        self._find_critical_points()

    def _get_p_q(self, phi, is_calculate_phi2=True):
        """
        Compute the coefficients p and q of the quadratic equation for phi1/phi2 given phi2/phi1.
        """
        a, b, c = self.chis[0, 0], self.chis[1, 1], self.chis[0, 1]
        if is_calculate_phi2 == False:
            a, b = b, a  # swap a and b if computing phi1 from phi2
        det = a * b - c**2
        p = (2 * phi * c - a - phi * (1 - phi) * det) / (a + det * phi)
        q = (1 + phi * (1 - phi) * b) / (a + det * phi)
        return p, q

    def phi2_from_phi1(self, phi1):
        """Given phi1, compute the two possible phi2 values from the quadratic formula."""
        p, q = self._get_p_q(phi1)
        discriminant = p**2 / 4 + q

        return -p / 2 + np.sqrt(discriminant), -p / 2 - np.sqrt(discriminant)

    def _domain_data(self, phi1_i, phi1_f, num_points=5000):
        """
        Compute the spinodal branches for phi1 in [phi1_i, phi1_f]
        and add them to the graph as nodes with attributes phi1, phi2, and pos=(phi1, phi2).
        """
        phi1_vals = np.linspace(phi1_i, phi1_f, num_points)
        phi2_branches = self.phi2_from_phi1(phi1_vals)

        # Iterate over both branches
        for phi2_vals in phi2_branches:
            pt_prev: SpinodalPoint = None
            for i, (phi1, phi2) in enumerate(zip(phi1_vals, phi2_vals)):
                pt = SpinodalPoint(self.node_id, phi1, phi2)
                self.spinodal_graph.add_node(pt)
                if i:
                    self.spinodal_graph.add_edge(pt_prev, pt)

                # Copy constructor
                pt_prev = pt
                self.node_id += 1

    def _spinodal_domains(self):
        """
        Find phi values where the discriminant of the quadratic equation for phi2 changes sign.
        These are the phi1 coordinates where the spinodal branches start/end
        """
        phi1_vals = np.linspace(0, 1, 100000)

        p, q = self._get_p_q(phi1_vals)
        discriminant = p**2 + 4 * q

        # Edge case where pole of p and q is resolved
        for pole_idx in np.where(np.isinf(p) | np.isinf(q))[0]:
            discriminant[pole_idx] = discriminant[pole_idx - 1]

        roots_idx = np.where(np.diff(np.sign(discriminant)))[0]

        # Check if discriminant is positive at the endpoints and add them to the roots if so, since the spinodal branches can start/end at the domain boundaries
        if discriminant[0] > 0:
            roots_idx = np.r_[0, roots_idx]
        if discriminant[-1] > 0:
            roots_idx = np.r_[roots_idx, -1]
        # Add 1 at beginning of domain so the discriminant is positive
        roots_idx[::2] += 1
        return phi1_vals[roots_idx]

    def _clip_to_domain(self):
        """
        Remove nodes whose (phi1, phi2) lie outside the correct domain
        Mutates and returns G for convenience.
        """
        if self.spinodal_graph.number_of_nodes() == 0:
            return
        to_remove = [
            n
            for n in self.spinodal_graph.nodes()
            if (n.phi1 < 0) or (n.phi2 < 0) or (n.phi1 + n.phi2 > 1)
        ]
        self.spinodal_graph.remove_nodes_from(to_remove)
        # Edge case remove subgraphs with single point
        comps = list(nx.connected_components(self.spinodal_graph))
        for comp in comps:
            if len(comp) == 1:
                self.spinodal_graph.remove_node(next(iter(comp)))

    def _connect_branches(self):
        """Connect disjoint branches of the spinodal curve by adding edges between closest endpoints (degree 1 nodes) of different components if they are within a certain distance threshold"""
        # TODO: Check if minimum bounding squares of subgraphs overlap before computing pairwise distances to speed up for large graphs
        if self.spinodal_graph.number_of_nodes() == 0:
            return
        comps = list(nx.connected_components(self.spinodal_graph))
        for compi in comps:
            sub_graphi = self.spinodal_graph.subgraph(compi)
            endpoints_i = [
                pt for pt in sub_graphi.nodes() if sub_graphi.degree[pt] == 1
            ]
            closest_pt_list = [0] * len(endpoints_i)
            closest_dist_list = [float("inf")] * len(endpoints_i)

            for compj in comps:
                sub_graphj = self.spinodal_graph.subgraph(compj)
                endpoints_j = [
                    pt for pt in sub_graphj.nodes() if sub_graphj.degree[pt] == 1
                ]
                for epi_idx, epi in enumerate(endpoints_i):
                    for epj in endpoints_j:
                        dist = epi.dist(epj)
                        if dist < closest_dist_list[epi_idx] and epi != epj:
                            # Check edge case epi and epj are only two points on spinodal branch
                            nb_epi = next(self.spinodal_graph.neighbors(epi))
                            if nb_epi == epj:
                                continue
                            closest_dist_list[epi_idx] = dist
                            closest_pt_list[epi_idx] = epj

            for epi_idx, epi in enumerate(endpoints_i):
                if closest_dist_list[epi_idx] < 0.05:
                    epj = closest_pt_list[epi_idx]
                    # Make sure the two branches are pointing towards each other
                    nb_epi = next(self.spinodal_graph.neighbors(epi))
                    nb_epj = next(self.spinodal_graph.neighbors(epj))
                    if epj.is_between(nb_epj, epi) and epi.is_between(nb_epi, epj):
                        self.spinodal_graph.add_edge(epi, epj)
                        # Call recursively because graph structure has changed
                        return self._connect_branches()

    def _coords_from_subgraph(self, sg):
        """
        Extract phi1 and phi2 coordinates from a subgraph's nodes. In the correct order.
        Returns two lists: phi1s and phi2s.
        """
        # Find endpoint (degree 1) if it exists (path case)
        endpoints = [n for n, d in sg.degree() if d == 1]

        if endpoints:
            start = endpoints[0]
        else:
            # cycle case (all degree 2)
            start = next(iter(sg.nodes()))

        ordered_points = list(nx.dfs_preorder_nodes(sg, source=start))

        phi1s = np.array([n.phi1 for n in ordered_points])
        phi2s = np.array([n.phi2 for n in ordered_points])
        return phi1s, phi2s

    def _third_derivative(self, phi1, phi2):
        phi0 = 1 - phi1 - phi2
        H_11 = 1 / phi1 + 1 / phi0 + self.chis[0, 0]
        H_12 = 1 / phi0 + self.chis[0, 1]
        return H_12**3 / (phi1**2) - H_11**3 / (phi2**2) + (H_11 - H_12) ** 3 / phi0**2

    def _find_critical_points(self):
        for comp in nx.connected_components(self.spinodal_graph):

            sg = self.spinodal_graph.subgraph(comp)
            phi1, phi2 = self._coords_from_subgraph(sg)

            third_deriv = self._third_derivative(phi1, phi2)
            if np.isnan(third_deriv).any():
                raise ValueError(
                    "NaN values found in third derivative, check for invalid phi1/phi2 values."
                )
            # Edge case where third derivative has extrema at root
            extrema_idxs = np.where(np.diff(np.sign(np.diff(third_deriv))))[0] + 1
            for extrema_idx in extrema_idxs:
                if abs(third_deriv[extrema_idx]) < 1e-4:
                    phi1_c, phi2_c = phi1[extrema_idx], phi2[extrema_idx]
                    dphi1 = phi1[extrema_idx + 1] - phi1[extrema_idx - 1]
                    dphi2 = phi2[extrema_idx + 1] - phi2[extrema_idx - 1]
                    self.critical_points.append(
                        CriticalPoint(-1, phi1_c, phi2_c, dphi1, dphi2)
                    )

            # Find roots of third derivative
            root_idxs = np.where(np.diff(np.sign(third_deriv)))[0]

            for r_idx in root_idxs:
                dphi1 = phi1[r_idx + 1] - phi1[r_idx - 1]
                dphi2 = phi2[r_idx + 1] - phi2[r_idx - 1]
                # For np.interp to work, we need to ensure the third_deriv values at r_idx and r_idx+1 are in increasing order. If not, swap them.
                xp = third_deriv[r_idx : r_idx + 2]
                phi1p, phi2p = phi1[r_idx : r_idx + 2], phi2[r_idx : r_idx + 2]
                order = np.argsort(third_deriv[r_idx : r_idx + 2])
                phi1_c = np.interp(0, xp[order], phi1p[order])
                phi2_c = np.interp(0, xp[order], phi2p[order])
                td_c = self._third_derivative(phi1_c, phi2_c)
                if td_c > 1e-2:
                    Warning(
                        f"Third derivative at critical point ({phi1_c:.2f}, {phi2_c:.2f}) is quite large: {td_c:.5f}.\nTry increasing resolution of spinodal curve."
                    )
                assert abs(td_c) < np.std(
                    xp
                ), f"Third derivative at critical point ({phi1_c:.2f}, {phi2_c:.2f}) is not close to zero: {td_c:.5f}\n(chi_11, chi_22, chi_12)=({self.chis[0, 0]}, {self.chis[1, 1]}, {self.chis[0, 1]})\nTry increasing resolution of spinodal curve."
                self.critical_points.append(
                    CriticalPoint(-1, phi1_c, phi2_c, dphi1, dphi2)
                )

    def plot(self):
        a, b, c = self.chis[0, 0], self.chis[1, 1], self.chis[0, 1]
        plt.figure()
        plt.title(
            r"$(\chi_{11}, \chi_{22}, \chi_{12})$ = " + f"({a:.5f}, {b:.5f}, {c:.5f})"
        )
        for i, comp in enumerate(nx.connected_components(self.spinodal_graph)):
            sg = self.spinodal_graph.subgraph(comp)

            # Find endpoint (degree 1) if it exists (path case)
            phi1s, phi2s = self._coords_from_subgraph(sg)
            if np.isnan(phi2s).any():
                print("NaN values found in ys, skipping plot for this component.")
            if i == 0:
                plt.plot(phi1s, phi2s, label="Spinodal curve")
            else:
                plt.plot(phi1s, phi2s)
            # plt.scatter(phi1s[0], phi2s[0], alpha=0.45)
            # plt.scatter(phi1s[-1], phi2s[-1], alpha=0.45)
        for i, crit_pt in enumerate(self.critical_points):
            if i == 0:
                crit_pt.plot(s=50, label="Critical point(s)", zorder=5)
            else:
                crit_pt.plot(s=50, zorder=5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot([0, 1], [1, 0], "k--")

        plt.xlabel(r"$\phi_1$")
        plt.ylabel(r"$\phi_2$")
        plt.legend()


if __name__ == "__main__":
    import signal
    from tqdm import tqdm

    chi_dr, chi_rs, chi_ds = 3.1, 3.1, 3.1
    chi_11, chi_22, chi_12 = -2 * chi_ds, -2 * chi_rs, chi_dr - chi_rs - chi_ds
    chis = np.array([[chi_11, chi_12], [chi_12, chi_22]])
    spinodal = Spinodal(chis)
    spinodal.build()  # Monitor this method call
    spinodal.plot()

    # Define a timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Method call exceeded 10 seconds.")

    # Register the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)

    for i in tqdm(range(1000), desc="Generating spinodal curves"):
        try:
            # Set an alarm for 10 seconds
            signal.alarm(10)

            chi_dr, chi_rs, chi_ds = np.random.random(3) + 2
            chi_11, chi_22, chi_12 = -2 * chi_ds, -2 * chi_rs, chi_dr - chi_rs - chi_ds
            chis = np.array([[chi_11, chi_12], [chi_12, chi_22]])
            spinodal = Spinodal(chis)
            spinodal.build()  # Monitor this method call
            spinodal.plot()
            plt.savefig(f"delete/{i}.png")
            plt.close()

            # Cancel the alarm if the method completes in time
            signal.alarm(0)
        except TimeoutError:
            print(
                f"Iteration {i}: Method call timed out.\n(chi_dr, chi_rs, chi_ds)=({chi_dr}, {chi_rs}, {chi_ds})"
            )
