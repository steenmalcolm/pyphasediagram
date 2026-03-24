from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely

from pyphasediagram.point import CriticalPoint, SpinodalPoint

# TODO: Throw error if polygons don't match number of connected components. This happens for chi=2.667


class Spinodal:

    DOMAIN_CORNERS = [(0, 0), (0, 1), (1, 0)]

    def __init__(self, chis: np.ndarray):
        """Initialize the Spinodal class with a 2x2 matrix of chi parameters. The spinodal curve will be computed based on these parameters. The graph structure to store the spinodal curve is initialized as an empty NetworkX graph, and a list to store critical points is also initialized."""
        self.chis = chis
        self.spinodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []
        self.polygons: list[shapely.geometry.Polygon] = None

    def build(self, num_points=10000):
        """
        Build the spinodal curve as a graph with nodes representing points (phi1, phi2) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.spinodal_graph.
        """
        phi1_domains = self._spinodal_domains()
        for i in range(0, len(phi1_domains), 2):
            self._domain_data(phi1_domains[i], phi1_domains[i + 1], num_points)
        self._clip_to_domain()
        self._connect_branches()
        self._find_critical_points()
        self._build_polygons()

    def _get_p_q(self, phi, is_calculate_phi2=True):
        """
        Compute the coefficients p and q of the quadratic equation for phi1/phi2 given phi2/phi1.
        """
        a, b, c = self.chis[1, 1], self.chis[0, 0], self.chis[0, 1]
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

    def eigenvalues_from_phi(self, phi1, phi2):
        """Given phi1 and phi2, compute the eigenvalues of the Hessian matrix of the free energy."""
        phi0 = 1 - phi1 - phi2
        H_11 = 1 / phi1 + 1 / phi0 + self.chis[0, 0]
        H_22 = 1 / phi2 + 1 / phi0 + self.chis[1, 1]
        H_12 = 1 / phi0 + self.chis[0, 1]

        trace = H_11 + H_22
        det = H_11 * H_22 - H_12**2

        eigenvalue1 = trace / 2 + np.sqrt(trace**2 / 4 - det)
        eigenvalue2 = trace / 2 - np.sqrt(trace**2 / 4 - det)

        return eigenvalue1, eigenvalue2

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

    @classmethod
    def _in_domain(cls, phi1, phi2):
        """Check if the points (phi1, phi2) are inside the open triangular domain defined by DOMAIN_CORNERS.
        Returns a boolean array (or scalar) that is True where (phi1 > 0, phi2 > 0, phi1 + phi2 < 1).
        """
        phi1 = np.asarray(phi1)
        phi2 = np.asarray(phi2)
        return (phi1 > 0) & (phi2 > 0) & (phi1 + phi2 < 1)

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
            if not self._in_domain(n.phi1, n.phi2)
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
        n = len(comps)

        # Endpoints have degree 1
        endpoints = [
            pt
            for pt in self.spinodal_graph.nodes()
            if self.spinodal_graph.degree[pt] == 1
        ]
        # Keep track of endpoints we've already connected to avoid adding multiple edges from the same endpoint in cases where more than 2 branches are close together
        invalid_endpoints = []

        for i, epti in enumerate(endpoints):
            if epti in invalid_endpoints:
                continue
            for eptj in endpoints[i + 1 :]:
                if eptj in invalid_endpoints:
                    continue
                dist = epti.dist(eptj)
                # Note this is a greedy algorithm, might need elaborate testing to make sure it works
                if dist < 0.05:
                    nb_epti = next(self.spinodal_graph.neighbors(epti))
                    nb_eptj = next(self.spinodal_graph.neighbors(eptj))
                    if eptj.is_between(nb_eptj, epti) and epti.is_between(
                        nb_epti, eptj
                    ):
                        self.spinodal_graph.add_edge(epti, eptj)
                        invalid_endpoints.extend([epti, eptj])

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
            mask = self._in_domain(phi1, phi2)
            phi1, phi2 = phi1[mask], phi2[mask]

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
                dphi1 = phi1[r_idx + 1] - phi1[r_idx]
                dphi2 = phi2[r_idx + 1] - phi2[r_idx]
                # For np.interp to work, we need to ensure the third_deriv values at r_idx and r_idx+1 are in increasing order. If not, swap them.
                xp = third_deriv[r_idx : r_idx + 2]
                phi1p, phi2p = phi1[r_idx : r_idx + 2], phi2[r_idx : r_idx + 2]
                o = np.argsort(third_deriv[r_idx : r_idx + 2])
                phi1_c = np.interp(0, xp[o], phi1p[o])
                phi2_c = np.interp(0, xp[o], phi2p[o])
                td_c = self._third_derivative(phi1_c, phi2_c)
                if td_c > 1e-2:
                    Warning(
                        f"Third derivative at critical point ({phi1_c:.2f}, {phi2_c:.2f}) is quite large: {td_c:.5f}.\nTry increasing resolution of spinodal curve."
                    )
                if abs(td_c) >= np.std(xp):
                    raise RuntimeError(
                        f"Third derivative at critical point ({phi1_c:.2f}, {phi2_c:.2f}) is not close to zero: {td_c:.5f}\n"
                        f"(chi_11, chi_22, chi_12)=({self.chis[0, 0]}, {self.chis[1, 1]}, {self.chis[0, 1]})\n"
                        f"Try increasing resolution of spinodal curve."
                    )
                self.critical_points.append(
                    CriticalPoint(-1, phi1_c, phi2_c, dphi1, dphi2)
                )

    def _build_polygons(self):
        """Build shapely polygons for the spinodal curve components to facilitate point-in-spinodal checks"""
        self.polygons = []

        # If needed, store which corner to add to which endpoint

        for comp_idx, comp in enumerate(nx.connected_components(self.spinodal_graph)):
            sg = self.spinodal_graph.subgraph(comp).copy()

            # Add boundary point if endpoint is close to boundary, such that polygons don't cut through unstable region.
            endpoints = [n for n, d in sg.degree() if d == 1]
            if len(endpoints) == 2:
                for e in endpoints:
                    boundary_point = None
                    if e.phi1 < 1e-2:
                        boundary_point = SpinodalPoint(-1, 0, e.phi2)
                    elif e.phi2 < 1e-2:
                        boundary_point = SpinodalPoint(-1, e.phi1, 0)
                    elif e.phi1 + e.phi2 > 1 - 1e-2:
                        boundary_point = SpinodalPoint(-1, e.phi1, 1 - e.phi1)
                    else:
                        # Edge case where displacement between e and its neighbor is vertical, such that the resolution is quite bad
                        nb = next(sg.neighbors(e))
                        d = e - nb
                        if (
                            abs(np.arctan(d[0] / d[1])) < 0.01
                        ):  # if angle with vertical is less than 0.01 radians (~0.57 degrees)
                            if d[1] < 0:

                                boundary_point = SpinodalPoint(-1, e.phi1, 0)
                            else:
                                boundary_point = SpinodalPoint(-1, e.phi1, 1 - e.phi1)
                    if boundary_point is not None:
                        sg.add_node(boundary_point)
                        sg.add_edge(e, boundary_point)
            endpoints = [n for n, d in sg.degree() if d == 1]

            is_add_corner = [0, 0]
            if len(endpoints) == 2:
                for i, e in enumerate(endpoints):
                    if e.phi1 < 1e-2:
                        is_add_corner[i] += (1 << 0) + (
                            1 << 1
                        )  # add corners (0,0) and (0,1)
                    if e.phi2 < 1e-2:
                        is_add_corner[i] += (1 << 0) + (
                            1 << 2
                        )  # add corners (0,0) and (1,0)
                    if e.phi1 + e.phi2 > 1 - 1e-2:
                        is_add_corner[i] += (1 << 1) + (
                            1 << 2
                        )  # add corners (0,1) and (1,0)
                # Remove the corner which both endpoints have in common
                # This guarantees that if both endpoints are on the same edge, we don't add any corners
                # and if they are on different edges we add the correct corners such that the polygon encloses the locally stable area of the domain
                common_corners = is_add_corner[0] & is_add_corner[1]
                if is_add_corner[0] ^ is_add_corner[1] > 0:
                    corner = self.DOMAIN_CORNERS[common_corners.bit_length() - 1]
                    sg.add_node(SpinodalPoint(-1, corner[0], corner[1]))
                    sg.add_edge(endpoints[0], SpinodalPoint(-1, corner[0], corner[1]))

            phi1, phi2 = self._coords_from_subgraph(sg)
            coords = np.column_stack((phi1, phi2))
            if len(coords) >= 3:  # Need at least 3 points to form a polygon

                poly = shapely.make_valid(shapely.geometry.Polygon(coords))
                if poly.is_valid:
                    center = poly.centroid
                    ev1, ev2 = self.eigenvalues_from_phi(center.x, center.y)
                    if ev1 < 0 and ev2 < 0:
                        continue

                    if ev1 * ev2 < 0:
                        poly = shapely.geometry.Polygon(self.DOMAIN_CORNERS).difference(
                            poly
                        )
                    for other in self.polygons:
                        if poly.intersects(other):
                            # Edge case where overlap is a line or point
                            if poly.intersection(other).area < 1e-10:
                                continue
                            poly = poly.intersection(other)
                            # Remove the other polygon from the list since it's now merged with the current one
                            self.polygons.remove(other)
                            break
                    self.polygons.append(poly)
                else:
                    print(
                        "Warning: Invalid polygon formed from spinodal component, skipping polygon creation for this component."
                    )
            else:
                print(
                    "Warning: Not enough points to form a polygon for this spinodal component, skipping polygon creation for this component."
                )

    def get_unstable_manifold(self) -> Optional[shapely.geometry.Polygon]:
        """Return a shapely polygon representing the locally unstable region of the phase diagram, which is the union of the polygons formed by the spinodal curve components."""
        if self.polygons is None:
            raise ValueError(
                "Polygons have not been built yet. Call _build_polygons() first."
            )
        if len(self.polygons) == 0:
            return None
        polygon = shapely.geometry.Polygon(self.DOMAIN_CORNERS)
        for poly in self.polygons:
            polygon = shapely.make_valid(polygon.difference(poly))
        return polygon

    def plot(self, **kwargs):
        a, b, c = self.chis[0, 0], self.chis[1, 1], self.chis[0, 1]
        plt.title(
            r"$(\chi_{11}, \chi_{22}, \chi_{12})$ = " + f"({a:.5f}, {b:.5f}, {c:.5f})"
        )
        ls = kwargs.get("linestyle", "--")
        for i, comp in enumerate(nx.connected_components(self.spinodal_graph)):
            sg = self.spinodal_graph.subgraph(comp)

            # Find endpoint (degree 1) if it exists (path case)
            phi1s, phi2s = self._coords_from_subgraph(sg)
            if np.isnan(phi2s).any():
                print("NaN values found in ys, skipping plot for this component.")
            if i == 0:
                plt.plot(phi1s, phi2s, label="Spinodal curve", linestyle=ls)
            else:
                plt.plot(phi1s, phi2s, linestyle=ls)

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
    import time

    from shapely.plotting import plot_polygon

    chi = 2.6667

    i = 0
    while i < 100:
        n = time.perf_counter()
        chi_dr, chi_rs, chi_ds = np.random.uniform(-10, 10, size=3)
        chi_11, chi_22, chi_12 = -2 * chi_ds, -2 * chi_rs, chi_dr - chi_rs - chi_ds
        chis = np.array([[chi_11, chi_12], [chi_12, chi_22]])
        spinodal = Spinodal(chis)
        spinodal.build()  # Monitor this method call
        u_poly = spinodal.get_unstable_manifold()
        spinodal.plot()
        if u_poly is not None:
            plot_polygon(u_poly)
        else:
            plt.close()
            continue
        plt.savefig(f"delete/{i}.png")
        plt.close()
        print(
            f"Completed iteration {i+1}/100 in {time.perf_counter() - n:.2f} seconds."
        )
        i += 1
