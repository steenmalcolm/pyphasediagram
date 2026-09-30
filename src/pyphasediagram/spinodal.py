"""Construct and analyze spinodal curves for ternary mixtures.

The spinodal is represented as an undirected graph of points in the two
independent composition coordinates ``(phi1, phi2)``. This module samples the
analytic spinodal branches, connects and clips them to the physically admissible domain,
locates critical points, and constructs polygons used to identify the
locally unstable region.
"""

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely

from pyphasediagram.point import CriticalPoint, SpinodalPoint

# TODO: Throw error if polygons don't match number of connected components. This happens for chi=2.667


class Spinodal:
    """Represent the spinodal curve of a ternary mixture.

    Parameters
    ----------
    chis : numpy.ndarray
        Reduced interaction matrix with shape ``(2, 2)`` for the two
        independent composition coordinates.

    Attributes
    ----------
    DOMAIN_CORNERS : list of tuple
        Vertices ``[(0, 0), (0, 1), (1, 0)]`` of the closed composition
        simplex.
    chis : numpy.ndarray
        Reduced interaction matrix with shape ``(2, 2)``.
    spinodal_graph : networkx.Graph
        Graph whose nodes are
        :class:`~pyphasediagram.point.SpinodalPoint` objects and whose edges
        follow sampled spinodal branches.
    node_id : int
        Used to assign a unique ID to graph nodes.
    critical_points : list of CriticalPoint
        Critical points detected on the spinodal branches.
    polygons : list of shapely.geometry.Polygon or None
        Locally stable regions bounded by the spinodal curves. ``None`` until
        polygon construction has run.

    Raises
    ------
    ValueError
        If ``chis`` does not have shape ``(2, 2)``.

    Notes
    -----
    Physical compositions lie in the open simplex ``phi1 > 0``, ``phi2 > 0``,
    and ``phi1 + phi2 < 1``. The dependent composition is
    ``phi0 = 1 - phi1 - phi2``.
    """

    DOMAIN_CORNERS = [(0, 0), (0, 1), (1, 0)]

    def __init__(self, chis: np.ndarray):
        if chis.shape != (2, 2):
            raise ValueError(
                f"Expected chis to be a 2x2 matrix, but got shape {chis.shape}"
            )
        self.chis = chis
        self.spinodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []
        self.polygons: list[shapely.geometry.Polygon] = None

    def build(self, num_points=10000):
        """Build the spinodal graph and derived geometric objects.

        Parameters
        ----------
        num_points : int, optional
            Number of ``phi1`` samples used for each discriminant-positive
            interval and for each of its two analytic branches.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If critical-point detection encounters a ``NaN`` third
            derivative.
        RuntimeError
            If interpolation does not locate a sufficiently accurate root of
            the third derivative.

        Notes
        -----
        The build pipeline samples the analytic branches, clips them to the
        open composition simplex, connects nearby continuations, locates
        critical points, and constructs locally stable polygons. Existing
        graph nodes and critical points are not cleared, so a ``Spinodal``
        instance should normally be built only once.
        """
        phi1_domains = self._spinodal_domains()
        for i in range(0, len(phi1_domains), 2):
            self._domain_data(phi1_domains[i], phi1_domains[i + 1], num_points)
        self._clip_to_domain()
        self._connect_branches()
        self._find_critical_points()
        self._build_polygons()

    def _get_p_q(self, phi, is_calculate_phi2=True):
        """Calculate coefficients of the quadratic spinodal equation.

        Parameters
        ----------
        phi : float or array-like
            Known independent composition coordinate.
        is_calculate_phi2 : bool, optional
            If ``True``, calculate coefficients for ``phi2`` given ``phi1``.
            If ``False``, calculate coefficients for ``phi1`` given ``phi2``.

        Returns
        -------
        p : float or numpy.ndarray
            Linear coefficient of ``x**2 + p*x - q = 0``.
        q : float or numpy.ndarray
            Negated constant coefficient of ``x**2 + p*x - q = 0``.

        Notes
        -----
        Outputs follow the scalar or broadcast array shape of ``phi``. Poles
        can occur where the common denominator of the coefficients vanishes.
        """
        a, b, c = self.chis[1, 1], self.chis[0, 0], self.chis[0, 1]
        if is_calculate_phi2 == False:
            a, b = b, a  # swap a and b if computing phi1 from phi2
        det = a * b - c**2
        p = (2 * phi * c - a - phi * (1 - phi) * det) / (a + det * phi)
        q = (1 + phi * (1 - phi) * b) / (a + det * phi)
        return p, q

    def get_spinodal_coords(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return ordered coordinates for every connected spinodal branch.

        Returns
        -------
        phi1s : list of numpy.ndarray
            First composition coordinates, one array per connected component.
        phi2s : list of numpy.ndarray
            Second composition coordinates corresponding elementwise to
            ``phi1s``.

        Notes
        -----
        Coordinates outside the open composition simplex are excluded. Both
        lists are empty when the graph has no components.
        """
        phi1s, phi2s = [], []
        for comp in nx.connected_components(self.spinodal_graph):
            sg = self.spinodal_graph.subgraph(comp)
            phi1, phi2 = self._coords_from_subgraph(sg)
            mask = self._in_domain(phi1, phi2)
            phi1s.append(phi1[mask])
            phi2s.append(phi2[mask])
        return phi1s, phi2s

    def phi2_from_phi1(self, phi1):
        """Calculate both analytic spinodal branches at specified ``phi1``.

        Parameters
        ----------
        phi1 : float or array-like
            First independent composition coordinate.

        Returns
        -------
        phi2_upper : float or numpy.ndarray
            Solution obtained with the positive square-root branch.
        phi2_lower : float or numpy.ndarray
            Solution obtained with the negative square-root branch.

        Notes
        -----
        Output shapes follow ``phi1``. Inputs for which the quadratic
        discriminant is negative produce ``NaN`` values.
        """
        p, q = self._get_p_q(phi1)
        discriminant = p**2 / 4 + q

        return -p / 2 + np.sqrt(discriminant), -p / 2 - np.sqrt(discriminant)

    def eigenvalues_from_phi(self, phi1, phi2):
        """Calculate the free-energy Hessian eigenvalues at compositions.

        Parameters
        ----------
        phi1 : float or array-like
            First independent composition coordinate.
        phi2 : float or array-like
            Second independent composition coordinate, broadcast-compatible
            with ``phi1``.

        Returns
        -------
        eigenvalue_max : float or numpy.ndarray
            Larger eigenvalue of the ``2 x 2`` free-energy Hessian.
        eigenvalue_min : float or numpy.ndarray
            Smaller eigenvalue of the ``2 x 2`` free-energy Hessian.

        Notes
        -----
        The calculation assumes an interior composition with positive
        ``phi1``, ``phi2``, and ``phi0 = 1 - phi1 - phi2``. Boundary values
        lead to divisions by zero.
        """
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
        """Sample two spinodal branches over a ``phi1`` interval.

        Parameters
        ----------
        phi1_i : float
            Inclusive lower endpoint of the sampling interval.
        phi1_f : float
            Inclusive upper endpoint of the sampling interval.
        num_points : int, optional
            Number of samples on each analytic branch.

        Returns
        -------
        None

        Notes
        -----
        Two path components are appended to :attr:`spinodal_graph`, including
        points that may subsequently be removed by :meth:`_clip_to_domain`.
        Node identifiers are assigned sequentially through :attr:`node_id`.
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
        """Find intervals with real analytic spinodal solutions.

        Returns
        -------
        numpy.ndarray
            Ordered ``phi1`` endpoints with shape ``(2*k,)``. Consecutive
            pairs delimit intervals where the quadratic discriminant is
            nonnegative.

        Notes
        -----
        The discriminant is sampled at 100,000 evenly spaced points on
        ``[0, 1]``. Sign changes are therefore located to the resolution of
        that grid rather than by a continuous root solver. Poles in the
        quadratic coefficients are replaced locally with a neighboring value.
        """
        phi1_vals = np.linspace(0, 1, 100000)

        p, q = self._get_p_q(phi1_vals)
        discriminant = p**2 + 4 * q

        # Edge case where pole of p and q is resolved
        for pole_idx in np.where(np.isinf(p) | np.isinf(q))[0]:
            if pole_idx == 0:
                discriminant[pole_idx] = discriminant[pole_idx + 1]
            else:
                discriminant[pole_idx] = discriminant[pole_idx - 1]

        roots_idx = np.where(np.diff(np.sign(discriminant)))[0]

        # Check if discriminant is positive at the endpoints and add them to the roots if so, since the spinodal branches can start/end at the domain boundaries
        if discriminant[0] > 0:
            roots_idx = np.r_[0, roots_idx]
        if discriminant[-1] >= 0:
            roots_idx = np.r_[roots_idx, len(phi1_vals) - 1]
        # Add 1 at beginning of domain so the discriminant is positive
        roots_idx[::2] += 1
        return phi1_vals[roots_idx]

    @classmethod
    def _in_domain(cls, phi1, phi2):
        """Check whether compositions lie in the open ternary simplex.

        Parameters
        ----------
        phi1 : float or array-like
            First independent composition coordinate.
        phi2 : float or array-like
            Second independent composition coordinate, broadcast-compatible
            with ``phi1``.

        Returns
        -------
        bool or numpy.ndarray
            ``True`` where ``phi1 > 0``, ``phi2 > 0``, and
            ``phi1 + phi2 < 1``. Boundary points are excluded.
        """
        phi1 = np.asarray(phi1)
        phi2 = np.asarray(phi2)
        return (phi1 > 0) & (phi2 > 0) & (phi1 + phi2 < 1)

    def _clip_to_domain(self):
        """Remove graph nodes outside the physical composition domain.

        Returns
        -------
        None

        Notes
        -----
        This method mutates :attr:`spinodal_graph`. After removing points
        outside the open simplex, it also removes connected components that
        contain only one node.
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
        """Join nearby graph endpoints that form a smooth continuation.

        Returns
        -------
        None

        Notes
        -----
        Endpoints less than ``0.05`` apart are connected when each endpoint
        lies approximately between the other endpoint and its own neighbor.
        Connections are chosen greedily, and each endpoint is used at most
        once during a call.
        """

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
        """Extract graph coordinates in traversal order.

        Parameters
        ----------
        sg : networkx.Graph
            Connected spinodal subgraph representing a path or cycle.

        Returns
        -------
        phi1 : numpy.ndarray
            First composition coordinates with shape ``(n_nodes,)``.
        phi2 : numpy.ndarray
            Corresponding second composition coordinates with shape
            ``(n_nodes,)``.

        Notes
        -----
        Traversal starts at an arbitrary endpoint for a path and at an
        arbitrary node for a cycle. Depth-first preorder then determines the
        orientation of the returned coordinates.
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
        """Evaluate the third-derivative criticality condition.

        Parameters
        ----------
        phi1 : float or array-like
            First independent composition coordinate.
        phi2 : float or array-like
            Second independent composition coordinate, broadcast-compatible
            with ``phi1``.

        Returns
        -------
        float or numpy.ndarray
            Criticality scalar with the broadcast shape of the inputs. Its
            zeros along a spinodal branch identify candidate critical points.

        Notes
        -----
        Inputs must lie in the open composition simplex to avoid singular
        denominators.
        """
        phi0 = 1 - phi1 - phi2
        H_11 = 1 / phi1 + 1 / phi0 + self.chis[0, 0]
        H_12 = 1 / phi0 + self.chis[0, 1]
        return H_12**3 / (phi1**2) - H_11**3 / (phi2**2) + (H_11 - H_12) ** 3 / phi0**2

    def _find_critical_points(self):
        """Locate and store critical points on every spinodal branch.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the third-derivative values contain ``NaN``.
        RuntimeError
            If an interpolated sign-change root is not sufficiently close to
            zero relative to its neighboring samples.

        Notes
        -----
        Candidates are obtained both from sign changes and from local extrema
        whose absolute value is below ``1e-4``. Sign-change roots are linearly
        interpolated in both composition coordinates. Results are appended to
        :attr:`critical_points`; the existing list is not cleared.
        """
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
        """Construct locally stable polygons from spinodal graph components.

        Returns
        -------
        None

        Notes
        -----
        Open branches are extended to nearby simplex boundaries and, when
        necessary, through a simplex corner. Candidate polygons are classified
        from the Hessian eigenvalues at their centroids; polygons enclosing a
        locally unstable centroid are complemented against the full simplex.
        Overlapping stable polygons are intersected. The resulting geometries
        replace :attr:`polygons`.

        Components with fewer than three coordinates or an invalid polygon
        emit a message and are skipped.
        """
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
        """Return the locally unstable region of the composition simplex.

        Returns
        -------
        shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
            Full composition simplex with every locally stable polygon
            removed. ``None`` is returned when no stable polygons exist.

        Raises
        ------
        ValueError
            If polygons have not yet been constructed. Call :meth:`build`
            first.
        """
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
        """Plot spinodal branches and critical points on the current axes.

        Parameters
        ----------
        **kwargs
            Plot options. The ``linestyle`` entry is applied to spinodal
            branches and defaults to ``"--"``; other entries are currently
            ignored.

        Returns
        -------
        None

        Notes
        -----
        The method sets the title, axis limits, labels, legend, and the
        ``phi1 + phi2 = 1`` boundary on the current Matplotlib axes.
        """
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
