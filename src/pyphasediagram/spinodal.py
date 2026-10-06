"""Construct and analyze spinodal curves for ternary mixtures.

The spinodal is represented as an undirected graph of points in the two
independent composition coordinates ``(phi1, phi2)``. This module samples the
analytic spinodal branches, connects and clips them to the physically admissible domain,
locates critical points, and constructs polygons used to identify the
locally unstable region.
"""

import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
from numpy.typing import ArrayLike

from pyphasediagram.point import CriticalPoint, SpinodalPoint

FloatOrArray = Union[float, np.ndarray]

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
    Compositions lie in the open domain ``phi1 > 0``, ``phi2 > 0``,
    and ``phi1 + phi2 < 1``. The solvent composition is
    ``phi0 = 1 - phi1 - phi2``.
    """

    DOMAIN_CORNERS = [(0, 0), (0, 1), (1, 0)]

    def __init__(self, chis: np.ndarray) -> None:
        if chis.shape != (2, 2):
            raise ValueError(
                f"Expected chis to be a 2x2 matrix, but got shape {chis.shape}"
            )
        self.chis = chis
        self.spinodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []
        self.polygons: list[shapely.geometry.Polygon] = None

    def build(self, num_points: int = 10000) -> None:
        """Build the spinodal graph and derived geometric objects.

        Parameters
        ----------
        num_points : int, optional
            Number of ``phi1`` samples used each domain interval.

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
        physically admissible domain, connects nearby branches, locates
        critical points, and constructs polygons which contain the locally stable regions in composition space.
        Existing graph nodes, critical points, and polygons are cleared before
        rebuilding, so repeated calls replace rather than append to the
        calculated state.
        """
        self.spinodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []
        self.polygons = None

        phi1_domains = self._spinodal_domains()
        for i in range(0, len(phi1_domains), 2):
            self._domain_data(phi1_domains[i], phi1_domains[i + 1], num_points)
        self._clip_to_domain()
        self._connect_branches()
        self._find_critical_points()
        self._build_polygons()

    def _get_quadratic_coefficients(
        self, phi: ArrayLike, is_calculate_phi2: bool = True
    ) -> tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
        """Return the unnormalized spinodal-polynomial coefficients.

        Parameters
        ----------
        phi : float or array-like
            Known independent composition coordinate.
        is_calculate_phi2 : bool, optional
            If ``True``, form the polynomial for ``phi2`` given ``phi1``.
            If ``False``, form the polynomial for ``phi1`` given ``phi2``.

        Returns
        -------
        quadratic : float or numpy.ndarray
            Coefficient of ``x**2``.
        linear : float or numpy.ndarray
            Coefficient of ``x``.
        constant : float or numpy.ndarray
            Constant term in ``quadratic*x**2 + linear*x + constant = 0``.
        """
        a, b, c = self.chis[1, 1], self.chis[0, 0], self.chis[0, 1]
        if is_calculate_phi2 == False:
            a, b = b, a  # swap a and b if computing phi1 from phi2

        det = a * b - c**2
        quadratic = a + det * phi
        linear = 2 * phi * c - a - phi * (1 - phi) * det
        constant = -(1 + phi * (1 - phi) * b)
        return quadratic, linear, constant

    @staticmethod
    def _polynomial_degree_masks(
        quadratic: ArrayLike, linear: ArrayLike, constant: ArrayLike
    ) -> tuple[Union[bool, np.ndarray], Union[bool, np.ndarray]]:
        """Classify polynomial samples as quadratic or linear.

        Coefficients smaller than floating-point resolution relative to the
        polynomial's coefficient scale are treated as zero.

        Returns
        -------
        is_quadratic : bool or numpy.ndarray
            Samples with a nonzero quadratic coefficient.
        is_linear : bool or numpy.ndarray
            Samples with a zero quadratic coefficient and nonzero linear
            coefficient.
        """
        coefficient_scale = np.maximum.reduce(
            [
                np.abs(quadratic),
                np.abs(linear),
                np.abs(constant),
                np.ones_like(quadratic, dtype=float),
            ]
        )
        zero_tolerance = 32 * np.finfo(float).eps * coefficient_scale
        is_quadratic = np.abs(quadratic) > zero_tolerance
        is_linear = ~is_quadratic & (np.abs(linear) > zero_tolerance)
        return is_quadratic, is_linear

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
        Coordinates outside the admissible domain are excluded. Both
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

    def _phi2_from_phi1(
        self, phi1: ArrayLike
    ) -> tuple[FloatOrArray, FloatOrArray]:
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
        discriminant is negative produce ``NaN`` values. If the quadratic
        coefficient vanishes but the linear coefficient does not, the sole
        linear solution is returned in both branch arrays. Fully degenerate
        equations, which do not define a unique solution, produce ``NaN``.
        """
        quadratic, linear, constant = self._get_quadratic_coefficients(phi1)
        is_quadratic, is_linear = self._polynomial_degree_masks(
            quadratic, linear, constant
        )

        upper = np.full_like(quadratic, np.nan, dtype=float)
        lower = np.full_like(quadratic, np.nan, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.divide(
                linear,
                quadratic,
                out=np.full_like(quadratic, np.nan, dtype=float),
                where=is_quadratic,
            )
            q = np.divide(
                -constant,
                quadratic,
                out=np.full_like(quadratic, np.nan, dtype=float),
                where=is_quadratic,
            )
            discriminant = p**2 / 4 + q
            has_real_quadratic_roots = (
                is_quadratic & np.isfinite(discriminant) & (discriminant >= 0)
            )
            square_root = np.sqrt(
                discriminant,
                where=has_real_quadratic_roots,
                out=np.full_like(discriminant, np.nan, dtype=float),
            )
            upper[has_real_quadratic_roots] = (
                -p[has_real_quadratic_roots] / 2 + square_root[has_real_quadratic_roots]
            )
            lower[has_real_quadratic_roots] = (
                -p[has_real_quadratic_roots] / 2 - square_root[has_real_quadratic_roots]
            )

            linear_root = np.divide(
                -constant,
                linear,
                out=np.full_like(linear, np.nan, dtype=float),
                where=is_linear,
            )
            upper[is_linear] = linear_root[is_linear]
            lower[is_linear] = linear_root[is_linear]

        return upper, lower

    def eigenvalues_from_phi(
        self, phi1: ArrayLike, phi2: ArrayLike
    ) -> tuple[FloatOrArray, FloatOrArray]:
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

    def _domain_data(
        self, phi1_i: float, phi1_f: float, num_points: int = 5000
    ) -> None:
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
        phi2_branches = self._phi2_from_phi1(phi1_vals)

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

    def _spinodal_domains(self) -> np.ndarray:
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
        ``[0, 1]``. Interval boundaries are therefore located to the
        resolution of that grid rather than by a continuous root solver.
        Linear samples, where the quadratic coefficient vanishes, are included
        when their linear coefficient is nonzero. Non-finite samples, fully
        degenerate equations, and isolated real solutions are excluded.
        """
        phi1_vals = np.linspace(0, 1, 100000)

        quadratic, linear, constant = self._get_quadratic_coefficients(phi1_vals)
        is_quadratic, is_linear = self._polynomial_degree_masks(
            quadratic, linear, constant
        )
        finite_coefficients = (
            np.isfinite(quadratic) & np.isfinite(linear) & np.isfinite(constant)
        )
        with np.errstate(invalid="ignore", over="ignore"):
            discriminant = linear**2 - 4 * quadratic * constant

        has_real_quadratic_roots = (
            finite_coefficients
            & is_quadratic
            & np.isfinite(discriminant)
            & (discriminant >= 0)
        )
        has_real_linear_root = finite_coefficients & is_linear
        has_real_solutions = has_real_quadratic_roots | has_real_linear_root
        domain_endpoints = []
        start_idx = None

        for idx, has_real_solution in enumerate(has_real_solutions):
            if has_real_solution and start_idx is None:
                # A new contiguous interval starts here.
                start_idx = idx
            elif not has_real_solution and start_idx is not None:
                # The previous sample was the end of the current interval.
                end_idx = idx - 1
                if start_idx < end_idx:
                    domain_endpoints.extend((phi1_vals[start_idx], phi1_vals[end_idx]))
                start_idx = None

        # Close an interval that reaches the upper edge of the sampled domain.
        if start_idx is not None:
            end_idx = len(phi1_vals) - 1
            if start_idx < end_idx:
                domain_endpoints.extend((phi1_vals[start_idx], phi1_vals[end_idx]))

        return np.asarray(domain_endpoints, dtype=float)

    @classmethod
    def _in_domain(
        cls, phi1: ArrayLike, phi2: ArrayLike
    ) -> Union[bool, np.ndarray]:
        """Check whether compositions lie in the open ternary domain.

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

    def _clip_to_domain(self) -> None:
        """Remove graph nodes outside the physical composition domain.

        Returns
        -------
        None

        Notes
        -----
        This method mutates :attr:`spinodal_graph`. After removing points
        outside the admissible domain, it also removes connected components that
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

    def _connect_branches(self) -> None:
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

    def _coords_from_subgraph(self, sg: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
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

    def _third_derivative(self, phi1: ArrayLike, phi2: ArrayLike) -> FloatOrArray:
        """Evaluate the third-derivative criticality condition.

        Parameters
        ----------
        phi1 : float or array-like
            First independent composition coordinate.
        phi2 : float or array-like
            Second independent composition coordinate, with the same shape as ``phi1``.

        Returns
        -------
        float or numpy.ndarray
            Criticality scalar with the broadcast shape of the inputs. Its
            zeros along a spinodal branch identify candidate critical points.

        Notes
        -----
        Inputs must lie in the admissible composition domain to avoid singular
        denominators.
        """
        phi0 = 1 - phi1 - phi2
        H_11 = 1 / phi1 + 1 / phi0 + self.chis[0, 0]
        H_12 = 1 / phi0 + self.chis[0, 1]
        return H_12**3 / (phi1**2) - H_11**3 / (phi2**2) + (H_11 - H_12) ** 3 / phi0**2

    def _find_critical_points(self) -> None:
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
                if abs(td_c) > 1e-2:
                    warnings.warn(
                        f"Third derivative at critical point ({phi1_c:.2f}, "
                        f"{phi2_c:.2f}) is quite large: {td_c:.5f}.\n"
                        "Try increasing resolution of spinodal curve.",
                        RuntimeWarning,
                        stacklevel=2,
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

    def _build_polygons(self) -> None:
        """Construct locally stable polygons from spinodal graph components.

        Returns
        -------
        None

        Notes
        -----
        Open branches are extended to nearby admissible domain boundaries and, when
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
                        angle_from_vertical = np.arctan2(abs(d[0]), abs(d[1]))
                        if (
                            angle_from_vertical < 0.01
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
            Full composition space with every locally stable polygon
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

    def plot(self, **kwargs: Any) -> None:
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
        has_plotted_spinodal = False
        for comp in nx.connected_components(self.spinodal_graph):
            sg = self.spinodal_graph.subgraph(comp)

            # Find endpoint (degree 1) if it exists (path case)
            phi1s, phi2s = self._coords_from_subgraph(sg)
            if np.isnan(phi1s).any() or np.isnan(phi2s).any():
                print("NaN coordinates found, skipping plot for this component.")
                continue
            if not has_plotted_spinodal:
                plt.plot(phi1s, phi2s, label="Spinodal curve", linestyle=ls)
                has_plotted_spinodal = True
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
