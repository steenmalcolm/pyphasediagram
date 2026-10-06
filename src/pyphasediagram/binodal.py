"""Construct and analyze binodal curves for ternary mixtures.

The module traces paired coexistence branches in the two
composition coordinates, detects candidate three-phase regions from branch
intersections, and constructs polygonal two- and three-phase regions. It also
provides helpers for plotting the resulting geometry and selecting a discrete
phase decomposition for a prescribed mean composition.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely

from pyphasediagram.point import BinodalInitialPoint, CriticalPoint
from pyphasediagram.stepper import Stepper


# TODO: Additional duplicate section removal after task list is exhausted and unstable manifold is removed.
class BinodalSection:
    r"""Store one traced section of a two-phase coexistence curve.

    Parameters
    ----------
    phis : numpy.ndarray
        Coexisting compositions with shape ``(2, 2, n_points)``. The axes
        identify phase, independent component, and position along the traced
        section, respectively.
    svs : numpy.ndarray
        Smallest residual-Jacobian singular value at each trace point, with
        shape ``(n_points,)``. This is used to determine branching points

    Attributes
    ----------
    phis : numpy.ndarray
        Coexisting compositions supplied to the constructor.
    svs : numpy.ndarray
        Singular-value history supplied to the constructor.
    line_a : shapely.LineString
        Composition-space curve traced by phase A.
    line_b : shapely.LineString
        Composition-space curve traced by phase B.

    Raises
    ------
    ValueError
        If ``phis`` and ``svs`` have different numbers of trace points.

    Notes
    -----
    For trace index :math:`k`, ``phis[:, :, k]`` contains the endpoints of
    one tie line. At least two trace points are expected. The constructor
    assumes that ``phis`` has shape ``(2, 2, n_points)`` and validates that
    ``svs`` contains one value per trace point. The Shapely lines and cached
    coordinates are created at initialization and do not track later
    mutations of ``phis``.
    """

    def __init__(self, phis: np.ndarray, svs: np.ndarray):
        if phis.shape[2] != len(svs):
            raise ValueError("phis and svs must have the same number of trace points")
        self.phis = phis
        self.svs = svs
        self.line_a = shapely.LineString(phis[0].T)
        self.line_b = shapely.LineString(phis[1].T)
        self._lines = (self.line_a, self.line_b)
        self._coords = tuple(np.asarray(line.coords) for line in self._lines)

    def plot(self, colora="red", colorb="blue", is_tie_lines=False, **kwargs):
        """Plot both coexistence branches on the current axes.

        Parameters
        ----------
        colora : color, optional
            Matplotlib color used for the phase-A branch.
        colorb : color, optional
            Matplotlib color used for the phase-B branch.
        is_tie_lines : bool, optional
            Whether to draw gray lines between corresponding phase
            compositions.
        **kwargs
            Additional keyword arguments passed to
            :func:`matplotlib.pyplot.plot` for both coexistence branches.

        Returns
        -------
        None

        Notes
        -----
        Tie lines are subsampled with a stride of
        ``max(1, n_points // 20)`` to reduce visual clutter.
        """
        plt.plot(self.phis[0, 0], self.phis[0, 1], color=colora, **kwargs)
        plt.plot(self.phis[1, 0], self.phis[1, 1], color=colorb, **kwargs)
        if is_tie_lines:
            # Plot at most 20 tie lines for clarity
            step_size = max(1, self.phis.shape[2] // 20)
            plt.plot(
                [self.phis[0, 0, ::step_size], self.phis[1, 0, ::step_size]],
                [self.phis[0, 1, ::step_size], self.phis[1, 1, ::step_size]],
                color="gray",
                alpha=0.5,
            )

    @staticmethod
    def _interpolate_on_segment(coords: np.ndarray, point) -> tuple[int, float]:
        r"""Locate a point on the nearest segment of a polyline.

        Parameters
        ----------
        coords : numpy.ndarray
            Ordered polyline vertices with shape ``(n_vertices, 2)``.
        point : shapely.Point
            Point whose nearest projected position is requested.

        Returns
        -------
        segment_index : int
            Index of the first vertex of the nearest segment.
        t : float
            Clipped interpolation coordinate :math:`t \in [0, 1]`, where
            ``coords[segment_index]`` corresponds to zero and the following
            vertex corresponds to one.

        Notes
        -----
        Squared segment lengths are bounded below by ``1e-30`` so repeated
        neighboring vertices can be handled without division by zero.
        """
        point_xy = np.array([point.x, point.y])
        seg_starts = coords[:-1]
        seg_vecs = coords[1:] - seg_starts
        seg_lens_sq = np.sum(seg_vecs**2, axis=1)
        seg_lens_sq = np.maximum(seg_lens_sq, 1e-30)
        t = np.clip(
            np.sum((point_xy - seg_starts) * seg_vecs, axis=1) / seg_lens_sq,
            0.0,
            1.0,
        )
        projections = seg_starts + t[:, np.newaxis] * seg_vecs
        distances_sq = np.sum((projections - point_xy) ** 2, axis=1)
        best = int(np.argmin(distances_sq))
        return best, float(t[best])

    def _intersection_params(self, self_idx: int, other, other_idx: int):
        """Find interpolation coordinates at intersections of two branches.

        Parameters
        ----------
        self_idx : {0, 1}
            Branch index from this section.
        other : BinodalSection
            Section providing the second branch.
        other_idx : {0, 1}
            Branch index from ``other``.

        Returns
        -------
        list of tuple
            Tuples ``(self_segment, self_t, other_segment, other_t)`` for
            every point intersection. Segment entries are integers and the
            interpolation entries are floats in the closed interval
            ``[0, 1]``.

        Notes
        -----
        Disjoint bounding boxes are rejected before performing the Shapely
        intersection. Only ``Point`` and ``MultiPoint`` results are retained;
        overlapping line segments are ignored.
        """
        line_1 = self._lines[self_idx]
        line_2 = other._lines[other_idx]

        # Skip if line bounding squares do not overlap
        minx_1, miny_1, maxx_1, maxy_1 = line_1.bounds
        minx_2, miny_2, maxx_2, maxy_2 = line_2.bounds
        if maxx_1 < minx_2 or maxx_2 < minx_1 or maxy_1 < miny_2 or maxy_2 < miny_1:
            return []

        intersection = line_1.intersection(line_2)
        if intersection.geom_type == "Point":
            points = (intersection,)
        elif intersection.geom_type == "MultiPoint":
            points = intersection.geoms
        else:
            return []

        coords_1 = self._coords[self_idx]
        coords_2 = other._coords[other_idx]
        return [
            (
                *self._interpolate_on_segment(coords_1, point),
                *self._interpolate_on_segment(coords_2, point),
            )
            for point in points
        ]

    def intersects_with(self, other):
        """Construct three-phase candidates from branch intersections.

        Parameters
        ----------
        other : BinodalSection
            Section to intersect with this section.

        Returns
        -------
        list of tuple of numpy.ndarray
            Candidate triples of phase compositions. Each tuple contains
            three arrays with shape ``(2,)``: the intersecting composition,
            its partner phase on this section, and the partner phase on
            ``other``.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a :class:`BinodalSection`.

        Notes
        -----
        All four pairings of the phase-A and phase-B branches are tested.
        Coordinates between trace points are obtained by linear
        interpolation along both paired branches.
        """
        if not isinstance(other, BinodalSection):
            raise NotImplementedError(
                "intersects_with can only be computed between BinodalSection instances."
            )

        tp_points = []
        for self_idx, _ in enumerate(self._lines):
            other_self_idx = 1 - self_idx
            for other_idx, _ in enumerate(other._lines):
                other_other_idx = 1 - other_idx
                params = self._intersection_params(self_idx, other, other_idx)
                for seg1, t1, seg2, t2 in params:
                    self_phi_a = (1 - t1) * self.phis[
                        self_idx, :, seg1
                    ] + t1 * self.phis[self_idx, :, seg1 + 1]
                    self_phi_b = (1 - t1) * self.phis[
                        other_self_idx, :, seg1
                    ] + t1 * self.phis[other_self_idx, :, seg1 + 1]
                    other_phi = (1 - t2) * other.phis[
                        other_other_idx, :, seg2
                    ] + t2 * other.phis[other_other_idx, :, seg2 + 1]
                    tp_points.append((self_phi_a, self_phi_b, other_phi))

        return tp_points

    def contained_in(self, other, n_samples=1000):
        r"""Check whether this section approximately duplicates another.

        Parameters
        ----------
        other : BinodalSection
            Reference section against which this section is compared.
        n_samples : int, optional
            Number of equally spaced arc-length samples taken from each
            branch of this section.

        Returns
        -------
        bool
            ``True`` when the mean distances from both sampled branches to
            the corresponding branches of ``other`` are below :math:`10^{-3}`.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a :class:`BinodalSection`.

        Notes
        -----
        Both assignments of this section's phase labels are tested because
        coexistence is invariant under exchanging phases A and B. The test is
        directional: it samples this section and measures distances to
        ``other``, but not conversely.
        """
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
        """Find candidate branching points from singular-value minima.

        Parameters
        ----------
        sv_branch_threshold : float, optional
            Exclusive upper bound on a local minimum of :attr:`svs` for it to
            be treated as a degeneracy.

        Returns
        -------
        list of BinodalInitialPoint
            Candidate initial states located at qualifying local minima. The
            tracing direction is estimated from the neighboring coexistence
            states by a centered difference.

        Notes
        -----
        A small singular value indicates that the residual Jacobian may have
        a null space of dimension greater than one, as expected at a branch
        point.
        """
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
        """Return the number of sampled coexistence states in the section."""
        return self.phis.shape[2]


class Binodal:
    """Trace and organize the coexistence regions of a ternary mixture.

    Parameters
    ----------
    chis : numpy.ndarray
        Reduced interaction matrix with shape ``(2, 2)`` for the two
        independent composition coordinates.
    critical_points : list of CriticalPoint, optional
        Spinodal critical points used as additional seeds for tracing
        coexistence sections.

    Attributes
    ----------
    SV_BRANCH_THRESHOLD : float
        Singular-value threshold used to identify candidate branch points.
    chis : numpy.ndarray
        Reduced interaction matrix supplied to the constructor.
    critical_points : list of CriticalPoint
        Critical-point seeds supplied to the constructor.
    binodal_sections : list of BinodalSection
        Accepted coexistence-curve sections. The list is empty until sections
        are built.
    three_phase_polygons : list of shapely.Polygon or None
        Candidate three-phase regions, or ``None`` before polygon
        construction.
    two_phase_polygons : list of shapely geometry or None
        Polygonal two-phase regions, or ``None`` before polygon construction.

    Notes
    -----
    Call :meth:`build` to trace the coexistence sections and populate the
    polygon collections. The input interaction matrix and critical-point list
    are stored without copying.
    """

    SV_BRANCH_THRESHOLD = 1e-2

    def __init__(self, chis: np.ndarray, critical_points: list[CriticalPoint] = []):
        self.chis = chis
        self.critical_points = critical_points
        self.binodal_sections: list[BinodalSection] = []
        self.three_phase_polygons: list[shapely.Polygon] = None
        self.two_phase_polygons: list[shapely.Polygon] = None
        self._stepper = Stepper(chis)
        self._bipt_task_list: list[BinodalInitialPoint] = []
        self._bipt_hist: list[BinodalInitialPoint] = []

    def _build_section(self, phi_init, v_init):
        """Trace and store one coexistence-curve section.

        Parameters
        ----------
        phi_init : numpy.ndarray
            Initial phase compositions with shape ``(4,)`` and ordering
            ``[phi1_a, phi2_a, phi1_b, phi2_b]``.
        v_init : numpy.ndarray
            Initial tracing direction with shape ``(4,)`` and matching
            coordinate ordering.

        Returns
        -------
        bool
            ``True`` if a section was accepted and stored; ``False`` if the
            trace contained fewer than three points or approximately
            duplicated an existing section.

        Notes
        -----
        Accepted sections contribute newly detected branch-point seeds to the
        pending task list. Their initial states are also recorded so similar
        seeds can be skipped later.
        """
        # Run the stepper to compute the binodal points along the section
        phis, svs, flags = self._stepper.run(phi_init, v_init)
        # if len(flags):
        #     print("\t", end=" ")
        #     for flag in flags:
        #         print(f"{flag}", end=", ")
        #     print()
        # Edge case where section is too short
        if phis.shape[-1] < 3:
            return False
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

    def build(self, unstable_manifold: shapely.Polygon = None) -> None:
        """Build the binodal sections and phase-region polygons.

        Parameters
        ----------
        unstable_manifold : Shapely polygonal geometry or None, optional
            Locally unstable composition region. Trace points for which either
            coexisting phase lies strictly inside this geometry are removed.
            If ``None``, no stability-based filtering is performed.

        Returns
        -------
        None

        Notes
        -----
        Tracing is seeded from every demixing binary limit and from
        :attr:`critical_points`. Newly detected branch points are processed
        until the task list is exhausted. The resulting sections are then
        filtered, deduplicated, and converted into two- and three-phase
        polygonal regions. Existing section and seed history is not cleared,
        so a fresh instance should be used when a complete rebuild is needed.
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

        self._remove_unstable_sections(unstable_manifold)
        self._remove_duplicate_sections()
        self._find_phase_polygons()

    def _remove_duplicate_sections(self):
        """Remove sections approximately contained in another section.

        Returns
        -------
        None

        Notes
        -----
        Approximate containment is evaluated by
        :meth:`BinodalSection.contained_in`. Pair bookkeeping prevents both
        members of a mutually contained pair from being removed.
        """
        unique_sections = []
        duplicate_pair_idxs = []
        for i, section in enumerate(self.binodal_sections):
            is_duplicate = False
            for j, other in enumerate(self.binodal_sections):
                # Avoid removing both duplicates
                if (j, i) in duplicate_pair_idxs:
                    continue
                if i != j and section.contained_in(other):
                    is_duplicate = True
                    duplicate_pair_idxs.append((i, j))
                    break
            if not is_duplicate:
                unique_sections.append(section)
        self.binodal_sections = unique_sections

    def _find_branching_points(
        self, section: BinodalSection
    ) -> list[BinodalInitialPoint]:
        """Refine branch-point candidates from a binodal section.

        Parameters
        ----------
        section : BinodalSection
            Newly traced section to inspect for degeneracies.

        Returns
        -------
        list of BinodalInitialPoint
            Initial states displaced along the secondary null-space direction
            at accepted branch points.

        Notes
        -----
        Candidates come from local singular-value minima below
        :attr:`SV_BRANCH_THRESHOLD`. A candidate is accepted when its current
        direction has a dot product greater than ``0.98`` with the primary
        right-null vector and the smallest Jacobian singular value remains
        below the threshold. The new state is displaced by ``1e-3`` along the
        secondary right-null vector.
        """
        branching_points: list[BinodalInitialPoint] = []
        for bp in section._degenerate_points(self.SV_BRANCH_THRESHOLD):
            J = self._stepper._jac_fn(bp.phi_init)
            U, S, Vt = self._stepper._svd(J, full_matrices=True)
            if np.dot(Vt[-1], bp.v_init) > 0.98 and S[-1] < self.SV_BRANCH_THRESHOLD:
                bp.v_init = np.asarray(Vt[-2])
                bp.phi_init += bp.v_init * 1e-3
                branching_points.append(bp)

        return branching_points

    def _find_phase_polygons(self):
        """Construct the three-phase and two-phase polygon collections.

        Returns
        -------
        None
        """
        self._find_three_phase_polygons()
        self._find_two_phase_polygons()

    def _find_three_phase_polygons(self, overlap_threshold: float = 0.98):
        r"""Construct candidate three-phase polygons from section crossings.

        Parameters
        ----------
        overlap_threshold : float, optional
            Exclusive overlap fraction above which a candidate is considered
            a duplicate of an existing polygon.

        Returns
        -------
        None

        Notes
        -----
        Every nondegenerate candidate triple returned by
        :meth:`BinodalSection.intersects_with` defines a triangle. Candidates
        with area below :math:`10^{-10}` are discarded. For intersecting
        candidates :math:`A` and existing polygons :math:`B`, the duplicate
        measure is

        .. math::

           \frac{\operatorname{area}(A \cap B)}
                {\min(\operatorname{area}(A), \operatorname{area}(B))}.

        This method replaces the existing :attr:`three_phase_polygons` list.
        """
        self.three_phase_polygons = []
        for i, section_a in enumerate(self.binodal_sections):
            for j, section_b in enumerate(self.binodal_sections):
                if i > j:
                    continue
                i_points = section_a.intersects_with(section_b)
                for i_point in i_points:
                    poly = shapely.Polygon(i_point)
                    if poly.area < 1e-10:
                        continue
                    # Skip near-duplicate polygons (overlap > threshold)
                    is_duplicate = False
                    for tp_poly in self.three_phase_polygons:
                        if not poly.intersects(tp_poly):
                            continue
                        intersection_area = poly.intersection(tp_poly).area
                        overlap = intersection_area / min(poly.area, tp_poly.area)
                        if overlap > overlap_threshold:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        self.three_phase_polygons.append(poly)

    def _find_two_phase_polygons(self):
        r"""Construct polygonal regions associated with two-phase coexistence.

        Returns
        -------
        None

        Notes
        -----
        If necessary, three-phase polygons are constructed first. Each
        binodal section is closed by traversing its phase-A branch forward and
        its phase-B branch in reverse. Geometries with area below
        :math:`10^{-6}` are discarded, three-phase regions are subtracted, and
        intersecting two-phase geometries are merged.

        Geometry repair, subtraction, and union may produce polygonal geometry
        types other than ``shapely.Polygon``. This method replaces the
        existing :attr:`two_phase_polygons` list.
        """
        if self.three_phase_polygons is None:
            self._find_three_phase_polygons()

        self.two_phase_polygons = []
        for section in self.binodal_sections:
            # Build a closed polygon: line_a forward, then line_b reversed
            coords_a = np.asarray(section.line_a.coords)
            coords_b = np.asarray(section.line_b.coords)
            ring = np.concatenate([coords_a, coords_b[::-1]])
            if len(ring) < 3:
                continue
            poly = shapely.make_valid(shapely.Polygon(ring))
            if poly.area < 1e-6:
                continue

            # Subtract three-phase regions
            for tp_poly in self.three_phase_polygons:
                if poly.intersects(tp_poly):
                    poly = shapely.make_valid(poly.difference(tp_poly))

            if poly.area >= 1e-6:
                polys_remove = []
                for tp_poly in self.two_phase_polygons:
                    if poly.intersects(tp_poly):
                        poly = shapely.make_valid(poly.union(tp_poly))
                        polys_remove.append(tp_poly)
                for tp_poly in polys_remove:
                    self.two_phase_polygons.remove(tp_poly)
                self.two_phase_polygons.append(poly)

    def _remove_unstable_sections(self, unstable_manifold: shapely.Polygon = None):
        """Remove trace samples that enter a locally unstable region.

        Parameters
        ----------
        unstable_manifold : Shapely geometry or None, optional
            Polygonal region to exclude. If ``None``, the section collection
            is left unchanged.

        Returns
        -------
        None

        Notes
        -----
        A sample is removed when either of its two phase compositions lies
        strictly inside ``unstable_manifold``. Polygon boundaries are
        retained. Each section is split into contiguous runs of retained
        samples, and runs containing fewer than two samples are discarded.
        Splitting occurs only at existing trace samples; no boundary
        intersections are inserted.
        """
        if unstable_manifold is None:
            return
        stable_sections = []
        for section in self.binodal_sections:
            # Vectorized containment check for both lines
            inside_a = shapely.contains_xy(
                unstable_manifold, section.phis[0, 0], section.phis[0, 1]
            )
            inside_b = shapely.contains_xy(
                unstable_manifold, section.phis[1, 0], section.phis[1, 1]
            )
            stable_mask = ~(inside_a | inside_b)

            if stable_mask.all():
                stable_sections.append(section)
                continue

            # Find contiguous runs of stable vertices
            stable_indices = np.where(stable_mask)[0]
            if len(stable_indices) < 2:
                continue
            segments = np.split(
                stable_indices, np.where(np.diff(stable_indices) > 1)[0] + 1
            )
            for seg in segments:
                if len(seg) < 2:
                    continue
                stable_sections.append(
                    BinodalSection(section.phis[:, :, seg], section.svs[seg])
                )

        self.binodal_sections = stable_sections

    def _free_energy(self, phi: np.ndarray) -> float:
        r"""Evaluate the dimensionless free-energy expression used for ranking.

        Parameters
        ----------
        phi : numpy.ndarray
            Independent composition coordinates
            :math:`\boldsymbol{\phi} = (\phi_1, \phi_2)` with shape ``(2,)``.

        Returns
        -------
        float
            Dimensionless free-energy density.

        Notes
        -----
        With :math:`\phi_0 = 1 - \phi_1 - \phi_2` and reduced interaction
        matrix :math:`C =` ``chis``, this method evaluates

        .. math::

           f(\boldsymbol{\phi}) = \sum_{i=0}^{2} \phi_i \ln \phi_i
           + \boldsymbol{\phi}^{\mathsf{T}} C\boldsymbol{\phi}.

        The logarithms require all three compositions to be positive for a
        finite real result. The method performs no domain validation.
        """
        phi_1, phi_2 = phi[0], phi[1]
        phi_0 = 1.0 - phi_1 - phi_2
        return (
            phi_1 * np.log(phi_1)
            + phi_2 * np.log(phi_2)
            + phi_0 * np.log(phi_0)
            + np.dot(phi, self.chis @ phi)
        )

    # TODO: Currently only the closest tie line is selected. It would be better to do an interpolation between the two closest tie lines
    def decomposition_from_composition(
        self, phi_means: np.ndarray
    ) -> tuple[np.ndarray | None, float | None]:
        r"""Select candidate coexisting phases for a mean composition.

        Parameters
        ----------
        phi_means : array-like
            Mean independent composition coordinates with shape ``(2,)`` and
            ordering ``[phi1, phi2]``.

        Returns
        -------
        phase_compositions : numpy.ndarray or None
            Coexisting phase compositions. A three-phase result has shape
            ``(3, 2)`` and a two-phase result has shape ``(2, 2)``; rows
            identify phases and columns identify independent components.
            ``None`` is returned if no candidate is found.
        free_energy : float or None
            Weighted free energy of the selected decomposition, or ``None``
            if no candidate is found.

        Notes
        -----
        Call :meth:`build` before this method so that three-phase polygons are
        available. For a mean composition strictly inside a three-phase
        polygon, phase fractions :math:`v_q` are obtained from the lever rule

        .. math::

           \overline{\boldsymbol{\phi}}
           = \sum_q v_q\boldsymbol{\phi}^{(q)},
           \qquad \sum_q v_q = 1,

        and the candidate free energy is

        .. math::

           f_{\mathrm{mix}} = \sum_q v_q f(\boldsymbol{\phi}^{(q)}).

        Two-phase candidates are selected from the discrete tie lines stored
        in each section. The closest tie line whose orthogonal projection
        parameter lies in ``[0, 1]`` is considered; adjacent tie lines are not
        interpolated, and exact incidence of the mean composition on the tie
        line is not required. The lowest-free-energy candidate is returned,
        without comparison against the homogeneous state's free energy.
        """
        phi_means = np.asarray(phi_means, dtype=float)
        pt = shapely.Point(phi_means)

        best_result = None
        best_f = np.inf

        # --- Three-phase candidates ---
        if len(self.three_phase_polygons):
            for poly in self.three_phase_polygons:
                if not poly.contains(pt):
                    continue
                verts = np.array(poly.exterior.coords[:3])  # (3, 2)
                # Solve for volume fractions: verts.T @ v = phi_means, sum(v) = 1
                A = np.vstack([verts.T, np.ones(3)])  # (3, 3)
                b = np.append(phi_means, 1.0)
                try:
                    vols = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                if np.any(vols < -1e-6):
                    continue
                f = float(
                    sum(v * self._free_energy(verts[i]) for i, v in enumerate(vols))
                )
                if f < best_f:
                    best_f = f
                    best_result = verts

        # --- Two-phase candidates ---
        for section in self.binodal_sections:
            # phis shape: (2, 2, N) — [phase, component, index]
            phi_a = section.phis[0]  # (2, N)
            phi_b = section.phis[1]  # (2, N)

            d = phi_b - phi_a  # (2, N)
            diff = phi_means[:, None] - phi_a  # (2, N)

            # Project phi_means onto each tie line: t = dot(diff, d) / dot(d, d)
            num = np.sum(diff * d, axis=0)  # (N,)
            den = np.sum(d * d, axis=0)  # (N,)

            valid = den > 1e-20
            if not np.any(valid):
                continue

            t = np.full(den.shape, np.nan)
            t[valid] = num[valid] / den[valid]

            # Distance from phi_means to its projection on each tie line segment
            t_clamped = np.clip(t, 0.0, 1.0)
            closest = phi_a + t_clamped[None, :] * d  # (2, N)
            dist = np.linalg.norm(phi_means[:, None] - closest, axis=0)  # (N,)

            # Only consider tie lines where phi_means is within the segment
            candidates = valid & (t >= 0.0) & (t <= 1.0)
            if not np.any(candidates):
                continue

            dist[~candidates] = np.inf
            idx = int(np.argmin(dist))

            if not np.isfinite(dist[idx]):
                continue

            # Interpolated free energy at phi_means on this tie line
            alpha = t[idx]
            f = (1.0 - alpha) * self._free_energy(
                phi_a[:, idx]
            ) + alpha * self._free_energy(phi_b[:, idx])

            if f < best_f:
                best_f = f
                best_result = section.phis[:, :, idx]

        if best_result is not None:
            return best_result, best_f
        else:
            return None, None

    def plot_sections(self, **kwargs):
        """Plot every stored binodal section on the current axes.

        Parameters
        ----------
        **kwargs
            Plot options forwarded to :meth:`BinodalSection.plot`. The first
            section is labeled ``"Binodal curve"``.

        Returns
        -------
        None
        """
        for i, section in enumerate(self.binodal_sections):
            if i == 0:
                section.plot(label="Binodal curve", **kwargs)
            else:
                section.plot(**kwargs)

    def plot_polygons(self, **kwargs):
        """Plot the two- and three-phase regions on the current axes.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to
            :func:`matplotlib.pyplot.fill`.

        Returns
        -------
        None

        Notes
        -----
        Missing polygon collections are constructed automatically.
        Three-phase regions are filled in blue and two-phase regions in
        orange, both with an opacity of ``0.5``. This plotting helper expects
        each stored geometry to expose a polygon ``exterior``.
        """
        if self.three_phase_polygons is None or self.two_phase_polygons is None:
            self._find_phase_polygons()

        for i, tp_poly in enumerate(self.three_phase_polygons):
            x, y = tp_poly.exterior.xy
            if i == 0:
                plt.fill(
                    x, y, color="blue", alpha=0.5, label="3-phase region", **kwargs
                )
            else:
                plt.fill(x, y, color="blue", alpha=0.5, **kwargs)

        for i, tp_poly in enumerate(self.two_phase_polygons):
            x, y = tp_poly.exterior.xy
            if i == 0:
                plt.fill(
                    x, y, color="orange", alpha=0.5, label="2-phase region", **kwargs
                )
            else:
                plt.fill(x, y, color="orange", alpha=0.5, **kwargs)
                plt.fill(x, y, color="orange", alpha=0.5, **kwargs)
