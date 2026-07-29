import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely

from pyphasediagram.point import BinodalInitialPoint, CriticalPoint
from pyphasediagram.stepper import Stepper

# TODO: Additional duplicate section removal after task list is exhausted and unstable manifold is removed.


class BinodalSection:

    def __init__(self, phis: np.ndarray, svs: np.ndarray):
        self.phis = phis
        self.svs = svs
        self.line_a = shapely.LineString(phis[0].T)
        self.line_b = shapely.LineString(phis[1].T)
        self._lines = (self.line_a, self.line_b)
        self._coords = tuple(np.asarray(line.coords) for line in self._lines)

    def plot(self, colora="red", colorb="blue", is_tie_lines=False, **kwargs):
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
        """Find the nearest line segment and return (seg_index, t).

        ``seg_index`` is the start-vertex index of the closest segment and
        ``t`` in [0, 1] is the interpolation parameter along that segment.
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
        """Check if this binodal section intersects with another binodal section by checking if their corresponding lines intersect using shapely."""
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
        self.three_phase_polygons: list[shapely.Polygon] = None
        self.two_phase_polygons: list[shapely.Polygon] = None
        self._stepper = Stepper(chis)
        self._bipt_task_list: list[BinodalInitialPoint] = []
        self._bipt_hist: list[BinodalInitialPoint] = []

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

        self._remove_unstable_sections(unstable_manifold)
        self._remove_duplicate_sections()
        self._find_phase_polygons()

    def _remove_duplicate_sections(self):
        """Remove duplicate binodal sections where one is contained in another."""
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

    def _find_phase_polygons(self):
        self._find_three_phase_polygons()
        self._find_two_phase_polygons()

    def _find_three_phase_polygons(self, overlap_threshold: float = 0.98):
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
        """Compute the dimensionless free energy density for a composition [phi1, phi2]."""
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
        """Given mean composition, return the coexisting phase compositions that
        the system decomposes into.

        If ``phi_means`` lies inside a three-phase polygon **and** that
        decomposition has lower free energy than any two-phase tie line, the
        three vertices of the polygon are returned.  Otherwise the best
        two-phase tie line through ``phi_means`` is returned.

        Parameters
        ----------
        phi_means : np.ndarray
            Mean composition [phi1, phi2].

        Returns
        -------
        alt : np.ndarray or None
            Coexisting phase compositions. Shape (2, 3) for three-phase, (2, 2) for two-phase, or None if no decomposition found.
        f_alt : float or None
            Free energy of the decomposition. None if no decomposition found.
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
        for i, section in enumerate(self.binodal_sections):
            if i == 0:
                section.plot(label="Binodal curve", **kwargs)
            else:
                section.plot(**kwargs)

    def plot_polygons(self, **kwargs):
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
