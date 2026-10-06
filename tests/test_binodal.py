import numpy as np
import pytest
import shapely

from pyphasediagram.binodal import Binodal, BinodalSection
from pyphasediagram.point import CriticalPoint


# TODO: Check these unit tests more thuroughly and make comments
# such as "Helpers" universal throughout project.
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_section(phi_a, phi_b):
    """Create a BinodalSection from two (2, N) arrays of compositions."""
    phi_a = np.asarray(phi_a)
    phi_b = np.asarray(phi_b)
    phis = np.stack([phi_a, phi_b], axis=0)  # (2, 2, N)
    svs = np.zeros(phis.shape[2])
    return BinodalSection(phis, svs)


def _simple_section():
    """A simple section: two straight-line branches."""
    t = np.linspace(0, 1, 50)
    phi_a = np.stack([0.1 + 0.3 * t, 0.2 + 0.1 * t], axis=0)  # (2, 50)
    phi_b = np.stack([0.5 + 0.2 * t, 0.1 + 0.3 * t], axis=0)
    return _make_section(phi_a, phi_b)


def _crossing_sections():
    """Two sections whose branches cross, forming a three-phase region."""
    t = np.linspace(0, 1, 100)
    # Section 1: branches along phi1 direction
    sec1_a = np.stack([0.1 + 0.5 * t, 0.05 * np.ones_like(t)], axis=0)
    sec1_b = np.stack([0.1 + 0.5 * t, 0.38 * np.ones_like(t)], axis=0)
    # Section 2: branches along phi2 direction
    sec2_a = np.stack([0.15 * np.ones_like(t), 0.1 + 0.4 * t], axis=0)
    sec2_b = np.stack([0.48 * np.ones_like(t), 0.1 + 0.4 * t], axis=0)
    return _make_section(sec1_a, sec1_b), _make_section(sec2_a, sec2_b)


# ---------------------------------------------------------------------------
# BinodalSection tests
# ---------------------------------------------------------------------------
class TestBinodalSection:

    def test_init_shapes(self):
        sec = _simple_section()
        assert sec.phis.shape == (2, 2, 50)
        assert sec.svs.shape == (50,)

    def test_init_rejects_incompatible_phis_and_svs_lengths(self):
        phis = _simple_section().phis
        svs = np.zeros(phis.shape[2] - 1)

        with pytest.raises(ValueError, match="same number of trace points"):
            BinodalSection(phis, svs)

    def test_lines_are_linestrings(self):
        sec = _simple_section()
        assert isinstance(sec.line_a, shapely.LineString)
        assert isinstance(sec.line_b, shapely.LineString)

    def test_len(self):
        sec = _simple_section()
        assert len(sec) == 50

    def test_contained_in_self(self):
        sec = _simple_section()
        assert sec.contained_in(sec)

    def test_not_contained_in_different(self):
        sec1 = _simple_section()
        t = np.linspace(0, 1, 50)
        phi_a = np.stack([0.8 - 0.3 * t, 0.05 + 0.1 * t], axis=0)
        phi_b = np.stack([0.01 + 0.2 * t, 0.7 + 0.1 * t], axis=0)
        sec2 = _make_section(phi_a, phi_b)
        assert not sec1.contained_in(sec2)

    def test_contained_in_rejects_non_section(self):
        sec = _simple_section()
        with pytest.raises(NotImplementedError):
            sec.contained_in("not a section")

    def test_intersects_with_rejects_non_section(self):
        sec = _simple_section()
        with pytest.raises(NotImplementedError):
            sec.intersects_with("not a section")

    def test_intersects_with_non_crossing(self):
        """Non-overlapping sections should return no intersection points."""
        t = np.linspace(0, 1, 50)
        sec1 = _make_section(
            np.stack([0.05 + 0.1 * t, 0.05 * np.ones_like(t)], axis=0),
            np.stack([0.05 + 0.1 * t, 0.1 * np.ones_like(t)], axis=0),
        )
        sec2 = _make_section(
            np.stack([0.7 + 0.1 * t, 0.05 * np.ones_like(t)], axis=0),
            np.stack([0.7 + 0.1 * t, 0.1 * np.ones_like(t)], axis=0),
        )
        assert sec1.intersects_with(sec2) == []

    def test_intersects_with_crossing(self):
        """Crossing sections should produce intersection points (three-phase candidates)."""
        sec1, sec2 = _crossing_sections()
        tp_points = sec1.intersects_with(sec2)
        assert len(tp_points) > 0
        for tri in tp_points:
            assert len(tri) == 3
            for vertex in tri:
                assert vertex.shape == (2,)

    def test_degenerate_points_empty_for_flat_svs(self):
        """When svs is constant (no minima), degenerate points list should be empty."""
        sec = _simple_section()
        assert sec._degenerate_points() == []


# ---------------------------------------------------------------------------
# Binodal (unit-level) tests
# ---------------------------------------------------------------------------
class TestBinodalUnit:

    @pytest.fixture
    def chis(self):
        return np.array([[-5.0, 0.5], [0.5, -5.5]])

    def test_init(self, chis):
        b = Binodal(chis)
        assert np.allclose(b.chis, chis)
        assert b.binodal_sections == []
        assert b.three_phase_polygons is None
        assert b.two_phase_polygons is None

    def test_init_with_critical_points(self, chis):
        cp = CriticalPoint(-1, 0.3, 0.3, 1.0, -1.0)
        b = Binodal(chis, critical_points=[cp])
        assert len(b.critical_points) == 1
        assert b.critical_points[0] is cp

    def test_find_three_phase_polygons_empty(self, chis):
        """No sections → no three-phase polygons."""
        b = Binodal(chis)
        b._find_three_phase_polygons()
        assert b.three_phase_polygons == []

    def test_find_two_phase_polygons_empty(self, chis):
        """No sections → no two-phase polygons."""
        b = Binodal(chis)
        b.three_phase_polygons = []
        b._find_two_phase_polygons()
        assert b.two_phase_polygons == []

    def test_find_two_phase_polygons_single_section(self, chis):
        """A single section should produce one two-phase polygon."""
        b = Binodal(chis)
        b.binodal_sections = [_simple_section()]
        b.three_phase_polygons = []
        b._find_two_phase_polygons()
        assert len(b.two_phase_polygons) == 1
        assert b.two_phase_polygons[0].area > 0

    def test_find_three_phase_polygons_with_crossing(self, chis):
        """Crossing sections should produce at least one three-phase polygon."""
        b = Binodal(chis)
        sec1, sec2 = _crossing_sections()
        b.binodal_sections = [sec1, sec2]
        b._find_three_phase_polygons()
        assert len(b.three_phase_polygons) > 0
        for poly in b.three_phase_polygons:
            assert poly.area > 0

    def test_find_two_phase_polygons_subtracts_three_phase(self, chis):
        """Two-phase polygons should not overlap with three-phase polygons."""
        b = Binodal(chis)
        sec1, sec2 = _crossing_sections()
        b.binodal_sections = [sec1, sec2]
        b._find_three_phase_polygons()
        b._find_two_phase_polygons()
        for tp in b.two_phase_polygons:
            for three in b.three_phase_polygons:
                intersection = tp.intersection(three)
                assert intersection.area < 1e-6

    def test_find_phase_polygons_calls_both(self, chis):
        """_find_phase_polygons should populate both polygon lists."""
        b = Binodal(chis)
        b.binodal_sections = [_simple_section()]
        b._find_phase_polygons()
        assert b.three_phase_polygons is not None
        assert b.two_phase_polygons is not None

    def test_remove_unstable_sections_noop_without_manifold(self, chis):
        """Passing None should not remove any sections."""
        b = Binodal(chis)
        sec = _simple_section()
        b.binodal_sections = [sec]
        b._remove_unstable_sections(None)
        assert len(b.binodal_sections) == 1

    def test_remove_unstable_sections_filters(self, chis):
        """Sections fully inside the unstable manifold should be removed."""
        b = Binodal(chis)
        sec = _simple_section()
        b.binodal_sections = [sec]
        # Create a polygon that encloses the entire section
        manifold = shapely.Polygon([(0, 0), (1, 0), (0, 1)])
        b._remove_unstable_sections(manifold)
        assert len(b.binodal_sections) == 0

    def test_remove_unstable_sections_partial(self, chis):
        """Sections partially inside should be clipped to stable segments."""
        b = Binodal(chis)
        t = np.linspace(0, 1, 100)
        phi_a = np.stack([0.1 + 0.8 * t, 0.05 * np.ones_like(t)], axis=0)
        phi_b = np.stack([0.1 + 0.8 * t, 0.1 * np.ones_like(t)], axis=0)
        sec = _make_section(phi_a, phi_b)
        b.binodal_sections = [sec]
        # Unstable region covers only the left half
        manifold = shapely.Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])
        b._remove_unstable_sections(manifold)
        # Should still have sections (the right part survives)
        assert len(b.binodal_sections) >= 1
        # All surviving points should be outside the manifold
        for sec in b.binodal_sections:
            for i in range(sec.phis.shape[2]):
                p1 = sec.phis[0, :, i]
                p2 = sec.phis[1, :, i]
                assert not (
                    manifold.contains(shapely.Point(p1))
                    and manifold.contains(shapely.Point(p2))
                )

    def test_remove_unstable_sections_splits_into_two(self, chis):
        """An unstable manifold cutting through the middle should split a section into two."""
        b = Binodal(chis)
        t = np.linspace(0, 1, 100)
        phi_a = np.stack([0.1 + 0.8 * t, 0.05 * np.ones_like(t)], axis=0)
        phi_b = np.stack([0.1 + 0.8 * t, 0.1 * np.ones_like(t)], axis=0)
        sec = _make_section(phi_a, phi_b)
        b.binodal_sections = [sec]
        # Unstable region covers the middle band
        manifold = shapely.Polygon([(0.2, 0), (0.5, 0), (0.5, 0.5), (0.2, 0.5)])
        b._remove_unstable_sections(manifold)
        assert len(b.binodal_sections) == 2

    def test_remove_unstable_sections_asymmetric_indices(self, chis):
        """When phi_a and phi_b traverse different x-ranges, they enter the
        unstable manifold at different indices.  The union of the two
        unstable index sets should still produce two stable segments."""
        b = Binodal(chis)
        t = np.linspace(0, 1, 100)
        # phi_a spans a wide x-range, phi_b a narrower one
        phi_a = np.stack([0.05 + 0.9 * t, 0.05 * np.ones_like(t)], axis=0)
        phi_b = np.stack([0.25 + 0.5 * t, 0.1 * np.ones_like(t)], axis=0)
        sec = _make_section(phi_a, phi_b)
        b.binodal_sections = [sec]
        manifold = shapely.Polygon([(0.3, 0), (0.6, 0), (0.6, 0.5), (0.3, 0.5)])

        # Verify the premise: inside_a and inside_b differ in their indices
        inside_a = shapely.contains_xy(manifold, phi_a[0], phi_a[1])
        inside_b = shapely.contains_xy(manifold, phi_b[0], phi_b[1])
        assert not np.array_equal(inside_a, inside_b)

        b._remove_unstable_sections(manifold)
        assert len(b.binodal_sections) == 2

        boundary = manifold.exterior
        step = 0.9 / 99  # spacing between consecutive phi_a x-values
        for sec in b.binodal_sections:
            # The clipped end of each section should be near the manifold boundary
            dists = []
            for end_idx in (0, -1):
                pa = shapely.Point(sec.phis[0, :, end_idx])
                pb = shapely.Point(sec.phis[1, :, end_idx])
                dists.append(min(boundary.distance(pa), boundary.distance(pb)))
            assert (
                min(dists) < 2 * step
            ), f"No endpoint close to manifold boundary: dists={dists}"

    def test_duplicate_three_phase_polygons_filtered(self, chis):
        """Near-duplicate three-phase polygons should be deduplicated."""
        b = Binodal(chis)
        sec1, sec2 = _crossing_sections()
        b.binodal_sections = [sec1, sec2]
        b._find_three_phase_polygons()
        n_polygons = len(b.three_phase_polygons)
        # Running again should not add more polygons
        b._find_three_phase_polygons()
        assert len(b.three_phase_polygons) == n_polygons


# ---------------------------------------------------------------------------
# End-to-end integration tests
# ---------------------------------------------------------------------------
class TestBinodalEndToEnd:
    """Integration tests that build full spinodal + binodal from chi matrices."""

    @pytest.fixture(scope="class")
    def symmetric_chis(self):
        """Symmetric chi matrix: chi_01 = chi_02 = chi_12 = chi."""
        chi = 2.75
        chi_11 = -2 * chi
        chi_22 = -2 * chi
        chi_12 = chi - chi - chi  # = -chi
        return np.array([[chi_11, chi_12], [chi_12, chi_22]])

    @pytest.fixture
    def asymmetric_chis(self):
        """Asymmetric chi matrix."""
        chi_01, chi_02, chi_12 = 2.65, 2.85, 2.75
        chi_11 = -2 * chi_01
        chi_22 = -2 * chi_02
        chi_12_r = chi_12 - chi_01 - chi_02
        return np.array([[chi_11, chi_12_r], [chi_12_r, chi_22]])

    def _build_binodal(self, chis):
        from pyphasediagram.spinodal import Spinodal

        spinodal = Spinodal(chis)
        spinodal.build()
        binodal = Binodal(chis, spinodal.critical_points)
        binodal.build(unstable_manifold=spinodal.get_unstable_manifold())
        return binodal

    @pytest.fixture(scope="class")
    def symmetric_binodal(self, symmetric_chis):
        """Build the shared symmetric phase diagram once for this test class."""
        return self._build_binodal(symmetric_chis)

    def test_symmetric_produces_sections(self, symmetric_binodal):
        assert len(symmetric_binodal.binodal_sections) > 0

    def test_symmetric_sections_have_valid_shapes(self, symmetric_binodal):
        for sec in symmetric_binodal.binodal_sections:
            assert sec.phis.shape[0] == 2  # two phases
            assert sec.phis.shape[1] == 2  # two components
            assert sec.phis.shape[2] >= 2  # at least 2 points
            assert sec.svs.shape[0] == sec.phis.shape[2]

    def test_compositions_in_domain(self, symmetric_binodal):
        """All binodal compositions should lie inside the Gibbs triangle."""
        for sec in symmetric_binodal.binodal_sections:
            for phase in range(2):
                phi1 = sec.phis[phase, 0]
                phi2 = sec.phis[phase, 1]
                assert np.all(phi1 >= -1e-10), f"phi1 negative: {phi1.min()}"
                assert np.all(phi2 >= -1e-10), f"phi2 negative: {phi2.min()}"
                assert np.all(
                    phi1 + phi2 <= 1 + 1e-10
                ), f"phi1+phi2 > 1: {(phi1+phi2).max()}"

    def test_phase_polygons_populated(self, symmetric_binodal):
        assert symmetric_binodal.three_phase_polygons is not None
        assert symmetric_binodal.two_phase_polygons is not None

    def test_two_phase_no_overlap_with_three_phase(self, symmetric_binodal):
        for tp in symmetric_binodal.two_phase_polygons:
            for three in symmetric_binodal.three_phase_polygons:
                assert tp.intersection(three).area < 1e-6

    def test_asymmetric_produces_sections(self, asymmetric_chis):
        binodal = self._build_binodal(asymmetric_chis)
        assert len(binodal.binodal_sections) > 0

    def test_three_phase_polygons_inside_domain(self, symmetric_binodal):
        """Three-phase polygon vertices should be inside the Gibbs triangle."""
        domain = shapely.Polygon([(0, 0), (1, 0), (0, 1)])
        for poly in symmetric_binodal.three_phase_polygons:
            assert domain.contains(poly) or domain.intersection(
                poly
            ).area == pytest.approx(poly.area, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_chi_builds_without_error(self, seed):
        """Random chi matrices in the typical range should build without raising."""
        rng = np.random.default_rng(seed)
        chi = 2.75
        spread = 0.4
        chi_01, chi_02, chi_12 = (rng.random(3) - 0.5) * spread + chi
        chi_11 = -2 * chi_01
        chi_22 = -2 * chi_02
        chi_12_r = chi_12 - chi_01 - chi_02
        chis = np.array([[chi_11, chi_12_r], [chi_12_r, chi_22]])
        # Should not raise
        binodal = self._build_binodal(chis)
        assert binodal.three_phase_polygons is not None
        assert binodal.two_phase_polygons is not None

    # def test_stable_parts_preserved_with_manifold(self, symmetric_chis):
    #     """Stable portions of sections built without the manifold should be
    #     contained in the sections of the binodal built with the manifold."""
    #     from pyphasediagram.spinodal import Spinodal

    #     spinodal = Spinodal(symmetric_chis)
    #     spinodal.build()
    #     manifold = spinodal.get_unstable_manifold()
    #     assert manifold is not None

    #     # Build with and without unstable manifold
    #     b_with = Binodal(symmetric_chis, spinodal.critical_points)
    #     b_with.build(unstable_manifold=manifold)

    #     b_without = Binodal(symmetric_chis, spinodal.critical_points)
    #     b_without.build(unstable_manifold=None)

    #     # Collect all 4D points (phi_a1, phi_a2, phi_b1, phi_b2) from
    #     # the with-manifold sections for fast nearest-neighbour lookup
    #     all_pts_with = []
    #     for sec in b_with.binodal_sections:
    #         pts = np.concatenate([sec.phis[0], sec.phis[1]], axis=0)  # (4, N)
    #         all_pts_with.append(pts.T)
    #     all_pts_with = np.concatenate(all_pts_with, axis=0)  # (M, 4)

    #     tol = 1e-3
    #     for sec in b_without.binodal_sections:
    #         inside_a = shapely.contains_xy(manifold, sec.phis[0, 0], sec.phis[0, 1])
    #         inside_b = shapely.contains_xy(manifold, sec.phis[1, 0], sec.phis[1, 1])
    #         stable_mask = ~(inside_a | inside_b)
    #         stable_idx = np.where(stable_mask)[0]
    #         if len(stable_idx) == 0:
    #             continue

    #         for i in stable_idx:
    #             pt = np.concatenate([sec.phis[0, :, i], sec.phis[1, :, i]])
    #             dists = np.linalg.norm(all_pts_with - pt, axis=1)
    #             assert dists.min() < tol, (
    #                 f"Stable point {pt} not found in with-manifold sections "
    #                 f"(min dist = {dists.min():.6f})"
    #             )
