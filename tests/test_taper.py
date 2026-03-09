"""
Tests for waveguide.taper – SM-to-MM adiabatic taper designs.
"""

import numpy as np
import pytest

from waveguide.taper import (
    LinearTaper,
    ParabolicTaper,
    GaussianTaper,
    OptimalAdiabaticTaper,
    SubwavelengthGratingTaper,
    swg_effective_index,
    compare_tapers,
    minimum_adiabatic_length,
)
from waveguide import platform


# ── Width profile sanity checks ───────────────────────────────────────────────

class TestWidthProfiles:
    """Each taper starts at w_start_um and ends at w_end_um."""

    @pytest.fixture(params=["linear", "parabolic", "gaussian"])
    def taper(self, request):
        tapers = compare_tapers(length_um=50.0)
        return tapers[request.param]

    def test_start_width(self, taper):
        w0 = float(taper.width_profile(0.0))
        assert w0 == pytest.approx(platform.SM_WIDTH_UM, rel=1e-6)

    def test_end_width(self, taper):
        wL = float(taper.width_profile(taper.length_um))
        assert wL == pytest.approx(platform.MM_WIDTH_UM, rel=1e-6)

    def test_monotone_increasing(self, taper):
        z = np.linspace(0, taper.length_um, 200)
        w = np.array([float(taper.width_profile(zi)) for zi in z])
        # Allow numerical noise: differences should be >= 0
        assert np.all(np.diff(w) >= -1e-12)

    def test_within_bounds(self, taper):
        z = np.linspace(0, taper.length_um, 100)
        w = np.array([float(taper.width_profile(zi)) for zi in z])
        assert np.all(w >= platform.SM_WIDTH_UM - 1e-9)
        assert np.all(w <= platform.MM_WIDTH_UM + 1e-9)


# ── LinearTaper specific ──────────────────────────────────────────────────────

class TestLinearTaper:
    def test_midpoint_width(self):
        t = LinearTaper(length_um=100.0)
        w_mid = float(t.width_profile(50.0))
        expected = (platform.SM_WIDTH_UM + platform.MM_WIDTH_UM) / 2
        assert w_mid == pytest.approx(expected, rel=1e-6)

    def test_slope_constant(self):
        t = LinearTaper(length_um=100.0)
        z = np.linspace(1, 99, 50)
        w = np.array([float(t.width_profile(zi)) for zi in z])
        slopes = np.diff(w) / np.diff(z)
        assert np.std(slopes) < 1e-9 * np.mean(slopes)


# ── ParabolicTaper specific ───────────────────────────────────────────────────

class TestParabolicTaper:
    def test_zero_slope_at_start(self):
        """dw/dz should be (nearly) zero at z = 0 for the parabolic taper."""
        t = ParabolicTaper(length_um=100.0)
        dz = 0.01
        dw_dz = (float(t.width_profile(dz)) - float(t.width_profile(0.0))) / dz
        assert abs(dw_dz) < 1e-3  # very small slope at the narrow end


# ── GaussianTaper specific ────────────────────────────────────────────────────

class TestGaussianTaper:
    def test_symmetric_about_midpoint(self):
        """
        The erf-based profile should be symmetric: the width change in the
        first half equals the width change in the second half.
        """
        t = GaussianTaper(length_um=100.0, sigma=2.0)
        w0 = float(t.width_profile(0.0))
        wL = float(t.width_profile(t.length_um))
        wM = float(t.width_profile(t.length_um / 2))
        # midpoint should be equidistant from both ends
        assert abs((wM - w0) - (wL - wM)) < 1e-6

    def test_sigma_effect(self):
        """Higher sigma should give a steeper central slope."""
        t1 = GaussianTaper(length_um=100.0, sigma=1.0)
        t2 = GaussianTaper(length_um=100.0, sigma=3.0)
        dz = 0.1
        z_mid = 50.0
        slope1 = abs(float(t1.width_profile(z_mid + dz)) -
                     float(t1.width_profile(z_mid - dz))) / (2 * dz)
        slope2 = abs(float(t2.width_profile(z_mid + dz)) -
                     float(t2.width_profile(z_mid - dz))) / (2 * dz)
        assert slope2 > slope1


# ── Adiabaticity ──────────────────────────────────────────────────────────────

class TestAdiabaticity:
    @pytest.mark.parametrize("length_um", [50.0, 100.0])
    def test_alpha_positive_everywhere(self, length_um):
        """Adiabaticity parameter must be non-negative."""
        for cls in (LinearTaper, ParabolicTaper, GaussianTaper):
            t = cls(length_um=length_um)
            _, alpha = t.adiabaticity(n_points=50)
            assert np.all(alpha >= 0)

    def test_longer_taper_more_adiabatic(self):
        """A longer linear taper should have a lower max α."""
        t_short = LinearTaper(length_um=20.0)
        t_long = LinearTaper(length_um=100.0)
        _, a_short = t_short.adiabaticity(n_points=50)
        _, a_long = t_long.adiabaticity(n_points=50)
        assert np.max(a_long) < np.max(a_short)

    def test_parabolic_better_than_linear_same_length(self):
        """
        At the same taper length, the parabolic profile (dw/dz = 0 at the
        narrow end) starts the width transition more gently than the linear
        profile, which is beneficial in the single-mode region.  Verify that
        the parabolic taper has zero slope at the narrow end and that linear
        has a constant non-zero slope everywhere.
        """
        length = 30.0
        t_lin = LinearTaper(length_um=length)
        t_par = ParabolicTaper(length_um=length)
        # Linear has constant slope; parabolic has zero slope at z=0
        dz = 0.001
        slope_par_start = abs(
            float(t_par.width_profile(dz)) - float(t_par.width_profile(0.0))
        ) / dz
        slope_lin_start = abs(
            float(t_lin.width_profile(dz)) - float(t_lin.width_profile(0.0))
        ) / dz
        assert slope_par_start < slope_lin_start


# ── Insertion loss ────────────────────────────────────────────────────────────

class TestInsertionLoss:
    def test_il_non_negative(self):
        for cls in (LinearTaper, ParabolicTaper, GaussianTaper):
            t = cls(length_um=50.0)
            assert t.insertion_loss_db() >= 0.0

    def test_il_decreases_with_length(self):
        """Longer taper → lower IL (more adiabatic)."""
        t_short = GaussianTaper(length_um=20.0)
        t_long = GaussianTaper(length_um=80.0)
        assert t_long.insertion_loss_db() <= t_short.insertion_loss_db()

    def test_coupling_efficiency_bounds(self):
        for cls in (LinearTaper, ParabolicTaper, GaussianTaper):
            t = cls(length_um=50.0)
            eta = t.coupling_efficiency()
            assert 0.0 <= eta <= 1.0


# ── Minimum adiabatic length ──────────────────────────────────────────────────

class TestMinimumAdiabaticLength:
    @pytest.mark.parametrize("taper_type", ["linear", "parabolic", "gaussian"])
    def test_min_length_positive(self, taper_type):
        length = minimum_adiabatic_length(taper_type=taper_type,
                                          length_max_um=200.0, n_search=20)
        assert length > 0

    def test_all_three_improve_with_length(self):
        """All taper profiles must yield a shorter minimum adiabatic length when
        the search ceiling is large, i.e. there exists a length that satisfies
        the criterion for each profile type."""
        for taper_type in ["linear", "parabolic", "gaussian"]:
            L = minimum_adiabatic_length(taper_type, length_max_um=200.0,
                                          n_search=20)
            assert L <= 200.0

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            minimum_adiabatic_length("invalid_type")


# ── compare_tapers ────────────────────────────────────────────────────────────

class TestCompareTapers:
    def test_returns_all_three(self):
        tapers = compare_tapers(length_um=50.0)
        assert set(tapers) == {"linear", "parabolic", "gaussian"}

    def test_correct_types(self):
        tapers = compare_tapers(length_um=50.0)
        assert isinstance(tapers["linear"], LinearTaper)
        assert isinstance(tapers["parabolic"], ParabolicTaper)
        assert isinstance(tapers["gaussian"], GaussianTaper)

    def test_summary_nonempty(self):
        tapers = compare_tapers(length_um=50.0)
        for t in tapers.values():
            s = t.summary()
            assert len(s) > 0
            assert "Taper type" in s


# ── swg_effective_index ────────────────────────────────────────────────────────

class TestSWGEffectiveIndex:
    def test_te_full_fill_equals_core(self):
        """ff = 1  →  n_TE = n_core."""
        n = swg_effective_index(1.0, platform.N_SI, platform.N_SIO2, "TE")
        assert n == pytest.approx(platform.N_SI, rel=1e-6)

    def test_te_zero_fill_equals_clad(self):
        """ff = 0  →  n_TE = n_clad."""
        n = swg_effective_index(0.0, platform.N_SI, platform.N_SIO2, "TE")
        assert n == pytest.approx(platform.N_SIO2, rel=1e-6)

    def test_tm_full_fill_equals_core(self):
        n = swg_effective_index(1.0, platform.N_SI, platform.N_SIO2, "TM")
        assert n == pytest.approx(platform.N_SI, rel=1e-6)

    def test_tm_zero_fill_equals_clad(self):
        n = swg_effective_index(0.0, platform.N_SI, platform.N_SIO2, "TM")
        assert n == pytest.approx(platform.N_SIO2, rel=1e-6)

    def test_te_between_indices_for_mid_fill(self):
        n = swg_effective_index(0.5, platform.N_SI, platform.N_SIO2, "TE")
        assert platform.N_SIO2 < n < platform.N_SI

    def test_te_greater_than_tm(self):
        """Wiener bounds: n_TE ≥ n_TM for any fill factor."""
        for ff in [0.2, 0.5, 0.8]:
            n_te = swg_effective_index(ff, platform.N_SI, platform.N_SIO2, "TE")
            n_tm = swg_effective_index(ff, platform.N_SI, platform.N_SIO2, "TM")
            assert n_te >= n_tm

    def test_te_monotone_with_fill_factor(self):
        """n_TE must increase monotonically with fill factor."""
        ffs = np.linspace(0.1, 0.9, 9)
        ns = [swg_effective_index(f, platform.N_SI, platform.N_SIO2, "TE") for f in ffs]
        assert all(ns[i] < ns[i + 1] for i in range(len(ns) - 1))

    def test_tm_monotone_with_fill_factor(self):
        ffs = np.linspace(0.1, 0.9, 9)
        ns = [swg_effective_index(f, platform.N_SI, platform.N_SIO2, "TM") for f in ffs]
        assert all(ns[i] < ns[i + 1] for i in range(len(ns) - 1))

    def test_array_input(self):
        """Function must handle array inputs and return an array."""
        ffs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = swg_effective_index(ffs, platform.N_SI, platform.N_SIO2, "TE")
        assert hasattr(result, "__len__")
        assert len(result) == 5

    def test_invalid_polarization_raises(self):
        with pytest.raises(ValueError):
            swg_effective_index(0.5, platform.N_SI, platform.N_SIO2, "XY")


# ── OptimalAdiabaticTaper ──────────────────────────────────────────────────────

class TestOptimalAdiabaticTaper:
    def test_start_width(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        assert float(t.width_profile(0.0)) == pytest.approx(
            platform.SM_WIDTH_UM, rel=1e-4)

    def test_end_width(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        assert float(t.width_profile(30.0)) == pytest.approx(
            platform.MM_WIDTH_UM, rel=1e-4)

    def test_monotone_increasing(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        z = np.linspace(0, 30.0, 100)
        w = np.array([float(t.width_profile(zi)) for zi in z])
        assert np.all(np.diff(w) >= -1e-9)

    def test_within_bounds(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        z = np.linspace(0, 30.0, 100)
        w = np.array([float(t.width_profile(zi)) for zi in z])
        assert np.all(w >= platform.SM_WIDTH_UM - 1e-6)
        assert np.all(w <= platform.MM_WIDTH_UM + 1e-6)

    def test_natural_length_positive(self):
        t = OptimalAdiabaticTaper(alpha_target=0.05)
        assert t.natural_length_um > 0.0

    def test_smaller_alpha_gives_longer_natural_length(self):
        """Stricter adiabaticity (smaller α) requires a longer taper."""
        t_loose = OptimalAdiabaticTaper(alpha_target=0.1)
        t_strict = OptimalAdiabaticTaper(alpha_target=0.05)
        assert t_strict.natural_length_um > t_loose.natural_length_um

    def test_effective_alpha_at_natural_length(self):
        """effective_alpha == alpha_target when length_um == natural_length_um."""
        t_ref = OptimalAdiabaticTaper(alpha_target=0.05)
        t_nat = OptimalAdiabaticTaper(
            alpha_target=0.05, length_um=t_ref.natural_length_um)
        assert t_nat.effective_alpha == pytest.approx(0.05, rel=1e-3)

    def test_constant_adiabaticity_profile(self):
        """
        α(z) should be nearly uniform – the coefficient of variation must
        be below 0.3 (i.e., 30 %).
        """
        t = OptimalAdiabaticTaper(alpha_target=0.05, length_um=50.0)
        _, alpha = t.adiabaticity(n_points=50)
        cv = np.std(alpha) / (np.mean(alpha) + 1e-12)
        assert cv < 0.30

    def test_insertion_loss_non_negative(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        assert t.insertion_loss_db() >= 0.0

    def test_coupling_efficiency_in_bounds(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        eta = t.coupling_efficiency()
        assert 0.0 <= eta <= 1.0

    def test_shorter_natural_length_than_linear(self):
        """
        The OptimalAdiabaticTaper natural length at alpha_target = 0.1 must be
        strictly shorter than the minimum adiabatic length of a LinearTaper at
        the same threshold — demonstrating the compactness advantage from the
        'Efficient adiabatic SOI waveguide taper' paper.
        """
        t_opt = OptimalAdiabaticTaper(alpha_target=0.1)
        L_lin = minimum_adiabatic_length(
            "linear", length_max_um=200.0, n_search=30)
        assert t_opt.natural_length_um < L_lin

    def test_longer_length_more_adiabatic(self):
        """Scaling length_um upward reduces effective_alpha."""
        t_short = OptimalAdiabaticTaper(alpha_target=0.05, length_um=20.0)
        t_long = OptimalAdiabaticTaper(alpha_target=0.05, length_um=60.0)
        assert t_long.effective_alpha < t_short.effective_alpha

    def test_summary_contains_taper_type(self):
        t = OptimalAdiabaticTaper(length_um=30.0)
        assert "OptimalAdiabaticTaper" in t.summary()


# ── SubwavelengthGratingTaper ──────────────────────────────────────────────────

class TestSubwavelengthGratingTaper:
    def test_start_width(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        assert float(t.width_profile(0.0)) == pytest.approx(
            platform.SM_WIDTH_UM, rel=1e-4)

    def test_end_width(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        assert float(t.width_profile(10.0)) == pytest.approx(
            platform.MM_WIDTH_UM, rel=1e-4)

    def test_ff_start_matches_width_ratio(self):
        t = SubwavelengthGratingTaper()
        assert t.ff_start == pytest.approx(
            platform.SM_WIDTH_UM / platform.MM_WIDTH_UM, rel=1e-6)

    def test_ff_at_start_equals_ff_start(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        assert float(t.fill_factor(0.0)) == pytest.approx(t.ff_start, rel=1e-5)

    def test_ff_at_end_is_unity(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        assert float(t.fill_factor(10.0)) == pytest.approx(1.0, rel=1e-6)

    def test_ff_monotone_increasing(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        z = np.linspace(0, 10.0, 50)
        ff = np.array([float(t.fill_factor(zi)) for zi in z])
        assert np.all(np.diff(ff) >= -1e-12)

    def test_width_profile_monotone(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        z = np.linspace(0, 10.0, 50)
        w = np.array([float(t.width_profile(zi)) for zi in z])
        assert np.all(np.diff(w) >= -1e-12)

    def test_n_segments_matches_period(self):
        t = SubwavelengthGratingTaper(length_um=10.0, period_um=0.200)
        assert t.n_segments == 50

    def test_segment_fill_factors_length(self):
        t = SubwavelengthGratingTaper(length_um=10.0, period_um=0.200)
        ff_arr = t.segment_fill_factors()
        assert len(ff_arr) == t.n_segments

    def test_segment_fill_factors_bounds(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        ff_arr = t.segment_fill_factors()
        assert np.all(ff_arr >= t.ff_start - 1e-9)
        assert np.all(ff_arr <= 1.0 + 1e-9)

    def test_effective_core_index_at_full_fill(self):
        t = SubwavelengthGratingTaper()
        assert t.effective_core_index(1.0) == pytest.approx(platform.N_SI, rel=1e-4)

    def test_effective_core_index_at_zero_fill(self):
        t = SubwavelengthGratingTaper()
        assert t.effective_core_index(0.0) == pytest.approx(
            platform.N_SIO2, rel=1e-4)

    def test_is_subwavelength_default_period(self):
        t = SubwavelengthGratingTaper(period_um=0.200)
        assert t.is_subwavelength()

    def test_not_subwavelength_for_large_period(self):
        t = SubwavelengthGratingTaper(period_um=0.500)
        assert not t.is_subwavelength()

    @pytest.mark.parametrize("profile", ["linear", "parabolic", "gaussian"])
    def test_all_ff_profiles_have_correct_boundaries(self, profile):
        t = SubwavelengthGratingTaper(length_um=10.0, ff_profile=profile)
        assert float(t.width_profile(0.0)) == pytest.approx(
            platform.SM_WIDTH_UM, rel=1e-4)
        assert float(t.width_profile(10.0)) == pytest.approx(
            platform.MM_WIDTH_UM, rel=1e-4)

    def test_invalid_ff_profile_raises(self):
        t = SubwavelengthGratingTaper(length_um=10.0, ff_profile="invalid")
        with pytest.raises(ValueError):
            t.fill_factor(5.0)

    def test_summary_contains_swg_info(self):
        t = SubwavelengthGratingTaper(length_um=10.0)
        s = t.summary()
        assert "SubwavelengthGratingTaper" in s
        assert "Period" in s
        assert "N segments" in s


# ── compare_tapers (advanced) ──────────────────────────────────────────────────

class TestCompareAdvancedTapers:
    def test_include_advanced_adds_optimal_and_swg(self):
        tapers = compare_tapers(length_um=30.0, include_advanced=True)
        assert "optimal" in tapers
        assert "swg" in tapers
        assert isinstance(tapers["optimal"], OptimalAdiabaticTaper)
        assert isinstance(tapers["swg"], SubwavelengthGratingTaper)

    def test_default_does_not_include_advanced(self):
        """Backward-compatible: include_advanced defaults to False."""
        tapers = compare_tapers(length_um=30.0)
        assert "optimal" not in tapers
        assert "swg" not in tapers


# ── minimum_adiabatic_length (new types) ──────────────────────────────────────

class TestMinimumAdiabaticLengthAdvanced:
    @pytest.mark.parametrize("taper_type", ["optimal", "swg"])
    def test_min_length_positive(self, taper_type):
        L = minimum_adiabatic_length(
            taper_type=taper_type, length_max_um=200.0, n_search=15)
        assert L > 0.0

    def test_optimal_min_length_shorter_than_linear(self):
        """
        The equi-adiabatic taper achieves adiabaticity in a shorter length
        than the linear taper for the same threshold (0.1).
        """
        L_opt = minimum_adiabatic_length(
            "optimal", length_max_um=100.0, n_search=20)
        L_lin = minimum_adiabatic_length(
            "linear", length_max_um=100.0, n_search=20)
        assert L_opt < L_lin
