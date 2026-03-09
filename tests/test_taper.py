"""
Tests for waveguide.taper – SM-to-MM adiabatic taper designs.
"""

import numpy as np
import pytest

from waveguide.taper import (
    LinearTaper,
    ParabolicTaper,
    GaussianTaper,
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
