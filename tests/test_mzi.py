"""
Tests for waveguide.mzi – Asymmetric MZI with low-loss multimode long arm.
"""

import numpy as np
import pytest

from waveguide.mzi import AsymmetricMZI
from waveguide.taper import GaussianTaper, LinearTaper, ParabolicTaper
from waveguide import platform


# ── Default MZI ───────────────────────────────────────────────────────────────

@pytest.fixture
def default_mzi():
    return AsymmetricMZI(delta_l_um=1000.0)


# ── Geometry checks ───────────────────────────────────────────────────────────

class TestMZIGeometry:
    def test_long_arm_longer_than_short(self, default_mzi):
        assert default_mzi.long_arm_total_length_um > default_mzi.short_arm_sm_length_um

    def test_mm_section_equals_delta_l(self, default_mzi):
        assert default_mzi.long_arm_mm_length_um == pytest.approx(
            default_mzi.delta_l_um, rel=1e-9)

    def test_long_arm_total_includes_tapers(self, default_mzi):
        expected = (2 * default_mzi.sm_arm_extra_um
                    + 2 * default_mzi.taper.length_um
                    + default_mzi.long_arm_mm_length_um)
        assert default_mzi.long_arm_total_length_um == pytest.approx(expected, rel=1e-9)


# ── Loss budget ───────────────────────────────────────────────────────────────

class TestMZILoss:
    def test_propagation_loss_positive(self, default_mzi):
        assert default_mzi.long_arm_loss_db() > 0
        assert default_mzi.short_arm_loss_db() > 0

    def test_long_arm_lossier_than_short(self, default_mzi):
        """Long arm is longer so it should have more propagation loss."""
        assert default_mzi.long_arm_loss_db() > default_mzi.short_arm_loss_db()

    def test_total_il_positive(self, default_mzi):
        assert default_mzi.total_insertion_loss_db() > 0

    def test_mm_arm_less_loss_than_allsm(self):
        """
        Using MM waveguide must save propagation loss vs. a fully single-mode
        routing of the same length.
        """
        mzi = AsymmetricMZI(delta_l_um=5000.0)  # 5 mm – large saving
        saving = mzi.loss_saving_vs_allsm_db()
        assert saving > 0

    def test_loss_saving_scales_with_delta_l(self):
        """More MM length → more loss saving."""
        mzi1 = AsymmetricMZI(delta_l_um=1000.0)
        mzi2 = AsymmetricMZI(delta_l_um=5000.0)
        assert mzi2.loss_saving_vs_allsm_db() > mzi1.loss_saving_vs_allsm_db()

    def test_longer_taper_reduces_il(self):
        """
        For any taper profile, a longer taper must have a lower insertion loss.
        This is a fundamental property: as length → ∞ the taper becomes
        perfectly adiabatic.
        """
        for cls in (LinearTaper, ParabolicTaper, GaussianTaper):
            t_short = cls(length_um=30.0)
            t_long = cls(length_um=100.0)
            assert t_long.insertion_loss_db() < t_short.insertion_loss_db()

    def test_all_taper_types_have_positive_il(self):
        """All taper types must have a non-negative insertion loss."""
        for cls in (LinearTaper, ParabolicTaper, GaussianTaper):
            t = cls(length_um=50.0)
            assert t.insertion_loss_db() >= 0


# ── FSR ───────────────────────────────────────────────────────────────────────

class TestMZIFSR:
    def test_fsr_positive(self, default_mzi):
        fsr = default_mzi.free_spectral_range_um()
        assert fsr > 0

    def test_fsr_in_expected_range(self):
        """
        For ΔL = 1 mm and n_g ~ 4.3 (rough estimate for Si strip at 1550 nm),
        FSR ≈ λ² / (n_g × ΔL) ≈ 1.55² / (4.3 × 1000) µm ≈ 0.56 nm.
        Check that FSR is between 0.1 nm and 5 nm for ΔL = 1 mm.
        """
        mzi = AsymmetricMZI(delta_l_um=1000.0)
        fsr_nm = mzi.free_spectral_range_um() * 1e3
        assert 0.1 < fsr_nm < 5.0

    def test_fsr_inversely_proportional_to_delta_l(self):
        """Doubling ΔL should roughly halve the FSR."""
        mzi1 = AsymmetricMZI(delta_l_um=1000.0)
        mzi2 = AsymmetricMZI(delta_l_um=2000.0)
        fsr1 = mzi1.free_spectral_range_um()
        fsr2 = mzi2.free_spectral_range_um()
        # Allow 20% tolerance due to dispersion
        assert fsr2 == pytest.approx(fsr1 / 2, rel=0.2)

    def test_fsr_thz_positive(self, default_mzi):
        assert default_mzi.free_spectral_range_thz() > 0


# ── Transmission spectrum ─────────────────────────────────────────────────────

class TestMZITransmission:
    def test_spectrum_shape(self, default_mzi):
        wl, T_thru, T_cross = default_mzi.transmission(n_points=100)
        assert wl.shape == T_thru.shape == T_cross.shape
        assert len(wl) == 100

    def test_power_conservation(self, default_mzi):
        """T_through + T_cross should equal 1 (lossless spectral model)."""
        _, T_thru, T_cross = default_mzi.transmission(n_points=100)
        total = T_thru + T_cross
        # Allow NaN at boundary points; check the rest
        valid = ~np.isnan(total)
        assert np.all(np.abs(total[valid] - 1.0) < 1e-10)

    def test_transmission_between_0_and_1(self, default_mzi):
        _, T_thru, T_cross = default_mzi.transmission(n_points=100)
        for arr in (T_thru, T_cross):
            valid = ~np.isnan(arr)
            assert np.all(arr[valid] >= -1e-12)
            assert np.all(arr[valid] <= 1.0 + 1e-12)

    def test_spectrum_has_oscillations(self):
        """The transmission spectrum should show MZI fringes."""
        mzi = AsymmetricMZI(delta_l_um=500.0)
        _, T_thru, _ = mzi.transmission(n_points=300)
        valid = ~np.isnan(T_thru)
        # Count sign changes in (T_thru - 0.5) to detect oscillations
        centred = T_thru[valid] - 0.5
        sign_changes = np.sum(np.diff(np.sign(centred)) != 0)
        assert sign_changes >= 2  # at least one full fringe


# ── Summary string ────────────────────────────────────────────────────────────

class TestMZISummary:
    def test_summary_nonempty(self, default_mzi):
        s = default_mzi.summary()
        assert len(s) > 0

    def test_summary_contains_key_info(self, default_mzi):
        s = default_mzi.summary()
        assert "FSR" in s
        assert "Loss" in s or "loss" in s
        assert "GaussianTaper" in s

    def test_custom_taper_reflected_in_summary(self):
        taper = LinearTaper(length_um=30.0)
        mzi = AsymmetricMZI(delta_l_um=500.0, taper=taper)
        s = mzi.summary()
        assert "LinearTaper" in s
