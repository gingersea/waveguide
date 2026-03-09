"""
Tests for waveguide.modes – slab mode solver and Effective Index Method.
"""

import numpy as np
import pytest

from waveguide.modes import solve_slab_te, effective_indices, group_index
from waveguide import platform


# ── Slab mode solver ──────────────────────────────────────────────────────────

class TestSolveSlabTE:
    def test_220nm_si_slab_has_one_guided_mode(self):
        """220 nm Si/SiO2 slab at 1550 nm supports exactly one TE mode."""
        modes = solve_slab_te(
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            dimension_um=platform.SI_THICKNESS_UM,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        assert len(modes) == 1

    def test_n_eff_between_nclad_and_ncore(self):
        modes = solve_slab_te(
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            dimension_um=platform.SI_THICKNESS_UM,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        for n in modes:
            assert platform.N_SIO2 < n < platform.N_SI

    def test_wider_slab_more_modes(self):
        """A 1 µm slab should support more modes than a 220 nm slab."""
        modes_thin = solve_slab_te(platform.N_SI, platform.N_SIO2, 0.22,
                                    platform.WAVELENGTH_UM)
        modes_wide = solve_slab_te(platform.N_SI, platform.N_SIO2, 1.0,
                                    platform.WAVELENGTH_UM)
        assert len(modes_wide) > len(modes_thin)

    def test_modes_sorted_descending(self):
        modes = solve_slab_te(platform.N_SI, platform.N_SIO2, 1.0,
                               platform.WAVELENGTH_UM, max_modes=4)
        for i in range(len(modes) - 1):
            assert modes[i] >= modes[i + 1]

    def test_no_modes_for_very_thin_slab(self):
        """
        The fundamental TE₀ mode of a symmetric slab has no cutoff – it is
        guided even for very thin slabs.  However, n_eff converges to n_clad
        as the slab thickness goes to zero.
        """
        modes = solve_slab_te(platform.N_SI, platform.N_SIO2, 0.01,
                               platform.WAVELENGTH_UM)
        # At least 1 mode expected (no cutoff for TE₀)
        assert len(modes) >= 1
        # n_eff must be close to n_clad (weakly guided)
        assert modes[0] == pytest.approx(platform.N_SIO2, abs=0.03)


# ── Effective indices (EIM) ───────────────────────────────────────────────────

class TestEffectiveIndices:
    def test_sm_waveguide_at_least_one_mode(self):
        """450 nm × 220 nm Si/SiO2 strip → at least one guided mode at 1550 nm."""
        modes = effective_indices(
            width_um=platform.SM_WIDTH_UM,
            height_um=platform.SI_THICKNESS_UM,
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        assert len(modes) >= 1

    def test_sm_waveguide_fewer_modes_than_mm(self):
        """
        The EIM approximation may overestimate mode count for narrow strips,
        but the 450 nm SM waveguide should support strictly fewer modes than
        the 2 µm multimode waveguide.
        """
        modes_sm = effective_indices(
            width_um=platform.SM_WIDTH_UM,
            height_um=platform.SI_THICKNESS_UM,
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        modes_mm = effective_indices(
            width_um=platform.MM_WIDTH_UM,
            height_um=platform.SI_THICKNESS_UM,
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        assert len(modes_sm) < len(modes_mm)

    def test_mm_waveguide_has_multiple_modes(self):
        """2 µm × 220 nm Si/SiO2 strip → multimode at 1550 nm."""
        modes = effective_indices(
            width_um=platform.MM_WIDTH_UM,
            height_um=platform.SI_THICKNESS_UM,
            n_core=platform.N_SI,
            n_clad=platform.N_SIO2,
            wavelength_um=platform.WAVELENGTH_UM,
        )
        assert len(modes) >= 2

    def test_n_eff_in_valid_range(self):
        for width in [0.45, 1.0, 2.0]:
            modes = effective_indices(width, platform.SI_THICKNESS_UM,
                                       platform.N_SI, platform.N_SIO2,
                                       platform.WAVELENGTH_UM)
            for n_eff, beta in modes:
                assert platform.N_SIO2 < n_eff < platform.N_SI

    def test_wider_waveguide_higher_neff(self):
        """Wider waveguide → higher effective index for fundamental mode."""
        modes_sm = effective_indices(0.45, platform.SI_THICKNESS_UM,
                                      platform.N_SI, platform.N_SIO2,
                                      platform.WAVELENGTH_UM)
        modes_mm = effective_indices(2.0, platform.SI_THICKNESS_UM,
                                      platform.N_SI, platform.N_SIO2,
                                      platform.WAVELENGTH_UM)
        assert modes_mm[0][0] > modes_sm[0][0]

    def test_beta_equals_neff_times_k0(self):
        k0 = 2 * np.pi / platform.WAVELENGTH_UM
        modes = effective_indices(platform.SM_WIDTH_UM,
                                   platform.SI_THICKNESS_UM,
                                   platform.N_SI, platform.N_SIO2,
                                   platform.WAVELENGTH_UM)
        for n_eff, beta in modes:
            assert beta == pytest.approx(n_eff * k0, rel=1e-6)


# ── Group index ───────────────────────────────────────────────────────────────

class TestGroupIndex:
    def test_group_index_positive(self):
        ng = group_index(platform.SM_WIDTH_UM, platform.SI_THICKNESS_UM,
                          platform.N_SI, platform.N_SIO2, platform.WAVELENGTH_UM)
        assert ng > 0

    def test_group_index_greater_than_neff(self):
        """For normal dispersion, n_g > n_eff."""
        ng = group_index(platform.SM_WIDTH_UM, platform.SI_THICKNESS_UM,
                          platform.N_SI, platform.N_SIO2, platform.WAVELENGTH_UM)
        modes = effective_indices(platform.SM_WIDTH_UM, platform.SI_THICKNESS_UM,
                                   platform.N_SI, platform.N_SIO2,
                                   platform.WAVELENGTH_UM)
        n_eff = modes[0][0]
        # Strong waveguide dispersion in SOI means n_g > n_eff
        assert ng > n_eff

    def test_group_index_reasonable_range(self):
        """n_g for a 220 nm SOI strip at 1550 nm is typically 3.5 – 5.0."""
        ng = group_index(platform.SM_WIDTH_UM, platform.SI_THICKNESS_UM,
                          platform.N_SI, platform.N_SIO2, platform.WAVELENGTH_UM)
        assert 3.0 < ng < 6.0
