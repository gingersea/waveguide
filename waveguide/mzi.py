"""
Asymmetric Mach-Zehnder Interferometer (MZI) design.

Layout
------
The MZI has two arms connected by a pair of 50:50 directional couplers:

  Input ──[DC1]──┬── SM ──[taper]── MM (long arm, ΔL) ──[taper]── SM ──┬──[DC2]── Output 1
                 │                                                       │
                 └──────────── SM (short arm) ─────────────────────────┘── Output 2

  DC  = 50:50 directional coupler (single-mode region)
  SM  = single-mode waveguide (450 nm width)
  MM  = multimode waveguide (2 µm width, low-loss)
  ΔL  = extra length in the long arm (multimode section)

Key design goal: replace the long single-mode routing section with a
multimode waveguide to reduce propagation loss, while keeping the
directional coupler region single-mode.  SM-to-MM transitions are handled
by adiabatic tapers (see waveguide.taper).

Classes
-------
AsymmetricMZI
    Full MZI layout and loss / spectral response calculator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import platform
from .taper import GaussianTaper, _BaseTaper
from .modes import effective_indices, group_index


@dataclass
class AsymmetricMZI:
    """
    Asymmetric MZI with low-loss multimode long arm.

    Parameters
    ----------
    delta_l_um : path length difference ΔL (µm).  The long arm contains an
                 extra section of multimode waveguide of this length plus the
                 mode-converter tapers.
    sm_arm_extra_um : extra single-mode routing length in each arm outside
                      the taper/MM section (µm).  Default 100 µm each side.
    taper : the SM-to-MM adiabatic taper object (default GaussianTaper with
            50 µm length).  The same taper geometry is used for both the
            SM→MM and MM→SM transitions.
    sm_width_um : single-mode waveguide width (µm).
    mm_width_um : multimode waveguide width (µm).
    height_um   : silicon layer thickness (µm).
    n_core      : core refractive index.
    n_clad      : cladding refractive index.
    wavelength_um : design wavelength (µm).
    coupler_il_db : excess insertion loss per directional coupler (dB).
    """

    delta_l_um: float = 1000.0          # 1 mm extra MM length → ΔL

    # Single-mode routing outside the taper pair (per arm)
    sm_arm_extra_um: float = 100.0

    # Taper object (SM→MM at one end, reversed MM→SM at the other)
    taper: Optional[_BaseTaper] = None

    # Platform defaults
    sm_width_um: float = platform.SM_WIDTH_UM
    mm_width_um: float = platform.MM_WIDTH_UM
    height_um: float = platform.SI_THICKNESS_UM
    n_core: float = platform.N_SI
    n_clad: float = platform.N_SIO2
    wavelength_um: float = platform.WAVELENGTH_UM

    # Loss parameters
    loss_sm_db_per_cm: float = platform.LOSS_SM_DB_PER_CM
    loss_mm_db_per_cm: float = platform.LOSS_MM_DB_PER_CM
    coupler_il_db: float = platform.COUPLER_IL_DB
    taper_il_db: Optional[float] = None  # if None, computed from taper object

    def __post_init__(self):
        if self.taper is None:
            self.taper = GaussianTaper(
                w_start_um=self.sm_width_um,
                w_end_um=self.mm_width_um,
                length_um=50.0,
                height_um=self.height_um,
                wavelength_um=self.wavelength_um,
            )
        if self.taper_il_db is None:
            self.taper_il_db = self.taper.insertion_loss_db()

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def long_arm_mm_length_um(self) -> float:
        """Length of the multimode section in the long arm (µm)."""
        return self.delta_l_um

    @property
    def short_arm_sm_length_um(self) -> float:
        """Total single-mode length of the short arm (µm)."""
        return 2 * self.sm_arm_extra_um

    @property
    def long_arm_total_length_um(self) -> float:
        """Total optical path length of the long arm (µm)."""
        return (2 * self.sm_arm_extra_um
                + 2 * self.taper.length_um
                + self.long_arm_mm_length_um)

    # ── Propagation loss ──────────────────────────────────────────────────────

    def _loss_db_per_um(self, loss_db_per_cm: float) -> float:
        return loss_db_per_cm / 1e4

    def _arm_propagation_loss_db(self, sm_len_um: float, mm_len_um: float) -> float:
        return (sm_len_um * self._loss_db_per_um(self.loss_sm_db_per_cm)
                + mm_len_um * self._loss_db_per_um(self.loss_mm_db_per_cm))

    def long_arm_loss_db(self) -> float:
        """Total loss in the long arm (propagation + tapers), in dB."""
        sm_len = 2 * self.sm_arm_extra_um
        mm_len = self.long_arm_mm_length_um
        prop_loss = self._arm_propagation_loss_db(sm_len, mm_len)
        taper_loss = 2 * self.taper_il_db   # SM→MM and MM→SM
        return prop_loss + taper_loss

    def short_arm_loss_db(self) -> float:
        """Total propagation loss in the short arm (single-mode only), in dB."""
        return self._arm_propagation_loss_db(self.short_arm_sm_length_um, 0.0)

    def total_insertion_loss_db(self) -> float:
        """
        Total MZI insertion loss (dB) from input to output at the peak
        transmission wavelength.

        Accounts for:
          - Two directional couplers
          - Propagation loss in both arms (arithmetic mean at the output
            because both arms interfere constructively at the transmission
            peak and the 3 dB splitting is recovered)
          - Taper insertion losses
        """
        coupler_loss = 2 * self.coupler_il_db
        avg_arm_loss = (self.long_arm_loss_db() + self.short_arm_loss_db()) / 2
        return coupler_loss + avg_arm_loss

    def loss_saving_vs_allsm_db(self) -> float:
        """
        Propagation loss saved in the long arm by using the MM waveguide
        instead of a fully single-mode routing, in dB.
        """
        sm_only_loss = self._arm_propagation_loss_db(
            self.long_arm_total_length_um, 0.0)
        mm_assisted_loss = self.long_arm_loss_db()
        return sm_only_loss - mm_assisted_loss

    # ── Spectral response ─────────────────────────────────────────────────────

    def _phase_difference(self, wavelength_um: float) -> float:
        """
        Optical phase difference Δφ between the two arms at *wavelength_um*.

        Δφ = β_long × L_long − β_short × L_short

        Uses the fundamental TE mode effective index of each waveguide width.
        """
        modes_sm = effective_indices(self.sm_width_um, self.height_um,
                                      self.n_core, self.n_clad, wavelength_um)
        modes_mm = effective_indices(self.mm_width_um, self.height_um,
                                      self.n_core, self.n_clad, wavelength_um)
        if not modes_sm or not modes_mm:
            raise ValueError("No guided modes found at wavelength "
                             f"{wavelength_um*1e3:.0f} nm.")

        beta_sm = modes_sm[0][1]  # µm⁻¹
        beta_mm = modes_mm[0][1]

        # Long arm: SM sections + MM section (tapers treated as SM for phase)
        taper_len = self.taper.length_um
        phi_long = (beta_sm * (2 * self.sm_arm_extra_um + 2 * taper_len)
                    + beta_mm * self.long_arm_mm_length_um)
        # Short arm: pure SM
        phi_short = beta_sm * self.short_arm_sm_length_um
        return phi_long - phi_short

    def transmission(
        self,
        wavelengths_um: Optional[np.ndarray] = None,
        n_points: int = 500,
        wl_range_um: Tuple[float, float] = (1.52, 1.58),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the MZI through-port and cross-port transmission spectra.

        Assumes ideal 50:50 couplers (lossless splitting) for the spectral
        calculation; overall insertion loss is added separately.

        Parameters
        ----------
        wavelengths_um : array of wavelengths (µm); if None, a grid over
                         wl_range_um with n_points is used.
        n_points       : number of wavelength points (used when wavelengths_um
                         is None).
        wl_range_um    : (min, max) wavelength range (µm).

        Returns
        -------
        (wl, T_through, T_cross) all shape (N,).
        T_through + T_cross = 1 (lossless 50:50 assumption for spectral shape).
        """
        if wavelengths_um is None:
            wavelengths_um = np.linspace(wl_range_um[0], wl_range_um[1],
                                          n_points)
        T_through = np.empty_like(wavelengths_um)
        T_cross = np.empty_like(wavelengths_um)

        for i, wl in enumerate(wavelengths_um):
            try:
                dphi = self._phase_difference(wl)
                T_through[i] = np.cos(dphi / 2) ** 2
                T_cross[i] = np.sin(dphi / 2) ** 2
            except ValueError:
                T_through[i] = np.nan
                T_cross[i] = np.nan

        return wavelengths_um, T_through, T_cross

    # ── FSR ───────────────────────────────────────────────────────────────────

    def free_spectral_range_um(self) -> float:
        """
        Estimate the MZI free spectral range (FSR) in µm:

          FSR ≈ λ² / (n_g_mm × ΔL_mm + n_g_sm × ΔL_sm)

        where ΔL_mm = extra multimode length and ΔL_sm = extra single-mode
        contribution.  Here ΔL_sm ≈ 0 because the extra path length is
        placed entirely in the multimode section; the short arm matches the
        SM portions of the long arm.

        A more accurate calculation uses the wavelength derivative of Δφ.
        """
        ng_mm = group_index(self.mm_width_um, self.height_um,
                             self.n_core, self.n_clad, self.wavelength_um)
        ng_sm = group_index(self.sm_width_um, self.height_um,
                             self.n_core, self.n_clad, self.wavelength_um)
        # The short arm has the same SM sections as the long arm (matched);
        # net OPD comes from the MM section plus the SM taper sections.
        # Tapers are approximated as SM waveguides at their midpoint width.
        taper_mid_w = (self.sm_width_um + self.mm_width_um) / 2
        ng_taper = group_index(taper_mid_w, self.height_um,
                                self.n_core, self.n_clad, self.wavelength_um)
        opd = (ng_mm * self.long_arm_mm_length_um
               + ng_taper * 2 * self.taper.length_um
               - ng_sm * (2 * self.taper.length_um))  # subtract SM reference
        if opd <= 0:
            opd = ng_mm * self.long_arm_mm_length_um
        return float(self.wavelength_um**2 / opd)

    def free_spectral_range_thz(self) -> float:
        """FSR in THz."""
        c_um_per_s = 2.998e14  # speed of light in µm/s
        fsr_um = self.free_spectral_range_um()
        # Convert: Δν ≈ c/λ² × FSR_λ
        return float(c_um_per_s / self.wavelength_um**2 * fsr_um * 1e-12)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the MZI design."""
        lines = [
            "━" * 60,
            "Asymmetric MZI – Low-Loss Design Summary",
            "━" * 60,
            f"  Platform     : ANT NanoSOI 220 nm Si / SiO₂",
            f"  Wavelength   : {self.wavelength_um*1e3:.0f} nm",
            "",
            "  Waveguide widths",
            f"    Single-mode : {self.sm_width_um*1e3:.0f} nm",
            f"    Multimode   : {self.mm_width_um*1e3:.0f} nm",
            "",
            "  Taper (SM → MM)",
            f"    Type        : {type(self.taper).__name__}",
            f"    Length      : {self.taper.length_um:.1f} µm",
            f"    IL per taper: {self.taper_il_db:.3f} dB",
            "",
            "  Arm lengths",
            f"    Short arm   : {self.short_arm_sm_length_um:.0f} µm (SM only)",
            f"    Long arm    : {self.long_arm_total_length_um:.0f} µm total",
            f"      └ SM extra    : {2*self.sm_arm_extra_um:.0f} µm",
            f"      └ Tapers (×2) : {2*self.taper.length_um:.0f} µm",
            f"      └ MM section  : {self.long_arm_mm_length_um:.0f} µm",
            "",
            "  Loss budget",
            f"    Short arm   : {self.short_arm_loss_db():.3f} dB",
            f"    Long arm    : {self.long_arm_loss_db():.3f} dB",
            f"    Couplers    : {2*self.coupler_il_db:.3f} dB",
            f"    Total IL    : {self.total_insertion_loss_db():.3f} dB",
            f"    Loss saving : {self.loss_saving_vs_allsm_db():.3f} dB "
            f"vs. all-SM routing",
            "",
            "  Spectral",
            f"    FSR         : {self.free_spectral_range_um()*1e3:.2f} nm  "
            f"({self.free_spectral_range_thz():.3f} THz)",
            "━" * 60,
        ]
        return "\n".join(lines)
