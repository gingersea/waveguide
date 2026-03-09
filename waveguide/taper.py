"""
Single-mode to multimode (SM-to-MM) adiabatic taper designs.

Five taper geometries are provided:

  LinearTaper              – width increases linearly with z.
  ParabolicTaper           – width follows a quadratic (parabolic) profile.
  GaussianTaper            – width follows a Gaussian-sigmoid (erf) profile.
  OptimalAdiabaticTaper    – equi-adiabatic profile (constant α(z));
                             shortest possible taper for a given loss budget.
                             Based on: "Efficient adiabatic silicon-on-insulator
                             waveguide taper".
  SubwavelengthGratingTaper – discrete-pixel sub-wavelength grating (SWG)
                             taper using effective medium theory (EMT);
                             enables ultra-compact designs.
                             Based on: "Adiabatic and Ultracompact Waveguide
                             Tapers Based on Digital Metamaterials".

Each taper class exposes:
  width_profile(z)          – width (µm) as a function of position z (µm).
  adiabaticity(z_points)    – local adiabaticity parameter α(z); the taper
                              is fully adiabatic when α(z) << 1 everywhere.
  coupling_efficiency()     – fraction of input power remaining in the
                              fundamental mode at the output, estimated
                              from the coupled-mode overlap integral.
  insertion_loss_db()       – taper insertion loss (dB).
  summary()                 – human-readable parameter table.

A standalone helper is also provided:
  swg_effective_index(ff, n_core, n_clad, polarization)
      Zeroth-order effective medium theory (EMT) index for a SWG composite.

Adiabaticity criterion (after Snyder & Love, 1983):
  α(z) = |dw/dz| / (Δβ(w) × w_local)
where Δβ(w) = β₀(w) − β₁(w) is the local differential propagation
constant between the fundamental (TE₀₀) and the first higher-order (TE₁₀)
mode.  A taper is considered adiabatic when α(z) < 0.1 everywhere.

Platform defaults correspond to ANT NanoSOI (220 nm Si / SiO2 cladding,
λ = 1550 nm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .modes import effective_indices
from . import platform


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delta_beta(
    width_um: float,
    height_um: float,
    n_core: float,
    n_clad: float,
    wavelength_um: float,
) -> float:
    """
    Return Δβ = β₀ − β₁ (µm⁻¹) at width *width_um*.

    If fewer than two modes are guided, a small residual value is returned
    so the adiabaticity ratio remains finite and dominated by |dw/dz|.
    """
    modes = effective_indices(width_um, height_um, n_core, n_clad,
                               wavelength_um, max_modes=2)
    if len(modes) < 2:
        # Only fundamental mode guided – treat Δβ as the radiation loss rate
        k0 = 2 * np.pi / wavelength_um
        return k0 * (modes[0][0] - n_clad) if modes else 0.1
    return float(modes[0][1] - modes[1][1])


# ── Base class ────────────────────────────────────────────────────────────────

@dataclass
class _BaseTaper:
    """Shared infrastructure for all taper geometries."""

    # Geometry
    w_start_um: float = platform.SM_WIDTH_UM   # narrow end (single-mode side)
    w_end_um: float = platform.MM_WIDTH_UM     # wide end (multimode side)
    length_um: float = 50.0                    # taper length (µm)

    # Platform defaults
    height_um: float = platform.SI_THICKNESS_UM
    n_core: float = platform.N_SI
    n_clad: float = platform.N_SIO2
    wavelength_um: float = platform.WAVELENGTH_UM

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        """Width w(z) in µm. Must be overridden by subclasses."""
        raise NotImplementedError

    # ── Adiabaticity ─────────────────────────────────────────────────────────

    def adiabaticity(
        self, z_points: Optional[np.ndarray] = None, n_points: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the local adiabaticity parameter α(z).

        α(z) = |dw/dz| / (Δβ(w(z)) × w(z))

        Parameters
        ----------
        z_points  : 1-D array of z positions (µm); if None, n_points
                    equally-spaced values over [0, length_um] are used.
        n_points  : number of sample points when z_points is None.

        Returns
        -------
        (z_arr, alpha_arr) both shape (N,)
        """
        if z_points is None:
            z_points = np.linspace(0, self.length_um, n_points)

        # Use local point spacing for the derivative; fall back to a uniform
        # step when only one point is provided.
        z_arr = np.asarray(z_points, dtype=float)
        n = len(z_arr)
        alpha = np.empty(n)
        for i, z in enumerate(z_arr):
            # Central-difference step: use half the spacing to the nearest
            # neighbour so the derivative stays within [0, length_um].
            if n > 1:
                h_left = z - z_arr[i - 1] if i > 0 else z_arr[1] - z_arr[0]
                h_right = z_arr[i + 1] - z if i < n - 1 else z_arr[-1] - z_arr[-2]
                dz_local = (h_left + h_right) / 2
            else:
                dz_local = self.length_um / 200  # fallback for single point
            z_lo = max(z - dz_local / 2, 0.0)
            z_hi = min(z + dz_local / 2, self.length_um)
            actual_dz = z_hi - z_lo if z_hi > z_lo else dz_local
            w = float(self.width_profile(z))
            dw_dz = float(
                (self.width_profile(z_hi) - self.width_profile(z_lo)) / actual_dz
            )
            db = _delta_beta(w, self.height_um, self.n_core, self.n_clad,
                              self.wavelength_um)
            # Guard against division by zero
            denominator = db * w if db * w > 1e-12 else 1e-12
            alpha[i] = abs(dw_dz) / denominator
        return z_arr, alpha

    def is_adiabatic(self, threshold: float = 0.1) -> bool:
        """Return True if max α(z) < threshold everywhere along the taper."""
        _, alpha = self.adiabaticity()
        return bool(np.max(alpha) < threshold)

    # ── Loss estimate ─────────────────────────────────────────────────────────

    def coupling_efficiency(self, n_points: int = 200) -> float:
        """
        Estimate the fraction of input power remaining in the fundamental
        mode at the taper output.

        The mode mismatch accumulated along the taper is integrated via a
        first-order coupled-mode approximation:

          η = exp(-π × ∫₀ᴸ α(z)² dz / L)

        This is a conservative estimate; real adiabatic tapers typically
        perform better.
        """
        z_arr, alpha = self.adiabaticity(n_points=n_points)
        # numpy renamed trapz → trapezoid in version 2.0; support both.
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        eta = float(np.exp(-np.pi * _trapz(alpha**2, z_arr) / self.length_um))
        return min(max(eta, 0.0), 1.0)

    def insertion_loss_db(self) -> float:
        """
        Return the taper insertion loss in dB:
          IL = -10 log₁₀(η)
        """
        eta = self.coupling_efficiency()
        if eta <= 0:
            return float("inf")
        return float(-10.0 * np.log10(eta))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the taper parameters."""
        _, alpha = self.adiabaticity()
        eta = self.coupling_efficiency()
        il = self.insertion_loss_db()
        lines = [
            f"Taper type  : {type(self).__name__}",
            f"Width range : {self.w_start_um*1e3:.0f} nm → "
            f"{self.w_end_um*1e3:.0f} nm",
            f"Length      : {self.length_um:.1f} µm",
            f"Height      : {self.height_um*1e3:.0f} nm Si",
            f"Wavelength  : {self.wavelength_um*1e3:.0f} nm",
            f"Max α(z)    : {np.max(alpha):.4f} "
            f"({'adiabatic' if np.max(alpha) < 0.1 else 'NOT adiabatic'})",
            f"Coupling η  : {eta*100:.2f} %",
            f"Insert. loss: {il:.3f} dB",
        ]
        return "\n".join(lines)


# ── Linear taper ──────────────────────────────────────────────────────────────

@dataclass
class LinearTaper(_BaseTaper):
    """
    Width increases linearly with position:

      w(z) = w_start + (w_end - w_start) × z / L

    This is the simplest geometry and requires the longest length to
    satisfy the adiabaticity condition, because dw/dz is constant while
    Δβ is smallest (and coupling strongest) at the narrow end.
    """

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(z, dtype=float) / self.length_um
        return self.w_start_um + (self.w_end_um - self.w_start_um) * t


# ── Parabolic taper ───────────────────────────────────────────────────────────

@dataclass
class ParabolicTaper(_BaseTaper):
    """
    Width follows a parabolic profile:

      w(z) = w_start + (w_end - w_start) × (z / L)²

    The taper rate dw/dz = 0 at z = 0 and is maximum at z = L.
    This concentrates the width change near the wide (multimode) end where
    Δβ is large and inter-modal coupling is weaker, enabling a shorter
    device than the linear taper.
    """

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(z, dtype=float) / self.length_um
        return self.w_start_um + (self.w_end_um - self.w_start_um) * t**2


# ── Gaussian taper ────────────────────────────────────────────────────────────

@dataclass
class GaussianTaper(_BaseTaper):
    """
    Width follows a Gaussian-sigmoid (error-function) profile:

      w(z) = w_start + (w_end - w_start) × [erf(σ × (2z/L - 1)) + 1] / 2

    where σ controls the steepness.  This produces a smooth S-shaped
    profile that limits dw/dz near the narrow end (where Δβ is small and
    mode coupling is strongest) and compresses the design length compared
    with the linear taper.  σ = 2 gives a moderately smooth transition;
    increasing σ sharpens the central transition and approaches the step-like
    limit.

    Parameters
    ----------
    sigma : shape parameter (dimensionless), default 2.0.
    """

    sigma: float = 2.0

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        from scipy.special import erf
        t = np.asarray(z, dtype=float) / self.length_um
        # Unnormalised erf-sigmoid: values at t=0 and t=1 are erf(±sigma)
        erf_0 = erf(-self.sigma)
        erf_1 = erf(self.sigma)
        raw = erf(self.sigma * (2 * t - 1))
        # Normalise so that profile is exactly 0 at t=0 and 1 at t=1
        profile = (raw - erf_0) / (erf_1 - erf_0)
        return self.w_start_um + (self.w_end_um - self.w_start_um) * profile


# ── SWG effective medium index ────────────────────────────────────────────────

def swg_effective_index(
    fill_factor: float | np.ndarray,
    n_core: float = platform.N_SI,
    n_clad: float = platform.N_SIO2,
    polarization: str = "TE",
) -> float | np.ndarray:
    """
    Zeroth-order effective medium theory (EMT) index for a sub-wavelength
    grating (SWG) composite.

    For a grating with fill factor *ff* (fraction of the period occupied by
    the high-index material):

      TE  (E ⊥ grating lines, parallel-plate / series-capacitor model):
          n_TE  = √( ff · n_core² + (1 − ff) · n_clad² )

      TM  (E ∥ grating lines, series / parallel-capacitor model):
          n_TM  = 1 / √( ff / n_core² + (1 − ff) / n_clad² )

    The zeroth-order approximation holds when the grating period
    ``period_um`` ≪ λ / (2 n_core), i.e., no grating diffraction occurs.

    Parameters
    ----------
    fill_factor  : fraction of the grating period occupied by the core
                   material (silicon).  Must be in [0, 1]; a ValueError is
                   raised for out-of-range values.  May be a scalar or array.
    n_core       : refractive index of the core material (default Si at 1550 nm).
    n_clad       : refractive index of the cladding (default SiO₂ at 1550 nm).
    polarization : "TE" (default) or "TM".

    Returns
    -------
    n_eff : effective refractive index of the SWG composite (same shape as
            fill_factor).

    References
    ----------
    Rytov, S. M. (1956). "Electromagnetic properties of a finely stratified
    medium." Soviet Physics JETP, 2(3), 466–475.
    """
    ff_arr = np.asarray(fill_factor, dtype=float)
    if np.any(ff_arr < 0.0) or np.any(ff_arr > 1.0):
        raise ValueError(
            f"fill_factor must be in [0, 1]; got values outside this range."
        )
    ff = ff_arr  # already validated; no clipping needed
    pol = polarization.upper()
    if pol == "TE":
        result = np.sqrt(ff * n_core**2 + (1.0 - ff) * n_clad**2)
    elif pol == "TM":
        result = 1.0 / np.sqrt(ff / n_core**2 + (1.0 - ff) / n_clad**2)
    else:
        raise ValueError(
            f"polarization must be 'TE' or 'TM', got '{polarization}'"
        )
    # Return a Python float when the input was scalar-like
    return float(result) if result.ndim == 0 else result


# ── Optimal (equi-adiabatic) taper ────────────────────────────────────────────

@dataclass
class OptimalAdiabaticTaper(_BaseTaper):
    """
    Equi-adiabatic taper: width profile designed to maintain a constant local
    adiabaticity parameter α(z) = alpha_target throughout the device.

    Based on: "Efficient adiabatic silicon-on-insulator waveguide taper"

    The width profile is the numerical inverse of the integral:

        z(w) = ∫_{w_start}^{w} dw' / ( alpha_target × Δβ(w') × w' )

    which enforces  dw/dz = alpha_target × Δβ(w) × w  at every z.

    This is the *most compact* profile for a given adiabaticity budget: by
    distributing the allowed coupling uniformly along the taper it avoids the
    wasted length of profiles that are over-cautious at some positions and
    stressed at others.

    The stored ``length_um`` scales the z-axis uniformly; the equi-adiabatic
    shape is preserved regardless of the chosen length.

    Parameters
    ----------
    alpha_target : target (uniform) adiabaticity value.  The standard
                   threshold is 0.1; smaller values give lower insertion loss
                   but require longer tapers.  Default 0.05.
    n_w_points   : number of width samples used in the trapezoidal integration
                   of z(w).  Default 200.

    Properties
    ----------
    natural_length_um : length at which α(z) = alpha_target exactly.
                        If length_um > natural_length_um the effective α is
                        proportionally smaller (more adiabatic).
    effective_alpha   : actual uniform α for the stored length_um.
    """

    alpha_target: float = 0.05
    n_w_points: int = 200

    def __post_init__(self) -> None:
        self._build_lookup()

    def _build_lookup(self) -> None:
        """
        Numerically integrate  z(w) = ∫ dw / (alpha_target × Δβ(w) × w)
        and store the resulting w(z) look-up table scaled to length_um.
        """
        w_arr = np.linspace(self.w_start_um, self.w_end_um, self.n_w_points)
        dz_dw = np.empty(self.n_w_points)
        for i, w in enumerate(w_arr):
            db = _delta_beta(
                w, self.height_um, self.n_core, self.n_clad, self.wavelength_um
            )
            denom = self.alpha_target * db * w
            # Guard: skip the z-contribution at widths where Δβ ≈ 0 to avoid
            # division by zero.  The threshold 1e-14 µm⁻² is many orders of
            # magnitude smaller than any physically meaningful Δβ × w product
            # (typically ≥ 0.1 µm⁻²), so it only activates in degenerate cases.
            dz_dw[i] = 1.0 / denom if abs(denom) > 1e-14 else 0.0

        # Cumulative trapezoidal integration
        z_arr = np.zeros(self.n_w_points)
        for i in range(1, self.n_w_points):
            dw = w_arr[i] - w_arr[i - 1]
            z_arr[i] = z_arr[i - 1] + 0.5 * (dz_dw[i] + dz_dw[i - 1]) * dw

        self._natural_length_um: float = float(z_arr[-1])
        # Scale z to fit the caller-specified length_um
        if self._natural_length_um > 0:
            z_arr = z_arr * (self.length_um / self._natural_length_um)
        self._z_lut = z_arr   # z positions (µm) at which each w_arr value occurs
        self._w_lut = w_arr   # corresponding widths (µm)

    @property
    def natural_length_um(self) -> float:
        """Taper length for exactly α(z) = alpha_target everywhere."""
        return self._natural_length_um

    @property
    def effective_alpha(self) -> float:
        """
        Uniform adiabaticity parameter for the stored length_um:

            effective_alpha = alpha_target × natural_length_um / length_um
        """
        if self.length_um > 0:
            return float(
                self.alpha_target * self._natural_length_um / self.length_um
            )
        return self.alpha_target

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        z_clip = np.clip(np.asarray(z, dtype=float), 0.0, self.length_um)
        return np.interp(z_clip, self._z_lut, self._w_lut)


# ── Sub-wavelength grating taper ──────────────────────────────────────────────

@dataclass
class SubwavelengthGratingTaper(_BaseTaper):
    """
    Sub-wavelength grating (SWG) taper using effective medium theory.

    Based on: "Adiabatic and Ultracompact Waveguide Tapers Based on Digital
    Metamaterials"

    The taper consists of ``n_segments`` discrete cells, each of length
    ``period_um`` (the SWG pitch).  At each cell the fraction of silicon
    (fill factor *ff*) varies monotonically from ``ff_start`` at the SM
    input to 1.0 at the MM output:

        ff_start = w_start_um / w_end_um

    The effective silicon width – the quantity used in the mode calculation –
    at position z is:

        w_eff(z) = ff(z) × w_end_um

    which recovers w_start at z = 0 and w_end at z = L exactly.

    At every cross-section the TE-mode effective medium index of the SWG
    composite is:

        n_core_eff(ff) = √( ff · n_Si² + (1 − ff) · n_SiO₂² )

    Because the period is sub-wavelength ( period_um < λ / (2 n_core) ≈ 0.22 µm
    at 1550 nm in silicon ), no grating diffraction occurs and the structure
    behaves as a graded-index medium.  This allows an ultra-compact transition
    footprint compared with conventional smooth tapers.

    Parameters
    ----------
    period_um  : grating pitch (µm).  Default 0.200 µm.  Should satisfy
                 period_um < λ / (2 n_core) ≈ 0.22 µm for the sub-wavelength
                 regime.
    ff_profile : fill-factor profile shape along z: ``"linear"``,
                 ``"parabolic"``, or ``"gaussian"``.  Default ``"linear"``.
    sigma      : Gaussian shape parameter; used only when
                 ``ff_profile="gaussian"``.  Default 2.0.
    """

    period_um: float = 0.200     # SWG pitch (µm)
    ff_profile: str = "linear"   # fill-factor profile
    sigma: float = 2.0           # Gaussian sigma (for ff_profile="gaussian")

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def ff_start(self) -> float:
        """Fill factor at the SM (input) end: w_start_um / w_end_um."""
        return float(self.w_start_um / self.w_end_um)

    @property
    def n_segments(self) -> int:
        """Number of discrete SWG cells that fit within length_um."""
        return max(1, round(self.length_um / self.period_um))

    def is_subwavelength(self) -> bool:
        """
        Return True when period_um < λ / (2 n_core), i.e., the grating
        operates in the zeroth-order regime where EMT is applicable.
        """
        return bool(self.period_um < self.wavelength_um / (2.0 * self.n_core))

    def effective_core_index(self, fill_factor: float) -> float:
        """TE effective medium index for a given fill factor (EMT formula)."""
        return swg_effective_index(fill_factor, self.n_core, self.n_clad, "TE")

    # ── Fill-factor profile ───────────────────────────────────────────────────

    def fill_factor(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Fill factor ff(z) at position z (µm).

        Varies from ff_start at z = 0 to 1.0 at z = length_um according to
        the chosen ff_profile.
        """
        t = (
            np.clip(np.asarray(z, dtype=float), 0.0, self.length_um)
            / self.length_um
        )
        ff_s = self.ff_start
        if self.ff_profile == "linear":
            return ff_s + (1.0 - ff_s) * t
        elif self.ff_profile == "parabolic":
            return ff_s + (1.0 - ff_s) * t**2
        elif self.ff_profile == "gaussian":
            from scipy.special import erf
            erf_0 = erf(-self.sigma)
            erf_1 = erf(self.sigma)
            raw = erf(self.sigma * (2.0 * t - 1.0))
            return ff_s + (1.0 - ff_s) * (raw - erf_0) / (erf_1 - erf_0)
        else:
            raise ValueError(
                f"ff_profile must be 'linear', 'parabolic', or 'gaussian', "
                f"got '{self.ff_profile}'"
            )

    def segment_fill_factors(self) -> np.ndarray:
        """
        Fill factor for each of the ``n_segments`` discrete SWG cells.

        Cell centres are at z = (i + 0.5) × period_um for i = 0, …, N−1.
        """
        z_centres = (np.arange(self.n_segments) + 0.5) * self.period_um
        # fill_factor() accepts arrays directly, avoiding a Python-level loop
        return np.asarray(self.fill_factor(z_centres), dtype=float)

    # ── Width profile (effective silicon width) ───────────────────────────────

    def width_profile(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Effective silicon width at position z:

            w_eff(z) = ff(z) × w_end_um

        This recovers w_start_um at z = 0 and w_end_um at z = length_um.
        """
        return self.fill_factor(z) * self.w_end_um

    # ── Extended summary ──────────────────────────────────────────────────────

    def summary(self) -> str:
        base = super().summary()
        extra = (
            f"\nPeriod      : {self.period_um*1e3:.0f} nm"
            f"\nN segments  : {self.n_segments}"
            f"\nff start    : {self.ff_start:.3f}"
            f"\nSub-λ check : {'OK' if self.is_subwavelength() else 'WARN – period not sub-wavelength'}"
        )
        return base + extra


# ── Taper comparison utility ──────────────────────────────────────────────────

def compare_tapers(
    length_um: float = 50.0,
    w_start_um: float = platform.SM_WIDTH_UM,
    w_end_um: float = platform.MM_WIDTH_UM,
    height_um: float = platform.SI_THICKNESS_UM,
    wavelength_um: float = platform.WAVELENGTH_UM,
    include_advanced: bool = False,
) -> dict[str, _BaseTaper]:
    """
    Instantiate one taper of each geometry and return a dict keyed by name.

    Parameters
    ----------
    include_advanced : if True, also include ``"optimal"``
                       (:class:`OptimalAdiabaticTaper`) and ``"swg"``
                       (:class:`SubwavelengthGratingTaper`) in the result.
                       Default False (backward-compatible).

    Example
    -------
    >>> tapers = compare_tapers(length_um=50)
    >>> for name, t in tapers.items():
    ...     print(name)
    ...     print(t.summary())
    """
    shared = dict(
        w_start_um=w_start_um,
        w_end_um=w_end_um,
        length_um=length_um,
        height_um=height_um,
        wavelength_um=wavelength_um,
    )
    result: dict[str, _BaseTaper] = {
        "linear": LinearTaper(**shared),
        "parabolic": ParabolicTaper(**shared),
        "gaussian": GaussianTaper(**shared),
    }
    if include_advanced:
        result["optimal"] = OptimalAdiabaticTaper(**shared)
        result["swg"] = SubwavelengthGratingTaper(**shared)
    return result


# ── Minimum adiabatic length ──────────────────────────────────────────────────

def minimum_adiabatic_length(
    taper_type: str = "gaussian",
    w_start_um: float = platform.SM_WIDTH_UM,
    w_end_um: float = platform.MM_WIDTH_UM,
    height_um: float = platform.SI_THICKNESS_UM,
    wavelength_um: float = platform.WAVELENGTH_UM,
    threshold: float = 0.1,
    length_min_um: float = 5.0,
    length_max_um: float = 200.0,
    n_search: int = 40,
) -> float:
    """
    Binary-search for the shortest taper length that satisfies the
    adiabaticity condition max α(z) < threshold.

    Parameters
    ----------
    taper_type   : one of ``"linear"``, ``"parabolic"``, ``"gaussian"``,
                   ``"optimal"``, or ``"swg"``
    threshold    : adiabaticity threshold (default 0.1)
    length_min_um: lower bound for search (µm)
    length_max_um: upper bound for search (µm)
    n_search     : number of bisection steps

    Returns
    -------
    length_um : minimum taper length (µm) satisfying the criterion,
                or length_max_um if no solution was found.
    """
    _cls = {
        "linear": LinearTaper,
        "parabolic": ParabolicTaper,
        "gaussian": GaussianTaper,
        "optimal": OptimalAdiabaticTaper,
        "swg": SubwavelengthGratingTaper,
    }
    if taper_type not in _cls:
        raise ValueError(f"taper_type must be one of {list(_cls)}")
    cls = _cls[taper_type]

    kwargs = dict(w_start_um=w_start_um, w_end_um=w_end_um,
                  height_um=height_um, wavelength_um=wavelength_um)

    lo, hi = length_min_um, length_max_um
    for _ in range(n_search):
        mid = (lo + hi) / 2
        t = cls(length_um=mid, **kwargs)
        if t.is_adiabatic(threshold):
            hi = mid
        else:
            lo = mid
    return hi
