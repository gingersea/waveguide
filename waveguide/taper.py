"""
Single-mode to multimode (SM-to-MM) adiabatic taper designs.

Three taper profile geometries are provided:

  LinearTaper     – width increases linearly with propagation distance z.
  ParabolicTaper  – width follows a quadratic (parabolic) profile.
  GaussianTaper   – width follows a Gaussian-sigmoid profile; the width
                    change is concentrated at both ends to maximise
                    adiabaticity in the narrow coupling region where the
                    inter-modal beat length is shortest.

Each taper class exposes:
  width_profile(z)          – width (µm) as a function of position z (µm).
  adiabaticity(z_points)    – local adiabaticity parameter α(z); the taper
                              is fully adiabatic when α(z) << 1 everywhere.
  coupling_efficiency()     – fraction of input power remaining in the
                              fundamental mode at the output, estimated
                              from the coupled-mode overlap integral.
  insertion_loss_db()       – taper insertion loss (dB).
  summary()                 – human-readable parameter table.

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

        dz = self.length_um / n_points
        alpha = np.empty_like(z_points)
        for i, z in enumerate(z_points):
            w = float(self.width_profile(z))
            dw_dz = float(
                (self.width_profile(min(z + dz / 2, self.length_um))
                 - self.width_profile(max(z - dz / 2, 0.0))) / dz
            )
            db = _delta_beta(w, self.height_um, self.n_core, self.n_clad,
                              self.wavelength_um)
            # Guard against division by zero
            denominator = db * w if db * w > 1e-12 else 1e-12
            alpha[i] = abs(dw_dz) / denominator
        return z_points, alpha

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
    increasing σ towards the linear limit.

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


# ── Taper comparison utility ──────────────────────────────────────────────────

def compare_tapers(
    length_um: float = 50.0,
    w_start_um: float = platform.SM_WIDTH_UM,
    w_end_um: float = platform.MM_WIDTH_UM,
    height_um: float = platform.SI_THICKNESS_UM,
    wavelength_um: float = platform.WAVELENGTH_UM,
) -> dict[str, _BaseTaper]:
    """
    Instantiate one taper of each geometry and return a dict keyed by name.

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
    return {
        "linear": LinearTaper(**shared),
        "parabolic": ParabolicTaper(**shared),
        "gaussian": GaussianTaper(**shared),
    }


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
    taper_type   : "linear", "parabolic", or "gaussian"
    threshold    : adiabaticity threshold (default 0.1)
    length_min_um: lower bound for search (µm)
    length_max_um: upper bound for search (µm)
    n_search     : number of bisection steps

    Returns
    -------
    length_um : minimum taper length (µm) satisfying the criterion,
                or length_max_um if no solution was found.
    """
    _cls = {"linear": LinearTaper, "parabolic": ParabolicTaper,
            "gaussian": GaussianTaper}
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
