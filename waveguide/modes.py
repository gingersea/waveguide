"""
Waveguide mode solver using the Effective Index Method (EIM).

The EIM decomposes a 2-D strip waveguide into two 1-D slab problems:
  Step 1 – vertical slab (height H, indices n_core / n_clad):
            solve for the effective index n_eff_y of each guided slab mode.
  Step 2 – horizontal slab (width W, indices n_eff_y / n_clad):
            solve for the strip-waveguide effective indices and propagation
            constants of the TE-like modes.

Functions
---------
solve_slab_te(n_core, n_clad, dimension_um, wavelength_um, max_modes)
    Exact-analytical solution for TE modes of a *symmetric* slab waveguide.

effective_indices(width_um, height_um, n_core, n_clad, wavelength_um)
    Return the list of (n_eff, beta) for TE-like modes of a strip waveguide,
    using the two-step EIM.

group_index(width_um, height_um, n_core, n_clad, wavelength_um, dw_um)
    Numerical group index d(n_eff * k0) / dk0 via finite differences in
    wavelength.  Used to compute MZI free spectral range.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
from scipy.optimize import brentq


# ── Symmetric slab solver ────────────────────────────────────────────────────

def _te_characteristic(kappa, n_core, n_clad, h_um, wavelength_um, even: bool):
    """
    Characteristic equation residual for TE modes of a symmetric slab.

    Parameters
    ----------
    kappa        : transverse wave-vector inside the core (µm⁻¹)
    n_core       : refractive index of the core
    n_clad       : refractive index of the cladding
    h_um         : slab thickness (µm)
    wavelength_um: free-space wavelength (µm)
    even         : True for even modes (TE₀, TE₂, …), False for odd modes
    """
    k0 = 2 * np.pi / wavelength_um
    gamma = np.sqrt(max(k0**2 * (n_core**2 - n_clad**2) - kappa**2, 0.0))
    if even:
        return np.tan(kappa * h_um / 2) - gamma / kappa
    else:
        return -1.0 / np.tan(kappa * h_um / 2) - gamma / kappa


def solve_slab_te(
    n_core: float,
    n_clad: float,
    dimension_um: float,
    wavelength_um: float,
    max_modes: int = 4,
) -> List[float]:
    """
    Solve TE modes of a symmetric slab waveguide.

    Returns
    -------
    n_effs : list of effective indices, sorted descending (fundamental first).
             Only guided modes (n_eff > n_clad) are returned.
    """
    k0 = 2 * np.pi / wavelength_um
    kappa_max = k0 * np.sqrt(n_core**2 - n_clad**2)
    if kappa_max <= 0:
        return []

    n_effs: List[float] = []
    mode_idx = 0  # counts total modes found (0 = TE₀, 1 = TE₁, …)

    while len(n_effs) < max_modes:
        even = (mode_idx % 2 == 0)
        # The m-th mode (0-indexed) sits in the kappa interval
        # [m*π/h, (m+1)*π/h] ∩ [0, kappa_max]
        m = mode_idx
        kappa_lo = m * np.pi / dimension_um + 1e-9
        kappa_hi = min((m + 1) * np.pi / dimension_um - 1e-9, kappa_max - 1e-9)
        if kappa_lo >= kappa_hi:
            break

        try:
            f_lo = _te_characteristic(kappa_lo, n_core, n_clad, dimension_um,
                                      wavelength_um, even)
            f_hi = _te_characteristic(kappa_hi, n_core, n_clad, dimension_um,
                                      wavelength_um, even)
            if f_lo * f_hi > 0:
                mode_idx += 1
                continue
            kappa_sol = brentq(_te_characteristic, kappa_lo, kappa_hi,
                               args=(n_core, n_clad, dimension_um,
                                     wavelength_um, even),
                               xtol=1e-10, rtol=1e-10)
            n_eff = np.sqrt(n_core**2 - (kappa_sol / k0)**2)
            if n_eff > n_clad:
                n_effs.append(float(n_eff))
        except ValueError:
            pass

        mode_idx += 1

    return sorted(n_effs, reverse=True)


# ── 2-D strip waveguide via EIM ───────────────────────────────────────────────

def effective_indices(
    width_um: float,
    height_um: float = 0.220,
    n_core: float = 3.476,
    n_clad: float = 1.444,
    wavelength_um: float = 1.55,
    max_modes: int = 4,
) -> List[Tuple[float, float]]:
    """
    Return (n_eff, beta) pairs for the TE-like modes of a strip waveguide.

    Uses the two-step Effective Index Method (EIM):
      1. Solve the vertical slab (height × n_core / n_clad) → n_eff_y
      2. Solve the horizontal slab (width × n_eff_y / n_clad) → strip modes

    Parameters
    ----------
    width_um      : waveguide width (µm)
    height_um     : silicon layer height (µm), default 0.220 µm
    n_core        : core refractive index (default Si at 1550 nm)
    n_clad        : cladding refractive index (default SiO2 at 1550 nm)
    wavelength_um : free-space wavelength (µm)
    max_modes     : maximum number of modes to return

    Returns
    -------
    list of (n_eff, beta) tuples sorted by n_eff descending.
    beta is in µm⁻¹.
    """
    # Step 1 – vertical slab
    n_eff_y_list = solve_slab_te(n_core, n_clad, height_um, wavelength_um,
                                  max_modes=1)
    if not n_eff_y_list:
        warnings.warn("No guided mode found in vertical slab.")
        return []
    n_eff_y = n_eff_y_list[0]  # fundamental vertical mode

    # Step 2 – horizontal slab (using n_eff_y as effective core index)
    n_eff_strip_list = solve_slab_te(n_eff_y, n_clad, width_um, wavelength_um,
                                      max_modes=max_modes)

    k0 = 2 * np.pi / wavelength_um
    return [(n, n * k0) for n in n_eff_strip_list]


def group_index(
    width_um: float,
    height_um: float = 0.220,
    n_core: float = 3.476,
    n_clad: float = 1.444,
    wavelength_um: float = 1.55,
    delta_wl_um: float = 0.001,
) -> float:
    """
    Numerical group index n_g = n_eff - λ × dn_eff/dλ for the fundamental TE
    mode, computed via finite differences in wavelength.

    Parameters
    ----------
    delta_wl_um : finite-difference step (µm), default 1 nm

    Returns
    -------
    n_g : group index (dimensionless)
    """
    modes_plus = effective_indices(width_um, height_um, n_core, n_clad,
                                    wavelength_um + delta_wl_um)
    modes_minus = effective_indices(width_um, height_um, n_core, n_clad,
                                     wavelength_um - delta_wl_um)

    if not modes_plus or not modes_minus:
        raise ValueError("Could not find fundamental mode for group-index calculation.")

    n_eff_plus = modes_plus[0][0]
    n_eff_minus = modes_minus[0][0]
    dn_eff_dl = (n_eff_plus - n_eff_minus) / (2 * delta_wl_um)

    n_eff_mid = effective_indices(width_um, height_um, n_core, n_clad,
                                   wavelength_um)[0][0]
    return float(n_eff_mid - wavelength_um * dn_eff_dl)
