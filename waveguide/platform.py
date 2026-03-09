"""
ANT NanoSOI platform constants and material properties.

Platform: Applied Nanotools NanoSOI
  - 220 nm silicon device layer
  - 2 µm buried oxide (BOX) layer
  - Oxide top cladding (SiO2)
  - Target wavelength: 1550 nm

References:
  https://www.appliednt.com/nanosoi/sys/resources/specs/
"""

# ── Wavelength ────────────────────────────────────────────────────────────────
WAVELENGTH_UM = 1.55  # µm (1550 nm)

# ── Material refractive indices at 1550 nm ────────────────────────────────────
N_SI = 3.476   # Silicon core
N_SIO2 = 1.444  # Silicon dioxide (BOX and top cladding)

# ── Layer geometry ────────────────────────────────────────────────────────────
SI_THICKNESS_UM = 0.220   # 220 nm silicon device layer
BOX_THICKNESS_UM = 2.0    # 2 µm buried oxide

# ── Waveguide width limits (single-mode vs. multimode) ───────────────────────
# Single-mode TE operation up to ~750 nm at 1550 nm for 220 nm SOI
SM_WIDTH_UM = 0.450    # 450 nm nominal single-mode routing width
MM_WIDTH_UM = 2.000    # 2 µm multimode routing width (low-loss)

# ── Measured propagation losses ───────────────────────────────────────────────
# Single-mode waveguide (450 nm wide): 1.5–2 dB/cm
LOSS_SM_DB_PER_CM_MIN = 1.5
LOSS_SM_DB_PER_CM_MAX = 2.0
LOSS_SM_DB_PER_CM = 1.75   # nominal (mid-range)

# Multimode waveguide (≥2 µm wide): 0.2–0.4 dB/cm
LOSS_MM_DB_PER_CM_MIN = 0.2
LOSS_MM_DB_PER_CM_MAX = 0.4
LOSS_MM_DB_PER_CM = 0.3   # nominal (mid-range)

# ── Component loss estimates ──────────────────────────────────────────────────
# Directional coupler (single-mode), per coupler: ~0.1 dB excess IL
COUPLER_IL_DB = 0.1

# Well-designed adiabatic SM-to-MM taper per taper: ~0.05–0.1 dB IL
TAPER_IL_DB = 0.05
