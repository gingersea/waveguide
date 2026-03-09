"""
Waveguide photonic design library for the ANT NanoSOI silicon photonics platform.

Modules
-------
platform  – Material constants and process parameters.
modes     – Slab-mode solver and Effective Index Method for strip waveguides.
taper     – SM-to-MM adiabatic taper designs (linear, parabolic, Gaussian,
             optimal equi-adiabatic, sub-wavelength grating).
mzi       – Low-loss asymmetric MZI with multimode long arms.
"""

from .platform import (
    WAVELENGTH_UM,
    N_SI,
    N_SIO2,
    SI_THICKNESS_UM,
    SM_WIDTH_UM,
    MM_WIDTH_UM,
    LOSS_SM_DB_PER_CM,
    LOSS_MM_DB_PER_CM,
)
from .modes import solve_slab_te, effective_indices, group_index
from .taper import (
    LinearTaper,
    ParabolicTaper,
    GaussianTaper,
    OptimalAdiabaticTaper,
    SubwavelengthGratingTaper,
    swg_effective_index,
    compare_tapers,
    minimum_adiabatic_length,
)
from .mzi import AsymmetricMZI

__all__ = [
    # platform constants
    "WAVELENGTH_UM",
    "N_SI",
    "N_SIO2",
    "SI_THICKNESS_UM",
    "SM_WIDTH_UM",
    "MM_WIDTH_UM",
    "LOSS_SM_DB_PER_CM",
    "LOSS_MM_DB_PER_CM",
    # mode solver
    "solve_slab_te",
    "effective_indices",
    "group_index",
    # tapers
    "LinearTaper",
    "ParabolicTaper",
    "GaussianTaper",
    "OptimalAdiabaticTaper",
    "SubwavelengthGratingTaper",
    "swg_effective_index",
    "compare_tapers",
    "minimum_adiabatic_length",
    # MZI
    "AsymmetricMZI",
]
