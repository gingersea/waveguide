# waveguide

Silicon photonics design library for the **Applied Nanotools (ANT) NanoSOI** platform:
220 nm Si device layer / 2 µm SiO₂ BOX, targeting 1550 nm.

The library realises a **low-loss asymmetric Mach-Zehnder Interferometer (MZI)**
by replacing long single-mode routing sections with multimode waveguides
connected through compact, adiabatic SM-to-MM mode converters (tapers).

---

## Motivation

| Waveguide type | Width | Propagation loss |
|---|---|---|
| Single-mode (SM) | 450 nm | 1.5–2 dB/cm |
| Multimode (MM) | ≥ 2 µm | 0.2–0.4 dB/cm |

The directional coupler regions are kept single-mode; only the long routing
arms switch to multimode via adiabatic tapers.

---

## Package structure

```
waveguide/
├── platform.py   – ANT NanoSOI material constants and loss figures
├── modes.py      – Slab-mode solver + Effective Index Method (EIM)
├── taper.py      – SM-to-MM adiabatic tapers:
│                   linear, parabolic, Gaussian,
│                   OptimalAdiabaticTaper (equi-adiabatic, most compact),
│                   SubwavelengthGratingTaper (SWG / digital metamaterial)
└── mzi.py        – Asymmetric MZI: loss budget, FSR, transmission spectrum

tests/
├── test_modes.py
├── test_taper.py
└── test_mzi.py
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick start

### Taper comparison

```python
from waveguide import compare_tapers, minimum_adiabatic_length

tapers = compare_tapers(length_um=50)
for name, t in tapers.items():
    print(t.summary())
```

```
Taper type  : LinearTaper
Width range : 450 nm → 2000 nm
Length      : 50.0 µm
Max α(z)    : 0.0574 (adiabatic)
Insert. loss: 0.022 dB

Taper type  : ParabolicTaper
...
```

Find the minimum adiabatic taper length for each profile:

```python
for tt in ["linear", "parabolic", "gaussian"]:
    L = minimum_adiabatic_length(tt, length_max_um=300)
    print(f"{tt:12s}: {L:.1f} µm")
```

### Asymmetric MZI design

```python
from waveguide import AsymmetricMZI

mzi = AsymmetricMZI(delta_l_um=2000.0)   # 2 mm extra MM path
print(mzi.summary())
```

```
Asymmetric MZI – Low-Loss Design Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Platform     : ANT NanoSOI 220 nm Si / SiO₂
  Wavelength   : 1550 nm
  ...
  Loss saving  : 0.238 dB vs. all-SM routing
  FSR          : 0.33 nm  (0.042 THz)
```

Compute the spectral transmission:

```python
import numpy as np
wl, T_thru, T_cross = mzi.transmission(n_points=500)
```

---

## Design overview

### SM-to-MM adiabatic taper

Five taper profiles are implemented.  All transition from 450 nm (single-mode)
to 2 µm (multimode) on the 220 nm Si / SiO₂ platform.

| Profile | Key property | Min adiabatic length |
|---|---|---|
| `LinearTaper` | constant dw/dz | ~29 µm |
| `ParabolicTaper` | dw/dz = 0 at narrow end | ~57 µm |
| `GaussianTaper` (σ=2) | smooth S-shape | ~47 µm |
| **`OptimalAdiabaticTaper`** | **constant α(z); most compact** | **~19 µm** |
| `SubwavelengthGratingTaper` | discrete pixels; ultra-compact footprint | — |

Adiabaticity criterion: `α(z) = |dw/dz| / (Δβ(w) × w) < 0.1`.

#### OptimalAdiabaticTaper (equi-adiabatic)

*From: "Efficient adiabatic silicon-on-insulator waveguide taper"*

Width profile obtained by numerically integrating the Snyder–Love criterion
with constant α:

```
dz/dw = 1 / (α_target × Δβ(w) × w)
```

This is the most compact profile for a given loss budget — it makes full use
of the permitted local coupling everywhere, achieving adiabaticity in ~19 µm
vs. ~29 µm for a linear taper.

```python
from waveguide import OptimalAdiabaticTaper

t = OptimalAdiabaticTaper(alpha_target=0.05)
print(f"Natural length: {t.natural_length_um:.1f} µm")
print(t.summary())
```

#### SubwavelengthGratingTaper (digital metamaterial)

*From: "Adiabatic and Ultracompact Waveguide Tapers Based on Digital
Metamaterials"*

The taper consists of `n_segments` discrete silicon cells (pitch `period_um`).
Each cell's fill factor `ff(z)` varies from `ff_start = w_start/w_end` to 1.0,
giving an effective silicon width `w_eff = ff × w_end`.  The TE effective
medium index at each cell is:

```
n_eff(ff) = sqrt(ff × n_Si² + (1-ff) × n_SiO₂²)
```

Because `period_um < λ/(2 n_Si) ≈ 0.22 µm`, no diffraction occurs and the
structure acts as a graded-index medium.

```python
from waveguide import SubwavelengthGratingTaper, swg_effective_index

t = SubwavelengthGratingTaper(length_um=10.0, period_um=0.200)
print(f"N segments: {t.n_segments}, sub-wavelength: {t.is_subwavelength()}")
print(f"EMT index at ff=0.5 (TE): {swg_effective_index(0.5, polarization='TE'):.3f}")
print(t.summary())
```

### Asymmetric MZI layout

```
Input ──[DC₁]──┬── SM ──[taper]── MM (ΔL) ──[taper]── SM ──┬──[DC₂]── Through
               │                                             │
               └──────────────── SM ─────────────────────── ┘── Cross
```

- **DC** = 50:50 directional coupler (single-mode)
- **SM** = 450 nm single-mode waveguide, loss 1.75 dB/cm
- **MM** = 2 µm multimode waveguide, loss 0.30 dB/cm
- **ΔL** = extra length in the long arm (multimode section)

The long arm saves `(1.75 − 0.30) dB/cm × ΔL` compared to all-SM routing.

---

## Running tests

```bash
python -m pytest tests/ -v
```
