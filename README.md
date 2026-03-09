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
├── taper.py      – SM-to-MM adiabatic tapers: linear, parabolic, Gaussian
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

Three taper profiles are implemented.  All start at 450 nm (single-mode) and
end at 2 µm (multimode) with 220 nm Si height.

| Profile | dw/dz at narrow end | Peak α (50 µm) | IL (50 µm) |
|---|---|---|---|
| Linear | constant | 0.057 | 0.022 dB |
| Parabolic | 0 | 0.114 | 0.034 dB |
| Gaussian (σ=2) | ≈ 0 | 0.094 | 0.035 dB |

Adiabaticity criterion: `α(z) = |dw/dz| / (Δβ(w) × w) < 0.1`.

The **Gaussian** taper is recommended: it has near-zero slope at the narrow
end (where Δβ is smallest) and approaches the single-mode cutoff gently,
limiting inter-modal coupling in the critical region.

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
