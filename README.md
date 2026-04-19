# Cascade Heat Pump Model for Industrial Drying

This repository contains the simulation code used in a research paper on
cascade heat pump systems integrated with industrial drying processes.  
The code is made publicly available to support reproducibility of the results.

> **Paper**: _currently under review — citation will be added upon publication._

---

## Overview

Three thermodynamic models are provided:

| Model | Entry point | Description |
|---|---|---|
| **Single-stage heat pump** | Single stage heat pump/`Optimiser.py` | Single-stage vapour-compression heat pump coupled to a drying loop. Optimises refrigerant cycle state for maximum COP. |
| **Cascade heat pump** | Cascade heat pump/`Optimiser.py` | Two-stage vapour-compression cascade heat pump coupled to a drying loop. Optimises refrigerant cycle states for maximum COP. |
| **Centrifugal compressor** | Compressor simulation`CompressorSolver.py` | 1-D loss-model sizing tool for multi-stage centrifugal compressors. Sweeps specific speed to find the optimal design point. |

---

## Repository layout

```
.
├── README.md
│
├── Cascade heat pump/
|   ├── Optimiser.py                  # Entry point – cascade heat pump simulation
|   ├── DryingIntegrated.py           # Psychrometric dryer model + Mollier diagram|
|   ├── SimpleCascade.py              # Top-level cascade heat pump class
|   └── simple_cascade_model/         # Package: cycle solver internals
│       ├── __init__.py
│       ├── constants.py              # Physical constants and solver settings
│       ├── core.py                   # UpperCycle and LowerCycle solvers
│       ├── cyclestate.py             # CycleState data container
│       ├── constraints.py            # Pinch-point constraint functions
│       ├── utils.py                  # Air temperature profiles, entropy production
│       └── plots.py                  # T-s and T-Q diagram utilities
│
├── Compressor simulation/
|   ├── CompressorSolver.py           # Entry point – compressor sizing sweep
|   └── Compressor.py                 # Single-stage centrifugal compressor model
|
└── Single stage heat pump/
    ├── Optimiser.py                  # Entry point – Single-stage heat pump simulation
    ├── DryingIntegrated.py           # Psychrometric dryer model + Mollier diagram
    ├── SimpleSingleStage.py          # Top-level single-stage heat pump class
    └── single_stage_model/           # Package: cycle solver internals
        ├── __init__.py
        ├── constants.py              # Physical constants and solver settings
        ├── core.py                   # UpperCycle and LowerCycle solvers
        ├── cyclestate.py             # CycleState data container
        ├── constraints.py            # Pinch-point constraint functions
        ├── utils.py                  # Air temperature profiles, entropy production
        └── plots.py                  # T-s and T-Q diagram utilities
```

---

## Dependencies

| Package | Purpose |
|---|---|
| [CoolProp](http://www.coolprop.org/) + REFPROP backend | Thermophysical properties of refrigerant mixtures and humid air |
| NumPy | Numerical arrays and calculations |
| SciPy | Scalar minimisation (`scipy.optimize.minimize_scalar`) |
| Matplotlib | Plotting (Mollier, T-s, T-Q diagrams) |

Install the Python dependencies with:

```bash
pip install numpy scipy matplotlib CoolProp
```

> **Note:** REFPROP requires a separate licence from NIST and must be installed
> independently. CoolProp uses REFPROP as a backend for mixture property
> calculations. See the [CoolProp documentation](http://www.coolprop.org/coolprop/REFPROP.html)
> for setup instructions.

---

## Running the models

### Cascade heat pump

```bash
python Optimiser.py
```

All user-adjustable inputs (fluid mixture, boundary conditions, pinch
temperatures, IHX flags) are defined at the top of `Optimiser.py`.

### Centrifugal compressor

```bash
python CompressorSolver.py
```

All inputs (fluid mixture, operating conditions, geometry ratios, specific
speed sweep range) are defined in the configuration section of
`CompressorSolver.py`.

---

## Model descriptions

### Cascade heat pump (`simple_cascade_model/`)

The model solves a two-stage vapour-compression cascade in which:

- the **upper cycle** rejects heat to the dryer air stream (via a condenser
  or optional heat transfer fluid loop);
- the **lower cycle** absorbs heat from the exhaust dryer air stream;
- the two cycles are thermally coupled through a shared heat exchanger.

Each cycle is solved iteratively to satisfy all pinch-point constraints.
Optional internal heat exchangers (IHX) can be enabled for both stages.
Performance is evaluated using first- and second-law metrics (COP, Lorenz COP,
entropy production per component).

### Centrifugal compressor (`Compressor.py`)

A one-dimensional stage model based on seven loss correlations (disk friction,
tip clearance, skin friction, blade loading, recirculation, jet-wake mixing,
and vaneless diffuser losses). The model iterates on the total-to-total
efficiency until convergence, then reports the stage isentropic efficiency,
geometry, and velocity triangles.

---

## Outputs

The cascade model prints:
- COP and second-law efficiency
- Per-component entropy production
- State points (T, p) for each cycle
- Volumetric flow rates at compressor inlets

Optionally, the following plots are generated:
- Mollier diagram (h–ω) of the drying process
- T-s diagrams for upper and lower refrigerant cycles
- T-Q diagrams for all heat exchangers

---

## Author

J.H. Lam  
_Affiliation and contact details to be added upon publication._

---

## Licence

This code is shared for academic reproducibility purposes.  
Please cite the associated paper (reference to follow) if you use this code
in your own work.
