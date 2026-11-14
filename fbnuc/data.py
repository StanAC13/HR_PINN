"""Calibration data sets.

There are, to the author's knowledge, no published in-nozzle bubble number
density measurements for flashing fuel injectors (that is the gap).  Two
clearly labelled data sources are therefore used:

* Case A - synthetic twin.  A hidden "true" injector is simulated with a
  *different* model structure (exponential free-nuclei population and Pareto
  wall cavities instead of the log-normal families used for inference) plus
  multiplicative measurement noise.  This tests the method, including its
  behaviour under model-form error, with a known answer.

* Case B - legacy-correlation anchor.  The Senda et al. (1994) correlation is
  treated as the only available summary of real data, evaluated at a few
  superheats with a half-decade uncertainty, for the fluid it is usually
  assumed to describe (n-pentane at 1 bar).  The product is a fluid-consistent
  replacement for that correlation with uncertainty.
"""
from __future__ import annotations

import numpy as np

from .closure import Geometry, make_batch, evaluate, senda

SPRAY_G = Geometry(D=165e-6, L_over_D=1.4, U=150.0)

# hidden truth for the synthetic twin (injector properties, shared across fluids)
THETA_TRUE = np.array([np.log(2.0e13),  # N_v  [1/m3]
                       np.log(0.8e-6),  # r_v  [m]
                       0.7,             # s_v  (unused by the exponential truth family)
                       np.log(5.0e10),  # N_s  [1/m2]
                       np.log(0.25e-6), # r_w  [m]
                       0.5,             # s_w  (unused by the Pareto truth family)
                       np.log(1.5)])    # We_d
TRUTH_FAMILIES = {"dist_bulk": "exponential", "dist_wall": "pareto", "m_wall": 2.5}


def ln_truth(sb):
    return np.log(evaluate(sb, THETA_TRUE, **TRUTH_FAMILIES))


def case_a_design(n_points: int = 6, p: float = 0.5e5, fluid: str = "iso-octane", geom=SPRAY_G):
    """Fuel-temperature sweep at fixed chamber pressure (the usual experimental protocol)."""
    designs = {
        3: [363.0, 383.0, 418.0],
        6: [357.0, 363.0, 373.0, 383.0, 398.0, 418.0],
        12: list(np.linspace(355.0, 425.0, 12)),
    }
    T = np.array(designs[n_points])
    return make_batch(fluid, T, np.full(len(T), p), geom)


def observe(ln_true, sigma_ln: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    return ln_true + sigma_ln * rng.standard_normal(len(ln_true))


def case_b_pseudo_data(dT_list=(5.0, 10.0, 20.0, 40.0, 60.0), fluid: str = "n-pentane", p: float = 1.0e5,
                       geom=SPRAY_G):
    """Senda correlation sampled as pseudo-observations (half-decade sd)."""
    from .properties import saturation_T
    T_s = saturation_T(fluid, p)
    dT = np.asarray(dT_list, dtype=float)
    sb = make_batch(fluid, T_s + dT, np.full(len(dT), p), geom)
    y = np.log(senda(dT))
    sigma = np.full(len(dT), np.log(10.0) * 0.5)
    return sb, y, sigma
