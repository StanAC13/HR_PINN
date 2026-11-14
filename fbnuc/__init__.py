"""fbnuc -- physics-prior Bayesian nucleation closure for flash-boiling sprays.

Addresses the data-scarcity gap for in-nozzle nucleation: the bubble number
density n_b supplied to flash-boiling spray / HRM / Sigma-Y solvers is built
from three physical mechanisms with bounded, interpretable parameters,

    n_b = N_v * P_act(Lambda_v)                     (pre-existing bulk nuclei)
        + 4 (L/D) N_s * P_act(Lambda_w) * f_dep / U (wall-cavity nucleation)
        + J_hom(T, p) * tau_exp                     (homogeneous CNT limit)

and the parameters are inferred by Bayesian calibration (with a GP model
discrepancy term) from whatever sparse data exist, reverting to the physics
prior where there are none.  See REPORT.md for the derivation.
"""

from .properties import get_props, fluid_info, FluidProps
from .closure import (Geometry, StateBatch, make_batch, evaluate, senda,
                      PARAM_NAMES, PRIOR_LO, PRIOR_HI, prior_center,
                      senda_equivalent_radius, activation_constant)
from .cnt import ln_J_hom, gibbs_number, superheat_limit
from .activation import P_act
from .growth import mrg_radius, mrg_time
from .spray_link import flash_breakup

__all__ = [
    "get_props", "fluid_info", "FluidProps", "Geometry", "StateBatch",
    "make_batch", "evaluate", "senda", "PARAM_NAMES", "PRIOR_LO", "PRIOR_HI",
    "prior_center", "senda_equivalent_radius", "activation_constant",
    "ln_J_hom", "gibbs_number", "superheat_limit", "P_act", "mrg_radius",
    "mrg_time", "flash_breakup",
]
