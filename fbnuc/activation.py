"""Activation of pre-existing nuclei (Harvey / Hsu criterion).

A pre-existing gas or vapour nucleus of radius r (a free micro-bubble in the
liquid, or the gas pocket trapped in a wall cavity) becomes mechanically
unstable and grows when the critical radius falls below it:

    r > r_c(Delta p) = 2 sigma / (p_v* - p)      <=>      Lambda := r_ref / r_c > r_ref / r.

With a size distribution of nuclei f(r / r_ref), the activated fraction is the
survival function of r/r_ref evaluated at 1/Lambda:

    P_act(Lambda) = Pr(r/r_ref > 1/Lambda).

Three families are provided.  All are monotone in Lambda, zero at Lambda=0 and
saturate at 1, so positivity, monotonicity in superheat and the saturation
seen in the legacy correlations are built in.

* exponential:  P = exp(-1/Lambda).  With the Clausius-Clapeyron relation
  Delta p ~ rho_v h_fg dT / T_sat this is exactly Senda's form
  N exp(-a / dT) with a = 2 sigma T_sat / (delta rho_v h_fg r_ref).
* lognormal:    P = Phi(ln(Lambda) / s)   (ln(r/r_ref) ~ N(0, s^2)).
* pareto:       P = min(1, Lambda^m)       (f ~ r^-(m+1) above r_min = r_ref),
  which gives the classical power laws N_a ~ dT^m of boiling-site-density
  correlations before saturation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


def P_act(Lambda, dist: str = "lognormal", s=0.7, m=3.0):
    """Activated fraction for activation number Lambda = r_ref / r_c (array-safe)."""
    L = np.asarray(Lambda, dtype=float)
    out = np.zeros(np.broadcast(L, np.asarray(s, dtype=float)).shape)
    pos = L > 0
    if dist == "exponential":
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            val = np.exp(-1.0 / np.where(pos, L, 1.0))
    elif dist == "lognormal":
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log(np.where(pos, L, 1.0)) / (np.asarray(s, dtype=float) * np.sqrt(2.0))
            val = 0.5 * erfc(-z)
    elif dist == "pareto":
        val = np.minimum(1.0, np.where(pos, L, 0.0) ** m)
    else:
        raise ValueError(f"unknown distribution {dist}")
    out = np.where(pos, val, 0.0)
    return out
