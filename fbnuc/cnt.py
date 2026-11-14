"""Classical nucleation theory (CNT) for the homogeneous limit.

Rate of formation of critical vapour embryos per unit liquid volume
(Blander & Katz 1975; Carey 2008, eq. form with the Poynting-corrected
vapour pressure p_v* = eta p_sat):

    J_hom = N_l sqrt(3 sigma / (pi m)) exp(-Delta G* / (k_B T)),
    Delta G* = 16 pi sigma^3 / (3 (p_v* - p)^2),   N_l = rho_l N_A / M,  m = M / N_A.

The "Gibbs number" Gb = Delta G*/(k_B T) is the natural dimensionless
superheat for this branch.  Because sigma^3 sits in the exponent, J_hom rises
by many decades per kelvin close to the kinetic superheat limit, which is the
temperature where J_hom reaches a conventional threshold (1e12 m^-3 s^-1,
i.e. 1e6 cm^-3 s^-1 in the droplet-column experiments).  CNT reproduces the
measured limits of most organic liquids to within about 1-2 K, so this branch
carries no free parameter.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .properties import FluidProps, get_props, saturation_T, fluid_info, K_B, N_A


def gibbs_number(pr: FluidProps) -> float:
    """Delta G* / (k_B T); inf when the liquid is not superheated."""
    d = pr.dp_eff
    if d <= 0:
        return math.inf
    return 16.0 * math.pi * pr.sigma ** 3 / (3.0 * K_B * pr.T * d * d)


def ln_J_hom(pr: FluidProps) -> float:
    """Natural log of the homogeneous nucleation rate [m^-3 s^-1] (-inf if none)."""
    Gb = gibbs_number(pr)
    if not math.isfinite(Gb):
        return -math.inf
    m = pr.M / N_A
    N_l = pr.rho_l * N_A / pr.M
    return math.log(N_l) + 0.5 * math.log(3.0 * pr.sigma / (math.pi * m)) - Gb


def heterogeneous_factor(theta_deg: float) -> float:
    """Geometric barrier reduction F(theta) = (2 + 3 cos theta - cos^3 theta)/4 for a
    spherical-cap embryo on a flat substrate with contact angle theta (through the
    liquid).  F=1 for theta=0 (fully wetting: no help), F->0 as theta->180 deg."""
    c = math.cos(math.radians(theta_deg))
    return (2.0 + 3.0 * c - c ** 3) / 4.0


def superheat_limit(fluid: str, p: float, J_threshold: float = 1e12) -> float:
    """Kinetic superheat limit T_sl(p) [K]: temperature where J_hom = J_threshold."""
    info = fluid_info(fluid)
    T_c = info["T_c"]
    T_s = saturation_T(fluid, p)
    lnJt = math.log(J_threshold)

    def g(T):
        return ln_J_hom(get_props(fluid, T, p)) - lnJt

    lo = T_s + 1.0
    hi = T_c - 1.5
    # J is monotone increasing in T at fixed p; bracket check
    if g(lo) > 0:
        return lo
    return brentq(g, lo, hi, xtol=1e-4)
