"""Thermophysical properties for the nucleation closure.

Two back-ends:

* CoolProp (Helmholtz-energy reference EOS) for n-pentane, ethanol, water,
  ammonia, nitrogen, methanol, n-heptane, n-hexane.
* A corresponding-states "lite" model for fluids CoolProp lacks.  Only
  iso-octane (2,2,4-trimethylpentane, the ECN Spray G fuel) is defined here:
  Lee-Kesler vapour pressure, Rackett liquid density, Peng-Robinson saturated
  vapour density, Watson latent heat, Guggenheim-Katayama surface tension and
  linear c_p / k / Andrade viscosity fits anchored on handbook values.

All liquid properties of the *metastable* (superheated) liquid at (T, p<p_sat)
are approximated by saturated-liquid values at T, which is standard because
the liquid is nearly incompressible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

R_UNIV = 8.314462618   # J/(mol K)
N_A = 6.02214076e23    # 1/mol
K_B = 1.380649e-23     # J/K

try:  # pragma: no cover - import guard
    import CoolProp.CoolProp as CP
    HAVE_COOLPROP = True
except ImportError:  # pragma: no cover
    CP = None
    HAVE_COOLPROP = False

COOLPROP_NAMES = {
    "n-pentane": "n-Pentane", "pentane": "n-Pentane",
    "ethanol": "Ethanol", "water": "Water", "ammonia": "Ammonia", "nh3": "Ammonia",
    "nitrogen": "Nitrogen", "ln2": "Nitrogen", "methanol": "Methanol",
    "n-heptane": "n-Heptane", "n-hexane": "n-Hexane",
}


@dataclass(frozen=True)
class FluidProps:
    """Properties of a superheated liquid at temperature T and local pressure p."""
    fluid: str
    T: float          # K, liquid temperature
    p: float          # Pa, local liquid pressure (ambient for a parcel, cell pressure in-nozzle)
    M: float          # kg/mol
    T_c: float        # K
    p_c: float        # Pa
    p_sat: float      # Pa, saturation pressure at T
    T_sat: float      # K, saturation temperature at p
    rho_l: float      # kg/m3, saturated liquid at T
    rho_v_T: float    # kg/m3, saturated vapour at T (for the Blander-Katz delta factor)
    rho_v: float      # kg/m3, saturated vapour at T_sat(p) (bubble interior during growth)
    sigma: float      # N/m at T
    h_fg: float       # J/kg at T_sat(p)
    cp_l: float       # J/(kg K) at T
    k_l: float        # W/(m K) at T
    mu_l: float       # Pa s at T

    # ---- derived quantities -------------------------------------------------
    @property
    def nu_l(self) -> float:
        return self.mu_l / self.rho_l

    @property
    def alpha_l(self) -> float:
        return self.k_l / (self.rho_l * self.cp_l)

    @property
    def dT(self) -> float:
        """Superheat T - T_sat(p) [K]."""
        return self.T - self.T_sat

    @property
    def dp(self) -> float:
        """Pressure undershoot p_sat(T) - p [Pa]."""
        return self.p_sat - self.p

    @property
    def delta(self) -> float:
        """Blander-Katz factor 1 - rho_v/rho_l (Poynting correction, first order)."""
        return 1.0 - self.rho_v_T / self.rho_l

    @property
    def eta(self) -> float:
        """Exact Poynting factor exp[(p - p_sat) v_l / (k_B T)]."""
        v_l = self.M / N_A / self.rho_l
        return math.exp((self.p - self.p_sat) * v_l / (K_B * self.T))

    @property
    def dp_eff(self) -> float:
        """Effective driving pressure p_v* - p = eta p_sat - p for the critical bubble."""
        return self.eta * self.p_sat - self.p

    @property
    def r_c(self) -> float:
        """Critical (Laplace) radius 2 sigma / (p_v* - p) [m]; inf if not superheated."""
        d = self.dp_eff
        return 2.0 * self.sigma / d if d > 0 else math.inf

    @property
    def Ja(self) -> float:
        """Jakob number rho_l c_p dT / (rho_v h_fg)."""
        return self.rho_l * self.cp_l * max(self.dT, 0.0) / (self.rho_v * self.h_fg)

    @property
    def Rp(self) -> float:
        """Pressure ratio p_sat(T)/p (>1 when flashing)."""
        return self.p_sat / self.p


# ---------------------------------------------------------------------------
# Lite corresponding-states fluid (used for iso-octane)
# ---------------------------------------------------------------------------
class LiteFluid:
    """Corresponding-states property model anchored on handbook data."""

    def __init__(self, name, M, T_c, p_c, omega, T_b, rho_ref, T_rho_ref,
                 sigma_ref, T_sigma_ref, h_fg_b, cp_ref, T_cp_ref, cp_slope,
                 k_ref, T_k_ref, k_slope, mu_ref, T_mu_ref, mu_B):
        self.name, self.M, self.T_c, self.p_c, self.omega, self.T_b = name, M, T_c, p_c, omega, T_b
        self.h_fg_b = h_fg_b
        self.sigma_ref, self.T_sigma_ref = sigma_ref, T_sigma_ref
        self.cp_ref, self.T_cp_ref, self.cp_slope = cp_ref, T_cp_ref, cp_slope
        self.k_ref, self.T_k_ref, self.k_slope = k_ref, T_k_ref, k_slope
        self.mu_B = mu_B
        self.mu_A = mu_ref / math.exp(mu_B / T_mu_ref)
        # Rackett Z_RA calibrated so that rho_l(T_rho_ref) = rho_ref
        Tr = T_rho_ref / T_c
        Vs = M / rho_ref
        self.Z_RA = (Vs * p_c / (R_UNIV * T_c)) ** (1.0 / (1.0 + (1.0 - Tr) ** (2.0 / 7.0)))

    # Lee-Kesler vapour pressure
    def p_sat(self, T):
        return self._p_sat(T)

    def _p_sat(self, T):
        Tr = T / self.T_c
        lnTr = math.log(Tr)
        f0 = 5.92714 - 6.09648 / Tr - 1.28862 * lnTr + 0.169347 * Tr ** 6
        f1 = 15.2518 - 15.6875 / Tr - 13.4721 * lnTr + 0.43577 * Tr ** 6
        return self.p_c * math.exp(f0 + self.omega * f1)

    def T_sat(self, p):
        if p >= self.p_c:
            raise ValueError("pressure above critical")
        return brentq(lambda T: self._p_sat(T) - p, 0.35 * self.T_c, self.T_c - 1e-6, xtol=1e-9)

    def rho_l(self, T):
        Tr = min(T / self.T_c, 0.999)
        Vs = (R_UNIV * self.T_c / self.p_c) * self.Z_RA ** (1.0 + (1.0 - Tr) ** (2.0 / 7.0))
        return self.M / Vs

    def rho_v(self, T):
        """Saturated vapour density from the Peng-Robinson vapour root."""
        p = self._p_sat(T)
        Tr = T / self.T_c
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        alpha = (1.0 + kappa * (1.0 - math.sqrt(Tr))) ** 2
        a = 0.45724 * (R_UNIV * self.T_c) ** 2 / self.p_c * alpha
        b = 0.07780 * R_UNIV * self.T_c / self.p_c
        A = a * p / (R_UNIV * T) ** 2
        B = b * p / (R_UNIV * T)
        roots = np.roots([1.0, -(1.0 - B), A - 3 * B ** 2 - 2 * B, -(A * B - B ** 2 - B ** 3)])
        Z = max(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > B)
        return p * self.M / (Z * R_UNIV * T)

    def sigma(self, T):
        Tr = T / self.T_c
        Tr_ref = self.T_sigma_ref / self.T_c
        return self.sigma_ref * (max(1.0 - Tr, 0.0) / (1.0 - Tr_ref)) ** 1.222

    def h_fg(self, T):
        Tr = T / self.T_c
        Trb = self.T_b / self.T_c
        return self.h_fg_b * (max(1.0 - Tr, 0.0) / (1.0 - Trb)) ** 0.38

    def cp_l(self, T):
        return self.cp_ref + self.cp_slope * (T - self.T_cp_ref)

    def k_l(self, T):
        return max(self.k_ref + self.k_slope * (T - self.T_k_ref), 0.02)

    def mu_l(self, T):
        return self.mu_A * math.exp(self.mu_B / T)


# iso-octane (2,2,4-trimethylpentane).  Anchors: T_c=543.9 K, p_c=2.57 MPa,
# omega=0.303, T_b=372.39 K, rho_l(20 C)=691.9 kg/m3, sigma(20 C)=18.77 mN/m,
# h_fg(T_b)=30.79 kJ/mol, c_p(25 C)=239 J/mol/K, k(25 C)=0.0954 W/m/K,
# mu(20 C)=0.503 mPa s (standard handbook / NIST WebBook values).
LITE_FLUIDS = {
    "iso-octane": LiteFluid(
        name="iso-octane", M=0.114229, T_c=543.9, p_c=2.57e6, omega=0.303, T_b=372.39,
        rho_ref=691.9, T_rho_ref=293.15, sigma_ref=18.77e-3, T_sigma_ref=293.15,
        h_fg_b=30.79e3 / 0.114229, cp_ref=239.0 / 0.114229, T_cp_ref=298.15, cp_slope=3.6,
        k_ref=0.0954, T_k_ref=298.15, k_slope=-2.5e-4, mu_ref=0.503e-3, T_mu_ref=293.15, mu_B=1130.0),
}
LITE_FLUIDS["isooctane"] = LITE_FLUIDS["iso-octane"]


def _canon(fluid: str) -> str:
    return fluid.strip().lower()


def fluid_info(fluid: str) -> dict:
    """Critical temperature/pressure and molar mass."""
    f = _canon(fluid)
    if f in LITE_FLUIDS:
        lf = LITE_FLUIDS[f]
        return {"T_c": lf.T_c, "p_c": lf.p_c, "M": lf.M, "backend": "lite"}
    if not HAVE_COOLPROP:
        raise RuntimeError("CoolProp not available and fluid not in lite set")
    name = COOLPROP_NAMES[f]
    return {"T_c": CP.PropsSI("Tcrit", name), "p_c": CP.PropsSI("pcrit", name),
            "M": CP.PropsSI("M", name), "backend": "coolprop"}


def saturation_T(fluid: str, p: float) -> float:
    f = _canon(fluid)
    if f in LITE_FLUIDS:
        return LITE_FLUIDS[f].T_sat(p)
    return CP.PropsSI("T", "P", p, "Q", 0, COOLPROP_NAMES[f])


def saturation_p(fluid: str, T: float) -> float:
    f = _canon(fluid)
    if f in LITE_FLUIDS:
        return LITE_FLUIDS[f]._p_sat(T)
    return CP.PropsSI("P", "T", T, "Q", 0, COOLPROP_NAMES[f])


@lru_cache(maxsize=200_000)
def get_props(fluid: str, T: float, p: float) -> FluidProps:
    """Properties of liquid `fluid` at temperature T [K] and local pressure p [Pa]."""
    f = _canon(fluid)
    T = float(T)
    p = float(p)
    if f in LITE_FLUIDS:
        lf = LITE_FLUIDS[f]
        if T >= lf.T_c:
            raise ValueError("T above critical temperature")
        T_s = lf.T_sat(p)
        return FluidProps(
            fluid=f, T=T, p=p, M=lf.M, T_c=lf.T_c, p_c=lf.p_c, p_sat=lf._p_sat(T), T_sat=T_s,
            rho_l=lf.rho_l(T), rho_v_T=lf.rho_v(T), rho_v=lf.rho_v(T_s), sigma=lf.sigma(T),
            h_fg=lf.h_fg(T_s), cp_l=lf.cp_l(T), k_l=lf.k_l(T), mu_l=lf.mu_l(T))
    if not HAVE_COOLPROP:
        raise RuntimeError("CoolProp not available")
    name = COOLPROP_NAMES[f]
    T_c = CP.PropsSI("Tcrit", name)
    if T >= T_c:
        raise ValueError("T above critical temperature")
    T_s = CP.PropsSI("T", "P", p, "Q", 0, name)
    h_fg = CP.PropsSI("H", "P", p, "Q", 1, name) - CP.PropsSI("H", "P", p, "Q", 0, name)
    try:
        sigma = CP.PropsSI("I", "T", T, "Q", 0, name)
    except ValueError:
        # some CoolProp sigma correlations fail within ~1 K of T_c: Guggenheim extrapolation
        T_ref = T_c - 2.0
        sigma = CP.PropsSI("I", "T", T_ref, "Q", 0, name) * ((T_c - T) / 2.0) ** 1.222
    return FluidProps(
        fluid=f, T=T, p=p, M=CP.PropsSI("M", name), T_c=T_c, p_c=CP.PropsSI("pcrit", name),
        p_sat=CP.PropsSI("P", "T", T, "Q", 0, name), T_sat=T_s,
        rho_l=CP.PropsSI("D", "T", T, "Q", 0, name), rho_v_T=CP.PropsSI("D", "T", T, "Q", 1, name),
        rho_v=CP.PropsSI("D", "P", p, "Q", 1, name), sigma=sigma,
        h_fg=h_fg, cp_l=CP.PropsSI("C", "T", T, "Q", 0, name), k_l=CP.PropsSI("L", "T", T, "Q", 0, name),
        mu_l=CP.PropsSI("V", "T", T, "Q", 0, name))
