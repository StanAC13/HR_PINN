"""Composite physics-prior nucleation closure.

Bubble number density per unit liquid volume at the nozzle exit (or in a
Lagrangian parcel / Eulerian cell):

    n_b(state; xi) = n_bulk + n_wall + n_hom

    n_bulk = N_v * P_act(r_v / r_c)                         pre-existing free nuclei
    n_wall = 4 (L/D) * N_s * P_act(r_w / r_c) * f_dep / U   wall-cavity sites convected out
    n_hom  = J_hom(T, p) * tau_exp                          homogeneous CNT

Wall term: each active cavity emits bubbles at frequency f_dep = 1 / t_g, where
t_g is the MRG growth time to the departure radius
r_dep = max(r_shear, kappa r_c), with the shear-stripping radius from a
near-wall Weber criterion in the viscous sublayer u(y) = u_tau^2 y / nu:

    rho_l u(r)^2 (2 r) / sigma = We_d   =>   r_shear = ( We_d sigma nu^2 / (2 rho_l u_tau^4) )^{1/3}.

Bubbles produced on the wetted area pi D L per second, divided by the liquid
volume flow U pi D^2/4, give the factor 4 (L/D)/U.  Geometry (L/D, D, U) thus
enters the closure explicitly, which legacy single-fluid correlations lack.

Parameter vector xi (7 entries, uniform priors in the transformed coordinates
below, bounds set from cavitation-nuclei and boiling-site literature):

    lnN_v  ln(free nuclei per m^3)           [1e10, 1e14]
    ln_rv  ln(size scale of free nuclei, m)  [50 nm, 5 um]
    s_v    log-spread of free nuclei sizes   [0.2, 1.5]
    lnN_s  ln(gas-trapping wall sites per m^2) [1e7, 1e12]
    ln_rw  ln(size scale of wall cavities, m)  [20 nm, 2 um]
    s_w    log-spread of wall cavity sizes     [0.2, 1.5]
    lnWe_d ln(departure Weber number)          [0.3, 10]

These are properties of the injector / fuel system (cleanliness, surface finish,
dissolved gas), not of the fluid.  The fluid enters only through r_c, J_hom and
the growth constants, which is the hypothesis that enables cross-fluid transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .properties import get_props, FluidProps
from .cnt import ln_J_hom
from .activation import P_act
from .growth import mrg_time

PARAM_NAMES = ("lnN_v", "ln_rv", "s_v", "lnN_s", "ln_rw", "s_w", "lnWe_d")
PARAM_LABELS = (r"$\ln N_v$", r"$\ln r_v$", r"$s_v$", r"$\ln N_s$", r"$\ln r_w$", r"$s_w$", r"$\ln We_d$")
PRIOR_LO = np.array([np.log(1e10), np.log(5e-8), 0.2, np.log(1e7), np.log(2e-8), 0.2, np.log(0.3)])
PRIOR_HI = np.array([np.log(1e14), np.log(5e-6), 1.5, np.log(1e12), np.log(2e-6), 1.5, np.log(10.0)])
KAPPA_DEP = 2.0          # departure radius must exceed kappa * r_c (supercritical)
R_FLOOR = 1e-9           # [m] smallest meaningful critical radius for the packing cap
SENDA_N0 = 5.757e12      # 1/m^3
SENDA_A = 5.279          # K


def prior_center() -> np.ndarray:
    return 0.5 * (PRIOR_LO + PRIOR_HI)


@dataclass
class Geometry:
    """Nozzle hole geometry and mean flow.  Defaults: ECN Spray G."""
    D: float = 165e-6        # hole diameter [m]
    L_over_D: float = 1.4    # hole length / diameter
    U: float = 150.0         # mean exit velocity [m/s]
    tau_exp: float | None = None   # exposure time to superheat [s]; default L/U
    C_f: float | None = None       # skin-friction coefficient; default Blasius / laminar

    def exposure_time(self) -> float:
        return self.tau_exp if self.tau_exp is not None else self.L_over_D * self.D / self.U


@dataclass
class StateBatch:
    """Vectorised, parameter-independent state quantities for P points."""
    fluid: list
    T: np.ndarray
    p: np.ndarray
    dT: np.ndarray
    dp: np.ndarray
    sigma: np.ndarray
    rho_l: np.ndarray
    nu_l: np.ndarray
    Ja: np.ndarray
    r_c: np.ndarray          # critical radius [m] (inf if not superheated)
    inv_rc: np.ndarray       # 1/r_c (0 if not superheated)
    A: np.ndarray            # MRG inertial velocity
    B: np.ndarray            # MRG thermal constant
    u_tau: np.ndarray
    Re: np.ndarray
    lnJ: np.ndarray          # ln J_hom
    tau_exp: np.ndarray
    L_over_D: np.ndarray
    U: np.ndarray
    D: np.ndarray
    n_sat: np.ndarray        # packing cap: one nucleus per critical-bubble volume
    rho_v: np.ndarray        # saturated vapour density at T_sat(p)
    cp_l: np.ndarray
    h_fg: np.ndarray         # at T_sat(p)
    alpha_l: np.ndarray      # liquid thermal diffusivity
    p_sat: np.ndarray        # p_sat(T_l)
    p_c: np.ndarray
    active: np.ndarray       # dT > 0 mask

    def __len__(self):
        return len(self.T)

    def gp_inputs(self, r0: float = 1e-6) -> np.ndarray:
        """Parameter-free coordinates for the discrepancy GP: (ln(r0/r_c), ln Ja)."""
        x1 = np.log(np.maximum(self.inv_rc * r0, 1e-30))
        x2 = np.log(np.maximum(self.Ja, 1e-30))
        return np.column_stack([x1, x2])


def make_batch(fluids, T, p, geom: Geometry | Sequence[Geometry] = Geometry()) -> StateBatch:
    """Build a StateBatch from arrays of fluid names, temperatures [K] and pressures [Pa]."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(T)
    if isinstance(fluids, str):
        fluids = [fluids] * n
    if isinstance(geom, Geometry):
        geoms = [geom] * n
    else:
        geoms = list(geom)
    cols = {k: np.zeros(n) for k in ("dT", "dp", "sigma", "rho_l", "nu_l", "Ja", "r_c", "inv_rc", "A", "B",
                                     "u_tau", "Re", "lnJ", "tau_exp", "L_over_D", "U", "D", "n_sat",
                                     "rho_v", "cp_l", "h_fg", "alpha_l", "p_sat", "p_c")}
    for i in range(n):
        pr = get_props(fluids[i], T[i], p[i])
        g = geoms[i]
        cols["rho_v"][i] = pr.rho_v
        cols["cp_l"][i] = pr.cp_l
        cols["h_fg"][i] = pr.h_fg
        cols["alpha_l"][i] = pr.alpha_l
        cols["p_sat"][i] = pr.p_sat
        cols["p_c"][i] = pr.p_c
        cols["dT"][i] = pr.dT
        cols["dp"][i] = pr.dp
        cols["sigma"][i] = pr.sigma
        cols["rho_l"][i] = pr.rho_l
        cols["nu_l"][i] = pr.nu_l
        cols["Ja"][i] = pr.Ja
        cols["r_c"][i] = pr.r_c
        cols["inv_rc"][i] = 0.0 if not np.isfinite(pr.r_c) else 1.0 / pr.r_c
        # packing cap: at most one nucleus per critical-bubble volume (4/3 pi r_c^3), with r_c
        # floored at R_FLOOR (1 nm, below which a "bubble" has no meaning) so the cap stays
        # finite and monotone up to the critical point.
        r_eff = max(pr.r_c, R_FLOOR) if np.isfinite(pr.r_c) else np.inf
        cols["n_sat"][i] = 1.0 / (4.0 / 3.0 * np.pi * r_eff ** 3) if np.isfinite(r_eff) else np.inf
        d_eff = max(pr.dp_eff, 0.0)
        cols["A"][i] = np.sqrt(2.0 * d_eff / (3.0 * pr.rho_l)) if d_eff > 0 else 1.0
        cols["B"][i] = np.sqrt(12.0 * pr.alpha_l / np.pi) * pr.Ja if pr.Ja > 0 else 1.0
        Re = g.U * g.D / pr.nu_l
        if g.C_f is not None:
            C_f = g.C_f
        else:
            C_f = 16.0 / Re if Re < 2300 else 0.079 * Re ** -0.25
        cols["u_tau"][i] = g.U * np.sqrt(C_f / 2.0)
        cols["Re"][i] = Re
        cols["lnJ"][i] = ln_J_hom(pr)
        cols["tau_exp"][i] = g.exposure_time()
        cols["L_over_D"][i] = g.L_over_D
        cols["U"][i] = g.U
        cols["D"][i] = g.D
    return StateBatch(fluid=list(fluids), T=T, p=p, active=cols["dT"] > 1e-3, **cols)


def evaluate(sb: StateBatch, theta, dist_bulk: str = "lognormal", dist_wall: str = "lognormal",
             m_bulk: float = 3.0, m_wall: float = 3.0, dlnJ: float = 0.0, components: bool = False):
    """Evaluate n_b [1/m^3 liquid] for parameter sets theta (shape (7,) or (S,7)) on the batch.

    Returns an array of shape (S, P) (squeezed to (P,) for a single theta), or a dict of
    component arrays when components=True.
    """
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    N_v = np.exp(th[:, 0])[:, None]
    r_v = np.exp(th[:, 1])[:, None]
    s_v = th[:, 2][:, None]
    N_s = np.exp(th[:, 3])[:, None]
    r_w = np.exp(th[:, 4])[:, None]
    s_w = th[:, 5][:, None]
    We_d = np.exp(th[:, 6])[:, None]

    inv_rc = sb.inv_rc[None, :]
    act = sb.active[None, :]

    # (1) pre-existing free nuclei
    n_bulk = N_v * P_act(r_v * inv_rc, dist_bulk, s_v, m_bulk)

    # (2) wall cavities: shear-limited supercritical departure, MRG growth time
    r_shear = (We_d * sb.sigma * sb.nu_l ** 2 / (2.0 * sb.rho_l * sb.u_tau ** 4)) ** (1.0 / 3.0)
    r_c_safe = np.where(sb.active, sb.r_c, 1.0)[None, :]
    r_dep = np.maximum(r_shear, KAPPA_DEP * r_c_safe)
    t_g = mrg_time(r_dep, sb.A[None, :], sb.B[None, :])
    f_dep = np.where(act, 1.0 / t_g, 0.0)
    N_a = N_s * P_act(r_w * inv_rc, dist_wall, s_w, m_wall)
    n_wall = 4.0 * sb.L_over_D[None, :] * N_a * f_dep / sb.U[None, :]

    # (3) homogeneous nucleation over the exposure time
    with np.errstate(over="ignore"):
        n_hom_1d = np.where(sb.active, np.exp(np.minimum(sb.lnJ + dlnJ, 700.0)) * sb.tau_exp, 0.0)
    n_hom = np.broadcast_to(n_hom_1d[None, :], n_bulk.shape)

    # (4) equilibrium-limit cap: beyond one nucleus per critical-bubble volume the liquid is
    # past the spinodal / fully foamed and the closure returns the saturated value (HEM limit).
    raw = n_bulk + n_wall + n_hom
    total = np.minimum(raw, sb.n_sat[None, :])
    if components:
        return {"total": total.squeeze(), "bulk": n_bulk.squeeze(), "wall": n_wall.squeeze(),
                "hom": n_hom.squeeze(), "n_sat": np.broadcast_to(sb.n_sat[None, :], total.shape).squeeze(),
                "capped": (raw > sb.n_sat[None, :]).squeeze(),
                "r_shear": np.broadcast_to(r_shear, total.shape).squeeze(),
                "r_dep": np.broadcast_to(r_dep, total.shape).squeeze(),
                "f_dep": np.broadcast_to(f_dep, total.shape).squeeze()}
    return total.squeeze()


def ln_evaluate(sb: StateBatch, theta, **kw) -> np.ndarray:
    return np.log(np.maximum(evaluate(sb, theta, **kw), 1e-300))


def senda(dT) -> np.ndarray:
    """Legacy correlation n = 5.757e12 exp(-5.279 K / dT) [1/m^3] (Senda et al. 1994)."""
    dT = np.asarray(dT, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(dT > 0, SENDA_N0 * np.exp(-SENDA_A / np.where(dT > 0, dT, 1.0)), 0.0)


def activation_constant(pr: FluidProps, r_ref: float) -> float:
    """Linearised activation constant a [K] such that P_act = exp(-a/dT) for an exponential
    nuclei population of scale r_ref:  a = 2 sigma T_sat / (delta rho_v h_fg r_ref)."""
    return 2.0 * pr.sigma * pr.T_sat / (pr.delta * pr.rho_v * pr.h_fg * r_ref)


def senda_equivalent_radius(pr: FluidProps, a: float = SENDA_A) -> float:
    """Nuclei size scale r_ref [m] that makes the exponential model identical to
    N0 exp(-a/dT) for this fluid at this saturation state."""
    return 2.0 * pr.sigma * pr.T_sat / (pr.delta * pr.rho_v * pr.h_fg * a)


def guardrail(n: np.ndarray, sb: StateBatch, calib_range_lnrc: tuple[float, float] | None = None,
              n_min: float = 1e6, n_max: float = 1e20):
    """Clip to physical bounds and flag out-of-distribution states for a CFD embedding.

    Returns (n_clipped, ood_flag).  OOD is raised when ln(1 um / r_c) lies outside the
    range covered by the calibration data; the caller then falls back to the prior mean.
    """
    n_c = np.clip(n, n_min, n_max)
    if calib_range_lnrc is None:
        return n_c, np.zeros(len(sb), dtype=bool)
    x = np.log(np.maximum(sb.inv_rc * 1e-6, 1e-30))
    ood = (x < calib_range_lnrc[0]) | (x > calib_range_lnrc[1])
    return n_c, ood
