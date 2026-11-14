"""Homogeneous Relaxation Model (HRM) relaxation time derived from the nucleation closure.

HRM transports the vapour mass fraction x towards its equilibrium value x_eq with a
relaxation time Theta:

    Dx/Dt = (x_eq - x) / Theta,          x_eq = (h_mix - h_l,sat(p)) / h_fg(p) ~ c_p dT0 / h_fg.

The empirical closure in nearly every flashing-injector solver is Downar-Zapolski et al.
(1996).  It has two forms with *different* pressure variables:

    low pressure  (p < 10 bar):  Theta = 6.51e-4 s * alpha^-0.257 * phi^-2.24,  phi = (p_sat - p)/p_sat
    high pressure (p > 10 bar):  Theta = 3.84e-7 s * alpha^-0.54  * psi^-1.76,  psi = (p_sat - p)/(p_c - p_sat)

with p_sat evaluated at the liquid temperature.  Flashing-injector solvers (HRMFoam,
CONVERGE, the Spray G studies in the review) use the high-pressure form, and Theta0 is
routinely re-tuned by many decades for fuel sprays (3.84e-7 -> 3.84e-13 s for Spray G).

Derivation used here (bubble-cloud cell model).  The closure supplies n_b bubbles per unit
liquid volume, activated together when the liquid becomes superheated.  Each grows at

    dR/dt = 1 / (1/A + 2R/B^2),     A = sqrt(2 dp_eff / (3 rho_l)),   B = sqrt(12 alpha_l/pi) Ja,

the harmonic blend of the Rayleigh (inertial) and Plesset-Zwick (thermal) rates, which has
the same two asymptotes as the Mikic-Rohsenow-Griffith solution.  With V = n_b 4/3 pi R^3
the vapour volume per unit liquid volume,

    alpha = V / (1 + V),      x = rho_v V / (rho_l + rho_v V),

and the adiabatic, constant-pressure energy balance consumes the superheat as vapour forms,

    dT = dT0 (1 - x / x_eq),   A ∝ sqrt(dT),   B ∝ dT.

The instantaneous relaxation time is then simply Theta = (x_eq - x) / (dx/dt), which an
HRM solver can evaluate pointwise from (alpha, state, n_b).  Early in the process
(x << x_eq, which holds up to and beyond the breakup void fraction for engine fuels because
rho_v << rho_l) the superheat is unconsumed and Theta has the closed form

    Theta(alpha) = (x_eq - x) (rho_l + rho_v V)^2 / (rho_v rho_l n_b 4 pi R^2 dR/dt),
    R = [3 alpha / (4 pi n_b (1 - alpha))]^{1/3}.

In the thermal regime this reduces to

    Theta ≈ 1 / (24 alpha_l Ja n_b^{2/3}) [3 alpha/(4 pi (1-alpha))]^{-1/3}
          ∝ n_b^{-2/3} Ja^{-1} alpha^{-1/3} alpha_l^{-1},

i.e. the alpha^-0.257 of Downar-Zapolski is the alpha^-1/3 of a fixed population of
thermally growing bubbles, the psi^-2.24 combines Ja ∝ psi with the superheat dependence
of n_b itself, and Theta0 is set by the bubble number density: Theta0 ∝ n_b^{-2/3}.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .closure import StateBatch

DZ_LOW = (6.51e-4, 0.257, 2.24)    # p < 10 bar, pressure variable phi (Downar-Zapolski et al. 1996)
DZ_HIGH = (3.84e-7, 0.54, 1.76)    # p >= 10 bar, pressure variable psi
THETA0_SPRAY_G_TUNED = 3.84e-13    # high-pressure form as tuned for Spray G (review, Section 3.3)


def psi_dz(sb: StateBatch) -> np.ndarray:
    """High-pressure-form pressure variable psi = (p_sat - p)/(p_c - p_sat)."""
    return np.maximum(sb.p_sat - sb.p, 0.0) / (sb.p_c - sb.p_sat)


def phi_dz(sb: StateBatch) -> np.ndarray:
    """Low-pressure-form pressure variable phi = (p_sat - p)/p_sat."""
    return np.maximum(sb.p_sat - sb.p, 0.0) / sb.p_sat


def theta_dz(alpha, pvar, variant="high", theta0=None):
    """Downar-Zapolski relaxation time [s].

    pvar is phi for variant 'low' and psi for variant 'high'; theta0 overrides the published constant.
    """
    t0, a, b = DZ_LOW if variant == "low" else DZ_HIGH
    if theta0 is not None:
        t0 = theta0
    alpha = np.asarray(alpha, dtype=float)
    pvar = np.asarray(pvar, dtype=float)
    with np.errstate(divide="ignore"):
        return t0 * np.where(alpha > 0, alpha, np.nan) ** (-a) * np.where(pvar > 0, pvar, np.nan) ** (-b)


def theta_dz_batch(sb: StateBatch, alpha, variant="high", theta0=None):
    """Downar-Zapolski relaxation time evaluated on a StateBatch with the correct pressure variable."""
    pvar = phi_dz(sb) if variant == "low" else psi_dz(sb)
    return theta_dz(alpha, pvar, variant, theta0)


def x_eq(sb: StateBatch) -> np.ndarray:
    """Equilibrium (adiabatic flash) quality c_p dT / h_fg."""
    return sb.cp_l * np.maximum(sb.dT, 0.0) / sb.h_fg


def R_of_alpha(alpha, n_b):
    """Bubble radius giving void fraction alpha with n_b bubbles per unit liquid volume."""
    alpha = np.asarray(alpha, dtype=float)
    V = alpha / (1.0 - alpha)
    return (3.0 * V / (4.0 * np.pi * np.asarray(n_b, dtype=float))) ** (1.0 / 3.0)


def growth_rate(R, A, B):
    """dR/dt = 1/(1/A + 2R/B^2): Rayleigh for small R, Plesset-Zwick for large R."""
    return 1.0 / (1.0 / A + 2.0 * R / B ** 2)


def theta_phys(sb: StateBatch, n_b, alpha=0.1):
    """Early-stage (unconsumed superheat) relaxation time [s] from the closure.

    n_b: shape (P,) or (S, P); alpha: scalar, (P,) or (S, P).  Returns inf where the state is
    not superheated or where the requested alpha already exceeds the equilibrium vapour volume.
    """
    n_b = np.asarray(n_b, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    V = alpha / (1.0 - alpha)
    xe = x_eq(sb)
    x = sb.rho_v * V / (sb.rho_l + sb.rho_v * V)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (3.0 * V / (4.0 * np.pi * n_b)) ** (1.0 / 3.0)
        Rdot = growth_rate(R, sb.A, sb.B)
        dxdt = sb.rho_v * sb.rho_l * n_b * 4.0 * np.pi * R ** 2 * Rdot / (sb.rho_l + sb.rho_v * V) ** 2
        th = (xe - x) / dxdt
    ok = sb.active & (xe > x)
    return np.where(ok, th, np.inf)


def theta_thermal_limit(sb: StateBatch, n_b, alpha=0.1):
    """Closed-form thermal-regime limit Theta ≈ [24 alpha_l Ja n_b^{2/3} (3 alpha/(4 pi (1-alpha)))^{1/3}]^-1."""
    n_b = np.asarray(n_b, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    g = (3.0 * alpha / (4.0 * np.pi * (1.0 - alpha))) ** (1.0 / 3.0)
    with np.errstate(divide="ignore"):
        th = 1.0 / (24.0 * sb.alpha_l * sb.Ja * n_b ** (2.0 / 3.0) * g)
    return np.where(sb.active, th, np.inf)


def theta0_equivalent(theta, alpha, pvar, variant="high"):
    """Theta0 that makes the Downar-Zapolski form reproduce `theta` at (alpha, pvar);
    pvar is phi for the low-pressure form and psi for the high-pressure form."""
    _, a, b = DZ_LOW if variant == "low" else DZ_HIGH
    return np.asarray(theta) * np.asarray(alpha) ** a * np.asarray(pvar) ** b


def cell_trajectory(sb: StateBatch, i: int, n_b: float, x_stop: float = 0.99, n_out: int = 400):
    """Integrate the bubble-cloud cell model for batch point i.

    Returns dict of arrays over time: t, R, V, alpha, x, x_over_xeq, dT, theta (instantaneous),
    plus scalars x_eq, dT0, t_e (time at which x reaches (1-1/e) x_eq).
    """
    dT0 = float(sb.dT[i])
    if dT0 <= 0:
        raise ValueError("point is not superheated")
    A0, B0 = float(sb.A[i]), float(sb.B[i])
    rho_l, rho_v = float(sb.rho_l[i]), float(sb.rho_v[i])
    xe = float(x_eq(sb)[i])
    r0 = float(sb.r_c[i]) * 2.0

    def x_of_R(R):
        V = n_b * 4.0 / 3.0 * np.pi * R ** 3
        return rho_v * V / (rho_l + rho_v * V), V

    def rhs(t, y):
        R = y[0]
        x, _ = x_of_R(R)
        f = max(1.0 - x / xe, 1e-6)
        A, B = A0 * np.sqrt(f), B0 * f
        return [growth_rate(R, A, B)]

    def done(t, y):
        return x_of_R(y[0])[0] - x_stop * xe
    done.terminal = True
    done.direction = 1

    V_f = xe * rho_l / (rho_v * (1.0 - xe))
    R_f = (3.0 * V_f / (4.0 * np.pi * n_b)) ** (1.0 / 3.0)
    t_max = 50.0 * (R_f / B0) ** 2 + 50.0 * R_f / A0
    sol = solve_ivp(rhs, (0.0, t_max), [r0], events=done, rtol=1e-7, atol=1e-14, dense_output=True,
                    max_step=t_max / 200)
    t_end = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    t = np.concatenate([[0.0], np.logspace(np.log10(max(t_end * 1e-6, 1e-12)), np.log10(t_end), n_out - 1)])
    R = sol.sol(t)[0]
    x, V = x_of_R(R)
    alpha = V / (1.0 + V)
    dT = dT0 * np.maximum(1.0 - x / xe, 0.0)
    f = np.maximum(1.0 - x / xe, 1e-6)
    Rdot = growth_rate(R, A0 * np.sqrt(f), B0 * f)
    dxdt = rho_v * rho_l * n_b * 4.0 * np.pi * R ** 2 * Rdot / (rho_l + rho_v * V) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (xe - x) / dxdt
    k = int(np.argmax(x >= (1.0 - np.exp(-1.0)) * xe))
    t_e = float(t[k]) if x[k] >= (1.0 - np.exp(-1.0)) * xe else np.nan
    return {"t": t, "R": R, "V": V, "alpha": alpha, "x": x, "x_over_xeq": x / xe, "dT": dT, "theta": theta,
            "x_eq": xe, "dT0": dT0, "t_e": t_e}
