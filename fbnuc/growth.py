"""Single-bubble growth in a uniformly superheated liquid.

Mikic-Rohsenow-Griffith (1970) composite solution spanning the inertia-controlled
(Rayleigh) and heat-diffusion-controlled (Plesset-Zwick) regimes:

    R+ = (2/3) [ (t+ + 1)^{3/2} - (t+)^{3/2} - 1 ],
    R+ = R A / B^2,   t+ = t A^2 / B^2,
    A  = sqrt( 2 (p_v* - p) / (3 rho_l) )          (inertial growth velocity)
    B  = sqrt( 12 alpha_l / pi ) Ja                (thermal growth constant)

Limits: R ~ A t for t+ << 1 (Rayleigh), R ~ B sqrt(t) for t+ >> 1 (Plesset-Zwick).
The inverse t(R) is obtained with a vectorised, safeguarded Newton iteration.
"""
from __future__ import annotations

import numpy as np


def _Rplus(tp):
    """(2/3)[(t+1)^1.5 - t^1.5 - 1], written to avoid cancellation at large t."""
    tp = np.asarray(tp, dtype=float)
    safe = np.where(tp > 0, tp, 1.0)
    big = safe ** 1.5 * np.expm1(1.5 * np.log1p(1.0 / safe))
    return (2.0 / 3.0) * (np.where(tp > 0, big, 1.0) - 1.0)


def mrg_radius(t, A, B):
    """Bubble radius R(t) [m] from the MRG solution (array-safe)."""
    t = np.asarray(t, dtype=float)
    tp = t * A ** 2 / B ** 2
    return _Rplus(tp) * B ** 2 / A


def mrg_time(R, A, B, iters: int = 30):
    """Time t(R) [s] for the MRG bubble to reach radius R (array-safe)."""
    R = np.asarray(R, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Rp = R * A / B ** 2
    # initial guess from the two asymptotes (always to the left of the root)
    tp = np.where(Rp < 1.0, Rp, Rp ** 2)
    for _ in range(iters):
        g = _Rplus(tp) - Rp
        dg = 1.0 / (np.sqrt(tp + 1.0) + np.sqrt(tp))   # = sqrt(t+1) - sqrt(t), stable form
        tp_new = tp - g / dg
        # g is concave: a Newton step from the left may overshoot; damp it
        tp = np.where(tp_new < 0.0, 0.5 * tp, tp_new)
    return tp * B ** 2 / A ** 2
