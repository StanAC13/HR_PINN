"""Link from bubble number density to Lagrangian flash-breakup quantities.

Given n_b bubbles per unit liquid volume growing at the parcel superheat, the
void fraction alpha = n_b V_b / (1 + n_b V_b) reaches the critical value
alpha_c (0.55, Kawano et al. 2004) when the bubble radius is

    R_b = [ 3 alpha_c / (4 pi n_b (1 - alpha_c)) ]^{1/3}.

The time to reach R_b under MRG growth is the flash-breakup time t_b.  At that
moment the liquid is a foam whose lamellae have mean thickness
h = (1 - alpha_c)/alpha_c * R_b / 3 (liquid volume per bubble over bubble area).
Lamella rupture sets the scale of the child droplets (d ~ few h); only h is
reported here because the rupture-to-droplet factor is itself uncertain.
"""
from __future__ import annotations

import numpy as np

from .closure import StateBatch
from .growth import mrg_time

ALPHA_CRIT = 0.55


def flash_breakup(sb: StateBatch, n_b, alpha_c: float = ALPHA_CRIT):
    """Return dict with R_b [m], t_b [s], h [m] for n_b of shape (...,P) or (P,)."""
    n_b = np.asarray(n_b, dtype=float)
    R_b = (3.0 * alpha_c / (4.0 * np.pi * n_b * (1.0 - alpha_c))) ** (1.0 / 3.0)
    t_b = mrg_time(R_b, sb.A, sb.B)
    h = (1.0 - alpha_c) / alpha_c * R_b / 3.0
    return {"R_b": R_b, "t_b": t_b, "h": h}
