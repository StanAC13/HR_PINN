### Activation of pre-existing nuclei.

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def P_act(Lambda, dist: str = "lognormal", s=0.7, m=3.0):
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
