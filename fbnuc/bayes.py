"""Bayesian calibration of the physics-prior closure with a GP discrepancy term.

Observation model (Kennedy-O'Hagan with the physics closure as the mean):

    y_i = ln n_obs,i = ln n_phys(x_i; xi) + delta(x_i) + eps_i,
    delta ~ GP(0, s_gp^2 k_SE(x, x'; ell)),      eps_i ~ N(0, sigma_obs,i^2)

with x = (ln(1 um / r_c), ln Ja) 
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .closure import StateBatch, ln_evaluate, PRIOR_LO, PRIOR_HI, prior_center

LN_SGP_LO, LN_SGP_HI = np.log(0.05), np.log(1.5)


def _logit(u):
    return np.log(u) - np.log1p(-u)


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def sq_exp_kernel(X1, X2, s, ell):
    d2 = (((X1[:, None, :] - X2[None, :, :]) / ell) ** 2).sum(-1)
    return s * s * np.exp(-0.5 * d2)


@dataclass
class Calibration:
    sb: StateBatch
    y: np.ndarray                      # ln n_obs
    sigma_obs: np.ndarray              # per-point noise sd in ln space
    ell: tuple = (1.0, 1.0)
    use_gp: bool = True
    dist_bulk: str = "lognormal"
    dist_wall: str = "lognormal"

    def __post_init__(self):
        self.lo = np.append(PRIOR_LO, LN_SGP_LO)
        self.hi = np.append(PRIOR_HI, LN_SGP_HI)
        self.d = len(self.lo)
        self.X = self.sb.gp_inputs()
        self.sigma_obs = np.broadcast_to(np.asarray(self.sigma_obs, dtype=float), self.y.shape).copy()

    # ---- transforms ---------------------------------------------------------
    def to_x(self, z):
        return self.lo + (self.hi - self.lo) * _expit(z)

    def to_z(self, x):
        u = np.clip((x - self.lo) / (self.hi - self.lo), 1e-9, 1 - 1e-9)
        return _logit(u)

    def log_jac(self, z):
        s = _expit(z)
        return np.sum(np.log(self.hi - self.lo) + np.log(s) + np.log1p(-s))

    # ---- likelihood -----------------------------------------------------------
    def mean(self, theta):
        return ln_evaluate(self.sb, theta, dist_bulk=self.dist_bulk, dist_wall=self.dist_wall)

    def cov(self, s_gp):
        K = np.diag(self.sigma_obs ** 2)
        if self.use_gp:
            K = K + sq_exp_kernel(self.X, self.X, s_gp, np.asarray(self.ell))
        return K

    def log_like(self, x):
        theta, ln_sgp = x[:7], x[7]
        r = self.y - self.mean(theta)
        if not np.all(np.isfinite(r)):
            return -np.inf
        K = self.cov(np.exp(ln_sgp))
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = cho_solve(cf, r)
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        return -0.5 * (r @ alpha + logdet + len(r) * np.log(2 * np.pi))

    def log_post_z(self, z):
        if np.any(np.abs(z) > 40):
            return -np.inf
        x = self.to_x(z)
        return self.log_like(x) + self.log_jac(z)   # uniform prior on x

    # ---- MAP + MCMC -----------------------------------------------------------
    def find_map(self, n_starts=6, seed=0):
        rng = np.random.default_rng(seed)
        best = None
        starts = [self.to_z(np.append(prior_center(), np.log(0.3)))]
        for _ in range(n_starts - 1):
            starts.append(self.to_z(self.lo + (self.hi - self.lo) * rng.uniform(0.05, 0.95, self.d)))
        for z0 in starts:
            res = minimize(lambda z: -self.log_post_z(z), z0, method="Nelder-Mead",
                           options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        return best.x

    def run_mcmc(self, n_iter=40000, seed=0, adapt_start=2000, z0=None, verbose=False):
        rng = np.random.default_rng(seed)
        if z0 is None:
            z0 = self.find_map(seed=seed)
        d = self.d
        z = z0.copy()
        lp = self.log_post_z(z)
        chain = np.zeros((n_iter, d))
        lps = np.zeros(n_iter)
        cov = np.eye(d) * 0.05
        sd = 2.38 ** 2 / d
        acc = 0
        mean_run = z.copy()
        cov_run = np.eye(d) * 0.05
        for i in range(n_iter):
            if i >= adapt_start and i % 50 == 0:
                cov = sd * cov_run + 1e-6 * np.eye(d)
            prop = rng.multivariate_normal(z, cov)
            lpp = self.log_post_z(prop)
            if np.log(rng.uniform()) < lpp - lp:
                z, lp = prop, lpp
                acc += 1
            chain[i] = z
            lps[i] = lp
            # running mean / covariance (Welford)
            n = i + 1
            dz = z - mean_run
            mean_run = mean_run + dz / n
            cov_run = cov_run + (np.outer(dz, z - mean_run) - cov_run) / n if n > 1 else cov_run
        self.accept_rate = acc / n_iter
        self.chain_z = chain
        self.chain_lp = lps
        burn = n_iter // 2
        self.samples = np.array([self.to_x(zz) for zz in chain[burn::max(1, (n_iter - burn) // 2000)]])
        if verbose:
            print(f"acceptance {self.accept_rate:.2f}, {len(self.samples)} posterior samples")
        return self.samples

    # ---- prediction -----------------------------------------------------------
    def predict(self, sb_new: StateBatch, samples=None, n_draw=400, seed=1, with_gp=True):
        """Posterior predictive draws of ln n at new states.  Returns (S, P_new)."""
        rng = np.random.default_rng(seed)
        samples = self.samples if samples is None else samples
        idx = rng.choice(len(samples), size=min(n_draw, len(samples)), replace=False)
        Xn = sb_new.gp_inputs()
        out = np.zeros((len(idx), len(sb_new)))
        for k, j in enumerate(idx):
            x = samples[j]
            theta, s_gp = x[:7], np.exp(x[7])
            mu_new = ln_evaluate(sb_new, theta, dist_bulk=self.dist_bulk, dist_wall=self.dist_wall)
            if with_gp and self.use_gp:
                r = self.y - self.mean(theta)
                K = self.cov(s_gp)
                cf = cho_factor(K, lower=True)
                Ks = sq_exp_kernel(Xn, self.X, s_gp, np.asarray(self.ell))
                m = mu_new + Ks @ cho_solve(cf, r)
                v = s_gp ** 2 - np.einsum("ij,ij->i", Ks, cho_solve(cf, Ks.T).T)
                v = np.maximum(v, 1e-12)
                out[k] = m + np.sqrt(v) * rng.standard_normal(len(sb_new))
            else:
                out[k] = mu_new
        return out


# ---------------------------------------------------------------------------
# Pure-data baselines (no physics)
# ---------------------------------------------------------------------------
def quadratic_fit(lnx, y):
    c = np.polyfit(lnx, y, 2)
    return lambda lnx_new: np.polyval(c, lnx_new)


def gp_fit_1d(lnx, y, sigma_obs, mean_mode="zero"):
    """Zero- or constant-mean GP in ln dT with ML-II hyper-parameters.
    Returns predictor(lnx_new) -> (mean, sd)."""
    lnx = np.asarray(lnx, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)
    m0 = 0.0 if mean_mode == "zero" else y.mean()
    r = y - m0
    sig = np.broadcast_to(np.asarray(sigma_obs, dtype=float), y.shape)

    def nll(h):
        s, ell = np.exp(h)
        K = sq_exp_kernel(lnx, lnx, s, ell) + np.diag(sig ** 2)
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e30
        a = cho_solve(cf, r)
        return 0.5 * (r @ a + 2 * np.sum(np.log(np.diag(cf[0]))))

    best = None
    for h0 in ([np.log(np.std(y) + 1e-3), np.log(0.5)], [np.log(5.0), np.log(1.0)], [np.log(1.0), np.log(2.0)]):
        res = minimize(nll, h0, method="Nelder-Mead")
        if best is None or res.fun < best.fun:
            best = res
    s, ell = np.exp(best.x)
    K = sq_exp_kernel(lnx, lnx, s, ell) + np.diag(sig ** 2)
    cf = cho_factor(K, lower=True)
    a = cho_solve(cf, r)

    def predict(lnx_new):
        Xn = np.asarray(lnx_new, dtype=float)[:, None]
        Ks = sq_exp_kernel(Xn, lnx, s, ell)
        m = m0 + Ks @ a
        v = s ** 2 - np.einsum("ij,ij->i", Ks, cho_solve(cf, Ks.T).T)
        return m, np.sqrt(np.maximum(v, 1e-12))

    return predict


def coverage(draws, truth, q=0.9):
    lo = np.quantile(draws, (1 - q) / 2, axis=0)
    hi = np.quantile(draws, 1 - (1 - q) / 2, axis=0)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def log10_rmse(pred_ln, truth_ln):
    return float(np.sqrt(np.mean((pred_ln - truth_ln) ** 2)) / np.log(10.0))
