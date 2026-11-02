"""Smoothing the OR class weights (experimental, opt-in; ``optimize_weights`` is unchanged).

Why: the leave-one-out threshold objective has its exact minimum at a degenerate OR solution that
puts (almost) all weight on the first two distance classes. The paper's weights decay smoothly. This
module offers regularised alternatives and chooses their strength by K-fold cross-validation over the
target points.

Speed-up: for the OR aggregation the probability of staying below the threshold of a target is a
linear-fractional function of the class weights,

    p_t(w) = sum_c w_c A[c, t] / sum_c w_c n[c, t],

(``A[c, t]``: the CDF value at z_c summed over the neighbours of class c, ``n[c, t]``: their number),
so the objective costs O(classes x targets) instead of building every PMF. The same holds for the
single-bin objective, where the probability of the bin containing the true value is used.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from pyher.infogram import Infogram
from pyher.weights import _initial_weights, fit_logprior, make_objective


class FastOrObjective:
    """Leave-one-out DKL of the OR aggregation as a function of the class weights.

    ``objective='threshold'``: DKL of the binarised CDF at ``z_thresh`` (sHER_E_04);
    ``objective='bin'``: DKL of the single bin that contains the true value (sHER_E_03).
    In both cases the probability that enters the log is linear-fractional in the weights."""

    def __init__(self, z_cal, ig: Infogram):
        cfg = ig.cfg
        z_cal = np.asarray(z_cal, dtype=float).ravel()
        obj = make_objective(z_cal, ig)
        edges = ig.edges_z
        self.kind = cfg.objective
        self.n = obj.n_in_class  # [R, N]
        if self.kind == "threshold":
            if cfg.z_thresh is None:
                raise ValueError("objective='threshold' needs cfg.z_thresh")
            thr = cfg.z_thresh
            k = int(np.searchsorted(edges, thr, side="right") - 1)  # edges[k] <= thr < edges[k+1]
            cdf_weight = np.zeros(ig.n_bins_z)
            cdf_weight[:k] = 1.0
            cdf_weight[k] = (thr - edges[k]) / (edges[k + 1] - edges[k])
            self.A = obj.S @ cdf_weight  # [R, N]
            self.below = z_cal < thr
        elif self.kind == "bin":
            idx = np.searchsorted(edges, z_cal, side="right") - 1  # edges[i] <= z < edges[i+1]
            ok = (idx >= 0) & (idx < ig.n_bins_z) & (z_cal < edges[-1])
            idx = np.where(ok, idx, 0)
            self.A = np.where(ok[None, :], np.take_along_axis(obj.S, idx[None, :, None], axis=2)[:, :, 0],
                              np.nan)
        else:
            raise ValueError(f"unknown objective {self.kind!r}")
        self.R, self.N = self.A.shape
        self.centers = ig.bin_centers_distance_classes_range

    def p_below(self, w):
        """Probability that enters the log score (below the threshold / true bin)."""
        w = np.asarray(w, dtype=float)
        return (w @ self.A) / (w @ self.n)

    def loss_per_target(self, w):
        p = self.p_below(w)
        with np.errstate(divide="ignore"):
            if self.kind == "bin":
                return -np.log2(p)
            return np.where(self.below, -np.log2(p), -np.log2(1 - p))

    def __call__(self, w, targets=None):
        loss = self.loss_per_target(w)
        return float(loss.mean() if targets is None else loss[targets].mean())


# ----------------------------------------------------------------------------------------------
# smoothing methods
# ----------------------------------------------------------------------------------------------
@dataclass
class SmoothSpec:
    """``method``: 'none' | 'ratio' | 'diff' | 'prior' | 'powerlaw'; ``param``: its strength
    (ratio: minimum w[i+1]/w[i] in [0, 1); diff / prior / logprior: penalty weight lambda;
    powerlaw: unused). 'logprior' shrinks log(w) towards the 1/distance guess in log space, which keeps
    the poorly identified far classes on that guess (what an early-stopped optimiser started there does)."""

    method: str
    param: float = 0.0

    def label(self):
        return self.method if self.method in ("none", "powerlaw") else f"{self.method}={self.param:g}"


def _penalty(spec: SmoothSpec, w, prior):
    wn = w / w.sum()
    if spec.method == "diff":
        return spec.param * np.sum(np.diff(wn) ** 2)
    if spec.method == "prior":
        return spec.param * np.sum((wn - prior) ** 2)
    return 0.0


def fit_or_weights(obj: FastOrObjective, spec: SmoothSpec, targets=None, x0=None):
    """Class weights (normalised to sum 1) minimising the objective under the chosen smoothing.

    Returns ``(w, value)`` where ``value`` is the penalised training objective."""
    R = obj.R
    prior = _initial_weights(obj.centers)
    if spec.method == "powerlaw":
        return _fit_powerlaw(obj, targets)
    if spec.method == "logprior":
        return _fit_logprior(obj, spec.param, targets, prior)

    def full(u):
        return np.concatenate(([1.0], u))

    def f(u):
        w = full(u)
        return obj(w, targets) + _penalty(spec, w, prior)

    lb = np.full(R - 1, 1e-6)
    if spec.method == "ratio":
        lb[0] = max(lb[0], spec.param)
    A = -np.eye(R - 1) + np.diag(np.ones(R - 2), 1)  # u[i+1] <= u[i] (rows 0..R-3)
    cons = [LinearConstraint(A[:-1], -np.inf, 0.0)]
    if spec.method == "ratio" and spec.param > 0:
        Rm = np.eye(R - 1, k=1) - spec.param * np.eye(R - 1)  # u[i+1] - rho*u[i] >= 0
        cons.append(LinearConstraint(Rm[:-1], 0.0, np.inf))
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    starts.append(prior[1:] / prior[0])
    best = None
    for s0 in starts:
        s0 = np.clip(s0, lb, 1.0)
        res = minimize(f, s0, method="SLSQP", bounds=Bounds(lb, np.ones(R - 1)), constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    w = full(best.x)
    return w / w.sum(), float(best.fun)


def _fit_logprior(obj: FastOrObjective, lam: float, targets, prior):
    w, val, _ = fit_logprior(lambda w: obj(w, targets), obj.R, prior, lam)
    return w, val


def _powerlaw(centers, a, p):
    return (1.0 + centers / a) ** (-p)


def _fit_powerlaw(obj: FastOrObjective, targets):
    """w_c = (1 + d_c / a)^-p, two parameters: a coarse grid search followed by Nelder-Mead."""
    c = obj.centers
    grid_a = np.geomspace(0.01, 5, 40)
    grid_p = np.linspace(0, 6, 61)
    vals = np.array([[obj(_powerlaw(c, a, p), targets) for p in grid_p] for a in grid_a])
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    res = minimize(lambda q: obj(_powerlaw(c, np.exp(q[0]), max(q[1], 0)), targets),
                   [np.log(grid_a[i]), grid_p[j]], method="Nelder-Mead")
    w = _powerlaw(c, np.exp(res.x[0]), max(res.x[1], 0))
    return w / w.sum(), float(res.fun)


# ----------------------------------------------------------------------------------------------
# cross-validation of the smoothing strength
# ----------------------------------------------------------------------------------------------
@dataclass
class CvResult:
    method: str
    params: list
    cv_mean: np.ndarray  # mean held-out loss per parameter value
    cv_se: np.ndarray  # standard error over the folds
    train: np.ndarray  # loss on all targets of the weights fitted on all targets
    best: int  # index of the minimum CV loss
    one_se: int  # smoothest parameter within one SE of the minimum


def cross_validate(obj: FastOrObjective, method: str, params, *, k: int = 5, seed: int = 0) -> CvResult:
    """K-fold cross-validation over the target points: fit on k-1 folds, score the held-out fold.
    ``params`` must be ordered from the least to the most smoothing."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(obj.N), k)
    cv = np.empty((len(params), k))
    train = np.empty(len(params))
    for ip, prm in enumerate(params):
        spec = SmoothSpec(method, prm)
        for ik, test in enumerate(folds):
            tr = np.setdiff1d(np.arange(obj.N), test)
            w, _ = fit_or_weights(obj, spec, tr)
            cv[ip, ik] = obj(w, test)
        w, _ = fit_or_weights(obj, spec)
        train[ip] = obj(w)
    mean, se = cv.mean(axis=1), cv.std(axis=1, ddof=1) / np.sqrt(k)
    best = int(np.argmin(mean))
    ok = np.flatnonzero(mean <= mean[best] + se[best])
    return CvResult(method, list(params), mean, se, train, best, int(ok.max()))
