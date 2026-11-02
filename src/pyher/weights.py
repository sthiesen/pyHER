"""HER2: weight optimisation by leave-one-out cross validation (port of ``f_her2_weight_thres.m``
and, via ``objective='bin'``, ``f_her2_weight.m``).

Speed-up compared with the MATLAB code: the PMF a neighbour contributes depends only on its
distance class and its own z value, so for every target the neighbours of one class can be summed
up once (``S``, ``L`` below). Each evaluation of the optimisation objective is then a small
weighted sum over classes instead of a loop over all N x N neighbour/target pairs.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from pyher.core import linear_aggregation, loglinear_aggregation, matlab_colon, performance_prob
from pyher.infogram import Infogram
from pyher.pmfs import bins_shift, window_pmf


# ----------------------------------------------------------------------------------------------
# helpers shared with HER3
# ----------------------------------------------------------------------------------------------
def classify_distances(dist: np.ndarray, edges_range: np.ndarray, *, prediction: bool = False):
    """Distance class (0-based) of each distance, classes being ``(e_c, e_c+1]``.

    Optimisation (``prediction=False``): distance 0 (the target itself) gets -1, MATLAB's class 0.
    Prediction: distance 0 gets the first class and distances beyond the last edge the last class.
    """
    n_classes = len(edges_range) - 1
    cls = np.searchsorted(edges_range, dist, side="left") - 1
    if prediction:
        return np.clip(cls, 0, n_classes - 1)
    return np.where((cls >= 0) & (cls < n_classes), cls, -1)


def neighbour_pmfs_by_class(z_nb: np.ndarray, ig: Infogram) -> np.ndarray:
    """z PMF of each neighbour for every distance class: array ``[n_nb, n_classes, n_bins_z]``."""
    R = ig.n_classes_range
    bs = bins_shift(z_nb, ig)
    pmf = ig.pmf_diff_z_by_class_obs_range_shift  # [R, n_bins_shift + 1]
    out = np.empty((len(bs), R, ig.n_bins_z))
    for i, b in enumerate(bs):
        for c in range(R):
            out[i, c] = window_pmf(pmf[c], int(b), ig)[0]
    return out


# ----------------------------------------------------------------------------------------------
# result container
# ----------------------------------------------------------------------------------------------
@dataclass
class Weights:
    best_w_OR: np.ndarray
    best_w_AND: np.ndarray
    w_OR_slope: np.ndarray
    w_AND_slope: np.ndarray
    best_alpha: float
    best_beta: float
    DKL_w_OR: float
    DKL_w_AND: float
    DKL_w_alpha_beta: float
    n_neighbor_aggreg: int
    idx_opt_nn: np.ndarray  # [n_neighbor_aggreg, N], 0-based neighbour indices of every target
    n_pairs_by_class_obs: np.ndarray  # [n_classes_range + 1, N]; row 0 = class 0 (target itself)
    nmean_pairs_by_class: np.ndarray
    target_idx_zero_neigh: np.ndarray
    result_or: object = None  # scipy OptimizeResult
    result_and: object = None
    nugget: bool = True  # slopes were computed with the nugget / NOnugget rule; the predictor checks it
    objective: str = "threshold"


# ----------------------------------------------------------------------------------------------
# objective functions (leave-one-out DKL of the OR / AND aggregation)
# ----------------------------------------------------------------------------------------------
class _LooObjective:
    """Leave-one-out DKL as a function of class weights."""

    def __init__(self, z_cal, cls, P, ig: Infogram, objective: str, z_thresh):
        N, R = len(z_cal), ig.n_classes_range
        self.z, self.ig, self.objective, self.z_thresh = z_cal, ig, objective, z_thresh
        self.R = R
        self.edges_thres = (np.array([ig.edges_z[0], z_thresh, ig.edges_z[-1]])
                            if objective == "threshold" else None)
        # S[c, t, :] = sum_i P_i,c over the neighbours i of class c of target t; L the same for log P
        self.S = np.zeros((R, N, ig.n_bins_z))
        self.L = np.zeros((R, N, ig.n_bins_z))
        self.n_in_class = np.zeros((R, N))
        logP = np.log(P)
        for t in range(N):
            for c in range(R):
                members = np.flatnonzero(cls[:, t] == c)
                if members.size:
                    self.S[c, t] = P[members, c].sum(axis=0)
                    self.L[c, t] = logP[members, c].sum(axis=0)
                    self.n_in_class[c, t] = members.size

    def or_pmfs(self, w):
        num = np.einsum("c,ctb->tb", w, self.S)
        return num / num.sum(axis=1, keepdims=True)

    def and_pmfs(self, w):
        lp = np.einsum("c,ctb->tb", w, self.L)
        e = np.exp(lp - lp.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def dkl(self, pmfs):
        return score_pmfs(pmfs, self.z, self.ig.edges_z, self.objective, self.z_thresh)

    def f_or(self, w):
        return self.dkl(self.or_pmfs(w))

    def f_and(self, w):
        return self.dkl(self.and_pmfs(w))


def threshold_probabilities(pmfs: np.ndarray, edges_z: np.ndarray, z_thresh: float) -> np.ndarray:
    """``[P(z <= thresh), 1 - P]`` for each PMF row, interpolating the CDF linearly."""
    cmf = np.concatenate((np.zeros((pmfs.shape[0], 1)), np.cumsum(pmfs, axis=1)), axis=1)
    p = np.array([np.interp(z_thresh, edges_z, row) for row in cmf])
    return np.column_stack((p, 1 - p))


def score_pmfs(pmfs, z_true, edges_z, objective: str, z_thresh=None) -> float:
    if objective == "threshold":
        probs = threshold_probabilities(pmfs, edges_z, z_thresh)
        edges = np.array([edges_z[0], z_thresh, edges_z[-1]])
        return performance_prob(z_true, probs, edges)
    return performance_prob(z_true, pmfs, edges_z)


def make_objective(z_cal, ig: Infogram) -> "_LooObjective":
    """Leave-one-out objective of the OR / AND class weights for a fitted infogram."""
    z_cal = np.asarray(z_cal, dtype=float).ravel()
    cls = classify_distances(ig.mat_euc_distance_xy, ig.edges_distance_classes_range)
    P = neighbour_pmfs_by_class(z_cal, ig)
    return _LooObjective(z_cal, cls, P, ig, ig.cfg.objective, ig.cfg.z_thresh)


# ----------------------------------------------------------------------------------------------
# optimisation
# ----------------------------------------------------------------------------------------------
def _monotone_constraint(n: int, n_rows: int) -> list:
    """w[i+1] <= w[i] for i = 0 .. n_rows-1  (MATLAB: A = -eye + diag(ones, 1), last row dropped).
    A list (empty when there is no constraint, e.g. a single free weight)."""
    if n_rows <= 0:
        return []
    A = -np.eye(n) + np.diag(np.ones(n - 1), 1)
    return [LinearConstraint(A[:n_rows], -np.inf, 0.0)]


def _initial_weights(centers: np.ndarray) -> np.ndarray:
    w = (1.0 / centers) / np.sum(1.0 / centers)
    return w


#: Strength of the default shrinkage of the OR weights for ``objective='threshold'`` (see
#: ``fit_logprior``). Calibrated on the Jura lead example so that the optimum has the same leave-one-out
#: loss as the ``fmincon`` solution of the paper (0.882 bit); the penalty is in the same bit units as the
#: objective, so it is a reasonable starting value for other data, but ``pyher.smooth.cross_validate``
#: can be used to tune it.
DEFAULT_OR_SHRINKAGE = 0.0043


def resolve_or_shrinkage(or_shrinkage, objective: str):
    """``"auto"`` (the default): ``DEFAULT_OR_SHRINKAGE`` for the threshold objective, none for the single-bin
    objective, whose exact optimum already reproduces MATLAB (sHER_E_03) and the HESS 2020 paper.
    ``None`` / 0 / a number are used as given."""
    if isinstance(or_shrinkage, str):
        if or_shrinkage != "auto":
            raise ValueError(f"or_shrinkage must be 'auto', None or a number, got {or_shrinkage!r}")
        return DEFAULT_OR_SHRINKAGE if objective == "threshold" else None
    return or_shrinkage


def fit_logprior(f, R: int, prior: np.ndarray, lam: float):
    """Minimise ``f(w) + lam * sum_c (log w_c - log w_0 - (log prior_c - log prior_0))^2`` over
    non-increasing weights ``w`` (``w_0`` fixed to 1), i.e. shrink the class weights towards the
    1/distance guess ``prior`` in log space. The far classes hardly influence the objective, so they
    stay on that guess and the weights decay smoothly instead of collapsing onto the first classes.

    ``lam = 0`` is the unconstrained optimum. Returns ``(w normalised to sum 1, penalised value,
    OptimizeResult)``."""
    ref = np.log(prior / prior[0])

    def objective(v):  # v[c] = log(w_c / w_0), c = 1..R-1
        full = np.concatenate(([0.0], v))
        return f(np.exp(full)) + lam * np.sum((full - ref) ** 2)

    cons = _monotone_constraint(R - 1, R - 2)  # v[i+1] <= v[i]
    lo = np.full(R - 1, np.log(1e-6))
    starts = [ref[1:], np.log(np.maximum(1e-6, np.concatenate(([0.5], 0.3 ** np.arange(1, R - 1)))))]
    best = None
    for v0 in starts:
        v0 = np.minimum(np.maximum(v0, lo), 0.0)
        res = minimize(objective, v0, method="SLSQP", bounds=Bounds(lo, np.zeros(R - 1)),
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    w = np.minimum.accumulate(np.exp(np.concatenate(([0.0], best.x))))  # undo solver tolerance violations
    return w / w.sum(), float(best.fun), best


def _slopes(best_w: np.ndarray, edges_range: np.ndarray, nugget: bool = True) -> np.ndarray:
    """Slope of the class weights used to interpolate them with distance.

    nugget: ``(w[i+1]-w[i]) / (e[i+1]-e[i])`` and 0 for the last class. NOnugget (the paper):
    ``(w[i]-w[i-1]) / (e[i]-e[i-1])`` for i >= 1; the first class (slope NaN) is handled by the 1/d rule.
    """
    s = np.zeros(len(best_w))
    if nugget:
        s[:-1] = np.diff(best_w) / np.diff(edges_range)[: len(best_w) - 1]
    else:
        s[:] = np.nan
        s[1:] = np.diff(best_w) / np.diff(edges_range)[: len(best_w) - 1]
    return s


def uses_continuous_alpha_beta(cfg) -> bool:
    """MATLAB: only ``f_her2_weight`` (objective 'bin', nugget) interpolates the class weights with
    distance in the alpha/beta search; all other variants use the discrete class weights."""
    if cfg.continuous_alpha_beta is not None:
        return bool(cfg.continuous_alpha_beta)
    return cfg.objective == "bin" and cfg.nugget


def search_alpha_beta(z_cal, ig: Infogram, dist, cls, P, best_w_OR, best_w_AND, alpha_beta=None):
    """Step 8 of HER2: grid search of the exponents (alpha, beta) of the log-linear combination of
    the AND and OR PMFs, using the ``n_neighbor`` nearest neighbours of each left-out point.
    ``alpha_beta=(alpha, beta)`` skips the search and only evaluates that pair."""
    cfg = ig.cfg
    N = len(z_cal)
    n_nn = min(cfg.n_neighbor, N - 1)  # a point is never its own neighbour
    # leave-one-out neighbours: the target itself and points at exactly the same location are excluded, as in
    # the class-weight optimisation (distance 0 = class 0 there)
    d_nb = np.where(dist > 0, dist, np.inf)
    order = np.argsort(d_nb, axis=0, kind="stable")
    idx_nn = order[:n_nn]
    pmf_and = np.empty((N, ig.n_bins_z))
    pmf_or = np.empty((N, ig.n_bins_z))
    continuous = uses_continuous_alpha_beta(cfg)
    if continuous:  # weight = class weight + slope * (distance - lower class edge), as in the predictor
        edges = ig.edges_distance_classes_range
        slope_or = _slopes(best_w_OR, edges, True)
        slope_and = _slopes(best_w_AND, edges, True)
    for t in range(N):
        nb = idx_nn[:, t]
        c = cls[nb, t]
        pm = P[nb, c]  # [n_nn, n_bins_z]
        if continuous:
            d = dist[nb, t]
            w_or_t = best_w_OR[c] + slope_or[c] * (d - edges[c])
            w_and_t = best_w_AND[c] + slope_and[c] * (d - edges[c])
        else:
            w_or_t, w_and_t = best_w_OR[c], best_w_AND[c]
        pmf_and[t] = loglinear_aggregation(pm, w_and_t)
        pmf_or[t] = linear_aggregation(pm, w_or_t / w_or_t.sum())
    log_and, log_or = np.log(pmf_and), np.log(pmf_or)
    alphas = matlab_colon(1.0, -0.05, 0.0)  # MATLAB 1:-.05:0
    pairs = ([tuple(alpha_beta)] if alpha_beta is not None
             else [(a, b) for a in alphas for b in alphas])
    dkl_grid, combos = [], []
    for a, b in pairs:
        lp = a * log_and + b * log_or
        e = np.exp(lp - lp.max(axis=1, keepdims=True))
        pm = e / e.sum(axis=1, keepdims=True)
        dkl_grid.append(score_pmfs(pm, z_cal, ig.edges_z, cfg.objective, cfg.z_thresh))
        combos.append((a, b))
    dkl_grid = np.array(dkl_grid)
    best = int(np.nanargmin(dkl_grid))  # first minimum: prefers larger alpha, then larger beta
    return combos[best][0], combos[best][1], float(dkl_grid[best]), n_nn, idx_nn


def evaluate_alpha_beta(z_cal, ig: Infogram, best_w_OR, best_w_AND, alpha_beta=None):
    """Run only the (alpha, beta) search (or evaluate a given pair) for given class weights."""
    z_cal = np.asarray(z_cal, dtype=float).ravel()
    dist = ig.mat_euc_distance_xy
    cls = classify_distances(dist, ig.edges_distance_classes_range)
    P = neighbour_pmfs_by_class(z_cal, ig)
    return search_alpha_beta(z_cal, ig, dist, cls, P, np.asarray(best_w_OR), np.asarray(best_w_AND),
                             alpha_beta)


def warn_if_degenerate(best_w_OR: np.ndarray) -> None:
    """The leave-one-out threshold objective has a degenerate minimum that puts all OR weight on the
    first classes; warn when an optimiser found it (the paper's weights decay smoothly instead)."""
    w = np.asarray(best_w_OR, dtype=float)
    if len(w) > 3 and w[2:].sum() < 0.02 * w.sum():
        warnings.warn(
            "OR weights collapsed onto the first two distance classes (%.1f%% of the weight in "
            "the others). This is the exact minimum of the leave-one-out objective but gives much "
            "flatter uncertainty maps than the weights of the paper; use the default "
            "or_shrinkage (or fixed_weights=pyher.reference.JURA_PB_THRESHOLD_WEIGHTS) to reproduce it." %
            (100 * w[2:].sum() / w.sum()), stacklevel=3)


def optimize_weights(
    x_cal, y_cal, z_cal, ig: Infogram, *, method: str = "SLSQP", options: dict | None = None,
    fixed_weights: tuple | None = None, alpha_beta: tuple | None = None,
    or_shrinkage: float | str | None = "auto"
) -> Weights:
    """HER2: optimise the class weights of the OR and AND aggregation and the AND/OR mix
    (alpha, beta) by leave-one-out cross validation on the calibration set.

    ``fixed_weights=(w_or, w_and)`` skips the weight optimisation and uses the given class weights
    (one per class of ``ig``), e.g. the MATLAB/paper weights in ``pyher.reference``;
    ``alpha_beta=(alpha, beta)`` skips the grid search of the AND/OR mix (the paper reports 0.65, 0).
    The leave-one-out DKL of the weights and the slopes used for interpolation are still computed.

    ``or_shrinkage`` shrinks log(OR weights) towards the 1/distance starting guess, which gives smooth,
    slowly decaying weights like those of the SERRA paper. The default ``"auto"`` uses
    ``DEFAULT_OR_SHRINKAGE`` (0.0043) for ``objective='threshold'`` and no shrinkage for ``objective='bin'``. The exact
    minimiser of the leave-one-out objective puts all OR weight on the first two classes; pass
    ``or_shrinkage=None`` (or 0) for that unregularised optimum (a warning is then issued if it
    collapses). The AND weights are never shrunk.
    """
    cfg = ig.cfg
    z_cal = np.asarray(z_cal, dtype=float).ravel()
    N, R = len(z_cal), ig.n_classes_range
    objective = cfg.objective
    if objective == "threshold" and cfg.z_thresh is None:
        raise ValueError("objective='threshold' needs cfg.z_thresh")
    if R < 3:
        raise ValueError(f"only {R - 1} distance class inside the information range: the class weights cannot be "
                         "optimised. Decrease lag_dist or binwidth, or check the data.")
    if len(z_cal) != ig.mat_euc_distance_xy.shape[0]:
        raise ValueError("z_cal must be the calibration data the infogram was fitted on")

    # step 3: class of each neighbour (row) relative to each target (column); -1 = the target itself
    dist = ig.mat_euc_distance_xy
    cls = classify_distances(dist, ig.edges_distance_classes_range)

    # step 4: neighbour cardinality
    counts = np.zeros((R + 1, N), dtype=int)  # row 0: class 0 (target itself)
    for t in range(N):
        counts[:, t] = np.bincount(cls[:, t] + 1, minlength=R + 1)
    nmean = counts.mean(axis=1)
    zero_neigh = np.flatnonzero(counts[-1] == counts[1:].sum(axis=0))

    # step 5: z PMF every neighbour contributes, per distance class
    P = neighbour_pmfs_by_class(z_cal, ig)  # [N, R, n_bins_z]
    obj = _LooObjective(z_cal, cls, P, ig, objective, cfg.z_thresh)

    res_or = res_and = None
    if fixed_weights is not None:
        w_or_in, w_and_in = (np.asarray(w, dtype=float).ravel() for w in fixed_weights)
        if len(w_or_in) != R or len(w_and_in) != R:
            raise ValueError(f"fixed_weights need one weight per class ({R}), got "
                             f"{len(w_or_in)} and {len(w_and_in)}")
        best_w_OR = w_or_in / w_or_in.sum()
        best_w_AND = w_and_in
    else:
        # step 6: OR weights (first class fixed to 1)
        w0 = _initial_weights(ig.bin_centers_distance_classes_range)
        or_shrinkage = resolve_or_shrinkage(or_shrinkage, objective)
        if or_shrinkage:
            from pyher.smooth import FastOrObjective  # same value as obj.f_or, linear-fractional in w
            best_w_OR, _, res_or = fit_logprior(FastOrObjective(z_cal, ig), R, w0, float(or_shrinkage))
        else:
            # two starts (the 1/distance guess and a geometric decay); keep the better optimum
            starts = [w0[1:] / w0[0], 0.3 ** np.arange(1, R)]
            fits = [minimize(
                lambda v: obj.f_or(np.concatenate(([1.0], v))), np.clip(x0, 1e-6, 1.0), method=method,
                bounds=Bounds(1e-6 * np.ones(R - 1), np.ones(R - 1)),
                constraints=_monotone_constraint(R - 1, R - 2), options=options or {"maxiter": 500},
            ) for x0 in starts]
            res_or = min(fits, key=lambda r: r.fun)
            best_w_OR = np.minimum.accumulate(np.concatenate(([1.0], res_or.x)))
            best_w_OR = best_w_OR / best_w_OR.sum()
            warn_if_degenerate(best_w_OR)

        # step 7: AND weights. The 1/distance guess can be so sharp that the true bins get ~zero
        # probability and the loss is flat (e.g. LR1-1500: SLSQP "converges" at the start with 11.8 bit
        # instead of 3.4), so several scaled starts are tried and the best optimum is kept.
        x0 = w0 / w0.max()
        fits = [minimize(
            obj.f_and, x0 * scale, method=method,
            bounds=Bounds(np.zeros(R), np.ones(R) + np.finfo(float).eps),
            constraints=_monotone_constraint(R, R - 1), options=options or {"maxiter": 500},
        ) for scale in (1.0, 0.5, 0.2, 0.1, 0.05)]
        res_and = min((r for r in fits if np.isfinite(r.fun)), key=lambda r: r.fun)
        # SLSQP satisfies w[i+1] <= w[i] only up to its tolerance; a tiny increase would make the
        # interpolated weights of the predictor negative, so enforce it exactly
        best_w_AND = np.minimum.accumulate(res_and.x)
    dkl_or = obj.f_or(best_w_OR)
    w_or_slope = _slopes(best_w_OR, ig.edges_distance_classes_range, cfg.nugget)
    dkl_and = obj.f_and(best_w_AND)
    w_and_slope = _slopes(best_w_AND, ig.edges_distance_classes_range, cfg.nugget)

    # step 8: grid search of alpha / beta for the log-linear combination of the AND and OR PMFs
    alpha, beta, dkl_ab, n_nn, idx_nn = search_alpha_beta(z_cal, ig, dist, cls, P, best_w_OR,
                                                          best_w_AND, alpha_beta)

    return Weights(
        best_w_OR=best_w_OR, best_w_AND=best_w_AND, w_OR_slope=w_or_slope, w_AND_slope=w_and_slope,
        best_alpha=alpha, best_beta=beta, DKL_w_OR=dkl_or, DKL_w_AND=dkl_and,
        DKL_w_alpha_beta=dkl_ab, n_neighbor_aggreg=n_nn, idx_opt_nn=idx_nn,
        n_pairs_by_class_obs=counts, nmean_pairs_by_class=nmean, target_idx_zero_neigh=zero_neigh,
        result_or=res_or, result_and=res_and, nugget=cfg.nugget, objective=objective,
    )
