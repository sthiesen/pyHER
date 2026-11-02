"""PMF statistics and performance metrics.

Ports of ``f_extract_pmf_statistics.m``, ``f_performance_det.m``, ``f_ROC.m``, ``f_G.m``, ``f_perf.m``
and the performance block of ``sHER_E_0304_Extract_pmf_statistics_performance_and_plot.m``.
"""

from dataclasses import dataclass, field

import numpy as np

from pyher.core import entropy, matlab_colon, performance_prob
from pyher.pmfs import matlab_round


def _round(x, decimals):
    f = 10.0 ** decimals
    return matlab_round(np.asarray(x, dtype=float) * f) / f


# ----------------------------------------------------------------------------------------------
# PMF statistics
# ----------------------------------------------------------------------------------------------
@dataclass
class PmfStatistics:
    entropy: np.ndarray  # [T] bit
    mean: np.ndarray  # [T] bin-centre weighted mean
    mode: np.ndarray  # [T] centre of the most probable bin
    prob_exceed: np.ndarray  # [T] P(z > z_thresh); NaN if no threshold is given


def extract_pmf_statistics(pmfs, edges_z, bin_centers_z, z_thresh=None) -> PmfStatistics:
    """Entropy, mean, mode and exceedance probability of PMFs (``f_extract_pmf_statistics.m``).

    As in MATLAB, the exceedance probability counts the whole bin containing ``z_thresh`` as being
    below the threshold (``1 - cumsum(pmf)[bin of z_thresh]``), without interpolation.
    """
    pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
    edges_z = np.asarray(edges_z, dtype=float)
    centers = np.asarray(bin_centers_z, dtype=float)
    H = np.array([entropy(p) for p in pmfs])
    mean = (pmfs * centers).sum(axis=1) / pmfs.sum(axis=1)
    mode = centers[np.argmax(pmfs, axis=1)]
    prob = np.full(len(pmfs), np.nan)
    if z_thresh is not None and not np.isnan(z_thresh):
        bin_thres = int((edges_z <= z_thresh).sum())  # MATLAB 1-based bin number
        if bin_thres >= 1:  # a threshold above the last edge: everything is below it
            prob = 1 - np.cumsum(pmfs, axis=1)[:, min(bin_thres, pmfs.shape[1]) - 1]
        else:
            prob = np.ones(len(pmfs))
    return PmfStatistics(H, mean, mode, prob)


def pmf_from_realizations(z_realiz, edges_z) -> np.ndarray:
    """Empirical PMFs on ``edges_z`` from simulated values ``[T, n_realizations]`` (the
    ``histcounts(..., 'probability')`` + renormalisation step of the MATLAB script)."""
    z_realiz = np.atleast_2d(np.asarray(z_realiz, dtype=float))
    out = np.empty((len(z_realiz), len(edges_z) - 1))
    for i, row in enumerate(z_realiz):
        c = np.histogram(row, bins=edges_z)[0].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = c / c.sum()
    return out


# ----------------------------------------------------------------------------------------------
# deterministic metrics
# ----------------------------------------------------------------------------------------------
@dataclass
class DeterministicMetrics:
    error_sign: np.ndarray
    RMSE: float
    ME: float
    MAE: float
    NSE: float
    correl_true_pred: float
    correl_true_residue: float


def performance_det(z_simulated, z_true) -> DeterministicMetrics:
    """``f_performance_det.m``: RMSE, ME, MAE, Nash-Sutcliffe efficiency and Pearson correlations."""
    sim = np.asarray(z_simulated, dtype=float).ravel()
    true = np.asarray(z_true, dtype=float).ravel()
    err = sim - true
    n = len(err)
    return DeterministicMetrics(
        error_sign=err,
        RMSE=float(np.sqrt((err ** 2).sum() / n)),
        ME=float(err.sum() / n),
        MAE=float(np.abs(err).sum() / n),
        NSE=float(1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()),
        correl_true_pred=float(np.corrcoef(true, sim)[0, 1]),
        correl_true_residue=float(np.corrcoef(true, err)[0, 1]),
    )


# ----------------------------------------------------------------------------------------------
# classification metrics
# ----------------------------------------------------------------------------------------------
def roc(true_z, predicted) -> dict:
    """Confusion-matrix based scores for 0/1 vectors (``f_ROC.m``). 0/0 gives NaN as in MATLAB."""
    t = np.asarray(true_z, dtype=float)
    p = np.asarray(predicted, dtype=float)
    TP, TN = (t * p).sum(), ((1 - t) * (1 - p)).sum()
    FP, FN = ((1 - t) * p).sum(), (t * (1 - p)).sum()
    P, N = TP + FN, TN + FP
    with np.errstate(divide="ignore", invalid="ignore"):
        TPR, TNR, FPR = TP / P, TN / N, FP / (FP + TN)
        return dict(
            TP=TP, TN=TN, FP=FP, FN=FN, TPR=TPR, TNR=TNR, FPR=FPR,
            Accuracy=(TP + TN) / (P + N),
            F1_score=2 * TP / (2 * TP + FP + FN),
            MCC=(TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
            DistanceROC=np.sqrt((1 - TPR) ** 2 + (0 - FPR) ** 2),
        )


def goodness(pmfs, edges_z, z_true):
    """Goodness statistic G and the fraction of true values inside symmetric probability intervals
    (Deutsch 1997; Goovaerts 2001) -- ``f_G.m``.

    Returns ``G, ksi_fraction, edges_interv, interv_size``.
    """
    pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
    z_true = np.asarray(z_true, dtype=float).ravel()
    edges_z = np.asarray(edges_z, dtype=float)
    p = matlab_colon(0.0, 0.01, 0.5)
    edges_interv = np.column_stack((0 + p, 1 - p))
    interv_size = edges_interv[:, 1] - edges_interv[:, 0]
    cmf = _round(np.concatenate((np.zeros((len(pmfs), 1)), np.cumsum(pmfs, axis=1)), axis=1), 2)
    ksi_fraction = np.empty(len(p))
    for i, (lo, hi) in enumerate(edges_interv):
        idx1 = np.argmax(cmf > lo, axis=1)  # first index with cmf > lo (MATLAB find(...,1))
        idx2 = np.argmax(cmf >= hi, axis=1)
        if not (cmf > lo).any(axis=1).all() or not (cmf >= hi).any(axis=1).all():
            raise ValueError("Interval edge not found in the cumulative PMF.")
        z1 = edges_z[idx1 - 1]  # left edge
        z2 = edges_z[idx2]  # right edge
        ksi_fraction[i] = np.mean((z_true > z1) & (z_true <= z2))
    w = np.where(ksi_fraction >= interv_size, 1.0, -2.0)
    G = 1 - np.sum(w * (ksi_fraction - interv_size)) / len(w)
    return float(G), ksi_fraction, edges_interv, interv_size


# ----------------------------------------------------------------------------------------------
# full validation block
# ----------------------------------------------------------------------------------------------
@dataclass
class ValidationMetrics:
    marginal_prob_thres_cal: float
    deterministic: DeterministicMetrics
    DKL_score_mean: float  # DKL of the true value's bin (HER03 objective)
    DKL_score_01_largersmaller_thres: np.ndarray
    DKL_score_01_largersmaller_thres_mean: float
    DKL_score_1_larger_thres: np.ndarray
    DKL_score_1_larger_thres_mean: float
    prob_thres: np.ndarray
    index_marg: int  # 0-based index into prob_thres (-1 if marginal probability not on the grid)
    roc: dict = field(default_factory=dict)  # arrays over prob_thres, keys as in ``roc()``
    misclassication: np.ndarray = None
    contamination: np.ndarray = None
    G: float = np.nan
    ksi_fraction: np.ndarray = None
    edges_interv: np.ndarray = None
    interv_size: np.ndarray = None
    ksi_fraction_uq_lq: float = np.nan
    misclass_mean: float = np.nan
    misclass_min: float = np.nan
    misclass_marginal: float = np.nan
    Accuracy_marginal: float = np.nan
    F1_score_marginal: float = np.nan
    MCC_marginal: float = np.nan
    DistanceROC_marginal: float = np.nan


def validate(pmfs, z_val, z_cal, edges_z, bin_centers_z, z_thresh) -> ValidationMetrics:
    """Performance of predicted PMFs at validation points (the metrics block of the MATLAB script).

    ``pmfs`` are the predicted (or empirical, for simulations) PMFs *at the validation locations*.
    The MATLAB script instead reads grid cells with ``idx_val``, which only works if the grid is the
    original sample grid; here the PMFs are expected at the validation locations directly.
    """
    if z_thresh is None or np.isnan(z_thresh):
        raise ValueError("validate needs a threshold: set HerConfig(z_thresh=...)")
    pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
    z_val = np.asarray(z_val, dtype=float).ravel()
    z_cal = np.asarray(z_cal, dtype=float).ravel()
    if len(pmfs) != len(z_val) or len(z_val) == 0:
        raise ValueError("pmfs and z_val must have one row / value per validation point (at least one)")
    n = len(z_val)
    st = extract_pmf_statistics(pmfs, edges_z, bin_centers_z, z_thresh)
    z_pred = _round(st.mean, 5)
    p_pred = _round(st.prob_exceed, 5)

    marginal = float(np.sum(z_cal > z_thresh) / len(z_cal))
    det = performance_det(z_pred, z_val)
    dkl_mean = performance_prob(z_val, pmfs, edges_z)

    larger = (z_val > z_thresh).astype(float)
    smaller = (z_val <= z_thresh).astype(float)
    with np.errstate(divide="ignore"):
        s01 = np.where(smaller == 1, -np.log2(1 - p_pred), -np.log2(p_pred))
        s1 = np.where(larger == 0, np.nan, -np.log2(p_pred))

    prob_thres = _round(np.arange(101) * 0.01, 2)
    hit = np.flatnonzero(prob_thres == _round(marginal, 2))
    idx_marg = int(hit[0]) if hit.size else -1
    keys = ["TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "Accuracy", "F1_score", "MCC", "DistanceROC"]
    cols = {k: np.empty(len(prob_thres)) for k in keys}
    contamination = np.empty(len(prob_thres))
    for i, thr in enumerate(prob_thres):
        flagged = (p_pred > thr).astype(float)
        r = roc(larger, flagged)
        for k in keys:
            cols[k][i] = r[k]
        contamination[i] = 100 * flagged.sum() / n
    misclass = 100 * (1 - cols["Accuracy"])

    G, ksi, e_int, size_int = goodness(pmfs, edges_z, z_val)
    uq = np.flatnonzero(np.isclose(e_int[:, 0], 0.25))
    def at(a):
        return a[idx_marg] if idx_marg >= 0 else np.nan

    return ValidationMetrics(
        marginal_prob_thres_cal=marginal, deterministic=det, DKL_score_mean=dkl_mean,
        DKL_score_01_largersmaller_thres=s01, DKL_score_01_largersmaller_thres_mean=float(s01.mean()),
        DKL_score_1_larger_thres=s1, DKL_score_1_larger_thres_mean=float(np.nanmean(s1)),
        prob_thres=prob_thres, index_marg=idx_marg, roc=cols, misclassication=misclass,
        contamination=contamination, G=G, ksi_fraction=ksi, edges_interv=e_int,
        interv_size=size_int, ksi_fraction_uq_lq=float(ksi[uq[0]]) if uq.size else np.nan,
        misclass_mean=100 * float(np.sum((z_val > z_thresh) != (z_pred > z_thresh))) / n,
        misclass_min=float(np.min(misclass)), misclass_marginal=at(misclass),
        Accuracy_marginal=100 * at(cols["Accuracy"]), F1_score_marginal=at(cols["F1_score"]),
        MCC_marginal=at(cols["MCC"]), DistanceROC_marginal=at(cols["DistanceROC"]),
    )
