"""HER1: spatial characterisation (port of ``f_her1_infogram.m``)."""

import copy
from dataclasses import dataclass

import numpy as np

from pyher.config import HerConfig
from pyher.core import entropy, matlab_colon, matlab_linspace


@dataclass
class Infogram:
    """Result of HER1. Indices are 0-based; field names follow the MATLAB ``her`` struct."""

    cfg: HerConfig
    # delta-z resolution
    n_bins_z_per_bin_shift: int
    binwidth_shift: float
    # all pairs
    mat_euc_distance_xy: np.ndarray  # [N, N]
    mat_diff_z: np.ndarray  # [N, N], z_i - z_j
    obs_diff_z: np.ndarray  # delta-z of all pairs with distance > 0
    # distance classes (lags) over the full distance range
    n_lag: int
    edges_distance_classes: np.ndarray
    bin_centers_distance_classes: np.ndarray
    emptyclass: np.ndarray  # 0-based indices of empty classes
    n_pairs_by_class: np.ndarray
    # delta-z bins
    edges_diff_z: np.ndarray
    edges_diff_z_shift: np.ndarray
    n_bins_shift: int
    prob_min: float
    # entropies
    H_diff_z: float
    H_diff_z_by_class: np.ndarray
    # information range: classes before the entropy exceeds H_lim, plus one class for the full data
    H_lim: float
    n_classes_range: int  # number of classes inside the range + 1 (full-dataset class)
    edges_distance_classes_range: np.ndarray  # [n_classes_range + 1]
    bin_centers_distance_classes_range: np.ndarray
    pmf_diff_z_by_class_obs_range: np.ndarray  # [n_classes_range, n_bins]
    pmf_diff_z_by_class_obs_range_shift: np.ndarray  # [n_classes_range, n_bins_shift + 1]
    H_diff_z_by_class_range: np.ndarray
    # z bins
    n_bins_z: int
    edges_z: np.ndarray
    bin_centers_edges_z: np.ndarray
    bin_centers_edges_diff_z: np.ndarray
    n_classes_limit: int | None = None


def _symmetric_edges(max_diff_z: float, binwidth: float) -> tuple[int, np.ndarray]:
    n_bins = 2 * int(np.floor(max_diff_z / binwidth + 1))
    half = n_bins / 2 * binwidth
    return n_bins, matlab_linspace(-half, half, n_bins + 1)


def _histcounts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # same bin convention as MATLAB histcounts: [e_i, e_i+1), last bin closed
    return np.histogram(x, bins=edges)[0].astype(float)


def fit_infogram(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: HerConfig) -> Infogram:
    x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
    if not (x.size == y.size == z.size) or x.size < 3:
        raise ValueError("x, y, z must have the same length (at least 3 points)")
    cfg = copy.copy(cfg)  # the infogram keeps its own copy: later changes to the caller's cfg do not leak in

    # delta-z resolution
    zm_res = np.diff(np.unique(z)).min()
    n_per = max(1, min(100, int(np.floor(cfg.binwidth_z / zm_res + 0.5))))  # MATLAB round(): half away from 0
    binwidth_shift = cfg.binwidth_z / n_per

    # step 1: distance and delta-z of all point pairs
    dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    diff_z = z[:, None] - z[None, :]

    # step 2: distance classes, (e_i, e_i+1]
    n_lag = int(np.ceil(dist.max() / cfg.lag_dist))
    edges_d = matlab_colon(0.0, cfg.lag_dist, cfg.lag_dist * n_lag)
    centers_d = (edges_d[:-1] + edges_d[1:]) / 2
    cls = np.searchsorted(edges_d, dist, side="left") - 1  # -1 for dist == 0
    in_cls = (cls >= 0) & (cls < n_lag)
    n_pairs_by_class = np.bincount(cls[in_cls], minlength=n_lag)
    emptyclass = np.flatnonzero(n_pairs_by_class == 0)
    if emptyclass.size:
        print(f"Attention! There are empty distance classes ({emptyclass + 1}). Consider "
              "increasing the lag or bin width to fill the classes into the range.")

    # step 3: delta-z bin edges (symmetric)
    max_diff_z = diff_z.max()
    n_bins, edges_dz = _symmetric_edges(max_diff_z, cfg.binwidth)
    n_bins_shift, edges_dz_shift = _symmetric_edges(max_diff_z, binwidth_shift)

    # step 4: delta-z PMF of the full dataset (+1 count per bin to avoid empty bins)
    obs = diff_z[dist > 0]
    prob_min = 1.0 / obs.size
    counts_all = _histcounts(obs, edges_dz) + 1
    pmf_all = counts_all / counts_all.sum()
    counts_shift = _histcounts(obs, edges_dz_shift) + 1.0 / n_per
    counts_shift = np.concatenate(([1.0 / n_per], counts_shift))  # empty bin at the beginning
    pmf_all_shift = counts_shift / counts_shift.sum()
    prob_min_shift = prob_min / n_per
    for p in (pmf_all, pmf_all_shift):
        if abs(p.sum() - 1) > 1e-5:
            raise ValueError("Probabilities do not sum to 1.")

    # step 5: delta-z PMF by distance class
    pmf_cls = np.full((n_lag, n_bins), np.nan)
    pmf_cls_shift = np.full((n_lag, n_bins_shift + 1), np.nan)
    for i in range(n_lag):
        o = diff_z[(dist > edges_d[i]) & (dist <= edges_d[i + 1])]
        with np.errstate(invalid="ignore", divide="ignore"):  # empty class -> NaN, as in MATLAB
            p = _histcounts(o, edges_dz) / o.size + prob_min
            pmf_cls[i] = p / p.sum()
            ps = _histcounts(o, edges_dz_shift) / o.size + prob_min_shift
            pmf_cls_shift[i, 1:] = ps / ps.sum()
    pmf_cls_shift[:, 0] = prob_min_shift

    # step 6: entropies
    H_all = entropy(pmf_all)
    H_cls = np.array([entropy(p) for p in pmf_cls])

    # step 7: information range = classes before the entropy exceeds that of the full dataset
    valid = ~np.isnan(H_cls)
    n_classes_limit = None
    if np.all(H_all > H_cls[valid]):
        n_classes_limit = min(20, n_lag)  # 20 is hard-coded in the MATLAB code ("look at the infogram")
        H_lim = float(H_cls[n_classes_limit - 1])
    else:
        H_lim = H_all
    exceeds = np.flatnonzero(H_cls > H_lim)
    if exceeds.size == 0:
        raise ValueError("Entropy never exceeds the limit: no information range found.")
    n_keep = int(exceeds[0])  # classes kept before the break
    n_classes_range = n_keep + 1

    edges_range = np.concatenate(([0.0], edges_d[1:n_keep + 1], [edges_d[-1]]))
    pmf_range = np.vstack([pmf_cls[:n_keep], pmf_all])
    pmf_range_shift = np.vstack([pmf_cls_shift[:n_keep], pmf_all_shift])
    H_range = np.concatenate((H_cls[:n_keep], [H_all]))
    centers_range = (edges_range[:-1] + edges_range[1:]) / 2
    if np.isnan(pmf_range).any():
        raise ValueError("Empty distance classes inside the information range (classes "
                         f"{(np.flatnonzero(np.isnan(pmf_range).any(axis=1)) + 1).tolist()}): the delta-z PMFs "
                         "cannot be interpolated. Increase lag_dist (or binwidth_z).")

    # edges of the z PMF
    min_dz, max_dz = edges_dz_shift[0], edges_dz_shift[-1]
    n_left = -int(np.floor((cfg.z_edges_factor * min_dz + z.min()) / cfg.binwidth_z))
    n_right = int(np.floor((cfg.z_edges_factor * max_dz + z.max()) / cfg.binwidth_z + 1))
    n_bins_z = n_left + n_right
    edges_z = matlab_linspace(-cfg.binwidth_z * n_left, cfg.binwidth_z * n_right, n_bins_z + 1)

    return Infogram(
        cfg=cfg, n_bins_z_per_bin_shift=n_per, binwidth_shift=binwidth_shift,
        mat_euc_distance_xy=dist, mat_diff_z=diff_z, obs_diff_z=obs,
        n_lag=n_lag, edges_distance_classes=edges_d, bin_centers_distance_classes=centers_d,
        emptyclass=emptyclass, n_pairs_by_class=n_pairs_by_class,
        edges_diff_z=edges_dz, edges_diff_z_shift=edges_dz_shift, n_bins_shift=n_bins_shift,
        prob_min=prob_min, H_diff_z=H_all, H_diff_z_by_class=H_cls, H_lim=H_lim,
        n_classes_range=n_classes_range, edges_distance_classes_range=edges_range,
        bin_centers_distance_classes_range=centers_range,
        pmf_diff_z_by_class_obs_range=pmf_range,
        pmf_diff_z_by_class_obs_range_shift=pmf_range_shift,
        H_diff_z_by_class_range=H_range, n_bins_z=n_bins_z, edges_z=edges_z,
        bin_centers_edges_z=edges_z[:-1] + cfg.binwidth_z / 2,
        bin_centers_edges_diff_z=edges_dz[:-1] + cfg.binwidth / 2,
        n_classes_limit=n_classes_limit,
    )


def infogram_entropy(x, y, z, lag_dist: float, binwidth: float, *, max_points: int | None = None,
                     seed: int = 0, n_classes: int | None = None):
    """Entropy of the delta-z PMF by distance class (steps 1-6 of ``f_her1_infogram``; this is what
    ``f_infogram_realizplot.m`` computes, without the range and z-edge steps).

    Used to compare the infogram of simulated fields, an E-type map and the data (ergodic fluctuations).
    ``max_points`` subsamples random points first, because the pair matrices grow with N**2.
    Returns ``(bin_centers, H_by_class, H_all)``.
    """
    x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
    if max_points is not None and len(x) > max_points:
        keep = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
        x, y, z = x[keep], y[keep], z[keep]
    dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    diff_z = z[:, None] - z[None, :]
    n_lag = int(np.ceil(dist.max() / lag_dist))
    edges_d = matlab_colon(0.0, lag_dist, lag_dist * n_lag)
    n_bins, edges_dz = _symmetric_edges(diff_z.max(), binwidth)
    obs = diff_z[dist > 0]
    prob_min = 1.0 / obs.size
    counts_all = _histcounts(obs, edges_dz) + 1
    H_all = entropy(counts_all / counts_all.sum())
    H = np.full(n_lag, np.nan)
    for i in range(n_lag):
        o = diff_z[(dist > edges_d[i]) & (dist <= edges_d[i + 1])]
        if o.size == 0:
            continue
        p = _histcounts(o, edges_dz) / o.size + prob_min
        H[i] = entropy(p / p.sum())
    centers = (edges_d[:-1] + edges_d[1:]) / 2
    if n_classes is not None:
        H, centers = H[:n_classes], centers[:n_classes]
        if len(H) < n_classes:
            H = np.pad(H, (0, n_classes - len(H)), constant_values=np.nan)
            centers = np.pad(centers, (0, n_classes - len(centers)), constant_values=np.nan)
    return centers, H, H_all
