"""End-to-end HER run (the flow of ``sHER_E_04_0_Jura_struct.m``): HER1 -> HER2 -> HER3 -> HERs4."""

from dataclasses import dataclass

import numpy as np

from pyher.config import HerConfig
from pyher.infogram import Infogram, fit_infogram
from pyher.predict import Predictor, simulate
from pyher.stats import ValidationMetrics, pmf_from_realizations, validate
from pyher.weights import Weights, optimize_weights


@dataclass
class HerResult:
    infogram: Infogram
    weights: Weights
    grid_x: np.ndarray  # [ny, nx] meshgrid of the target locations
    grid_y: np.ndarray
    pmf_pred: np.ndarray  # [ny, nx, n_bins_z]
    idx_zero_neigh: np.ndarray  # flat (column-major) indices of targets with no neighbour in range
    z_realiz: np.ndarray | None  # [ny, nx, nsim]
    validation: ValidationMetrics | None = None  # HER prediction at the validation points
    validation_sim: ValidationMetrics | None = None  # HERs at the validation points


def run_her(x_cal, y_cal, z_cal, cfg: HerConfig, *, grid=(0.0, 6.0, 0.05), nsim: int = 0,
            seed: int | None = None, validation=None, nsim_val: int | None = None,
            fixed_weights=None, alpha_beta=None, or_shrinkage="auto") -> HerResult:
    """Run HER1-HER3 (and HERs4 if ``nsim > 0``).

    ``fixed_weights=(w_or, w_and)`` and ``alpha_beta=(alpha, beta)`` replace the optimisation of HER2
    (see ``pyher.reference`` for the paper's Jura weights). ``or_shrinkage`` is the strength of the
    shrinkage of the OR weights towards 1/distance (``"auto"``: on for the threshold objective, off for 'bin';
    ``None`` switches it off, see ``optimize_weights``).

    ``validation = (x_val, y_val, z_val)`` additionally evaluates the predicted PMFs at the validation
    points and the PMFs built from ``nsim_val`` (default ``nsim``) simulations that use the validation
    locations as their targets (the MATLAB script reads the grid at ``idx_val`` instead, which only
    works when the validation points are grid nodes).
    """
    if validation is not None and cfg.z_thresh is None:
        raise ValueError("validation needs a threshold: set HerConfig(z_thresh=...)")
    z_cal = np.round(np.asarray(z_cal, dtype=float), cfg.round_decimals)
    ig = fit_infogram(x_cal, y_cal, z_cal, cfg)
    w = optimize_weights(x_cal, y_cal, z_cal, ig, fixed_weights=fixed_weights, alpha_beta=alpha_beta,
                         or_shrinkage=or_shrinkage)

    lo, hi, step = grid
    axis = np.round(np.arange(0, int(round((hi - lo) / step)) + 1) * step + lo, 10)  # no duplicate nodes for fine steps
    gx, gy = np.meshgrid(axis, axis)
    xt, yt = gx.ravel(order="F"), gy.ravel(order="F")  # MATLAB x(:) is column-major
    pmf, zero = Predictor(x_cal, y_cal, z_cal, ig, w).predict(xt, yt)
    pmf = pmf.reshape(gx.shape[1], gx.shape[0], -1).transpose(1, 0, 2)  # column-major -> [ny, nx]

    z_real = None
    if nsim:
        rng = np.random.default_rng(seed)
        sims = [simulate((x_cal, y_cal, z_cal, ig, w), xt, yt, rng=rng)[0] for _ in range(nsim)]
        z_real = np.stack([s.reshape(gx.shape[1], gx.shape[0]).T for s in sims], axis=2)
    val = val_sim = None
    if validation is not None:
        xv, yv, zv = (np.asarray(a, dtype=float) for a in validation)
        zv = np.round(zv, cfg.round_decimals)
        pmf_v, _ = Predictor(x_cal, y_cal, z_cal, ig, w).predict(xv, yv)
        val = validate(pmf_v, zv, z_cal, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
        nsim_val = nsim if nsim_val is None else nsim_val
        if nsim_val:
            rng = np.random.default_rng(None if seed is None else seed + 1)
            sims = np.column_stack([
                simulate((x_cal, y_cal, z_cal, ig, w), xv, yv, rng=rng)[0] for _ in range(nsim_val)])
            val_sim = validate(pmf_from_realizations(sims, ig.edges_z), zv, z_cal, ig.edges_z,
                               ig.bin_centers_edges_z, cfg.z_thresh)
    return HerResult(ig, w, gx, gy, pmf, zero, z_real, val, val_sim)
