"""Check pyher against the MATLAB results of a HER run of Thiesen et al. (2020, HESS) stored in a .mat file
(e.g. ``LR1_output_randomfield_her_cal600_alphabeta_loglinear.mat``).

The class weights (``best_w_OR``, ``best_w_AND``) and the exponents (``best_alpha``, ``best_beta``) of the MATLAB
run are fed into ``optimize_weights(fixed_weights=..., alpha_beta=...)``; the mean prediction on the 100 x 100
grid (``z_target_mean_pred``) and the test / validation scores of the MATLAB run are then compared with the ones
reproduced by pyher, for both code versions (nugget / NOnugget).

Usage: python verify_paper_her.py <results.mat> <folder with the HER_datasets .mat files> [--field LR1 --size 600]"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyher import HerConfig, load_her_synthetic
from pyher.core import performance_prob
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.weights import optimize_weights

ap = argparse.ArgumentParser()
ap.add_argument("mat")
ap.add_argument("folder")
ap.add_argument("--field", default="LR1")
ap.add_argument("--size", type=int, default=600)
args = ap.parse_args()

NAMES = ["best_w_OR", "best_w_AND", "best_alpha", "best_beta", "lag_dist", "binwidth", "binwidth_z", "n_neighbor",
         "n_neighbor_andor", "n_classes_range", "edges_z", "bin_centers_edges_z", "z_target_mean_pred", "x_target",
         "y_target", "idx_cal", "idx_test", "idx_val", "z_cal", "z_test", "z_val", "RMSE_test", "RMSE_val",
         "MAE_test", "MAE_val", "NSE_test", "NSE_val", "ME_test", "ME_val", "DKL_score_mean_test", "DKL_score_mean_val",
         "DKL_score_mean_cal", "weights_or", "weights_and"]
m = {k: np.squeeze(v) for k, v in loadmat(args.mat, variable_names=NAMES).items() if not k.startswith("__")}
print("MATLAB run: lag %.3g, binwidth %.3g / %.3g, n_neighbor %d (andor %d), classes in range %d, alpha %.2f, beta %.2f" % (
    m["lag_dist"], m["binwidth"], m["binwidth_z"], m["n_neighbor"], m["n_neighbor_andor"], m["n_classes_range"],
    m["best_alpha"], m["best_beta"]))

d = load_her_synthetic(Path(args.folder) / f"{args.field}_input_randomfield_cal{args.size}.mat")
z = np.round(d.z_cal, 5)
print("same data: z_cal max |diff| %.2e, z_test max |diff| %.2e" % (
    np.abs(m["z_cal"] - d.z_cal).max(), np.abs(m["z_test"] - d.z_test).max()))
ok_ = (np.array_equal(m["idx_test"].astype(int) - 1, d.idx_test) and np.array_equal(m["idx_val"].astype(int) - 1, d.idx_val)
       and np.array_equal(m["idx_cal"].astype(int) - 1, d.idx_cal))
print("same calibration / validation / test split:", ok_)
w_or, w_and = np.ravel(m["best_w_OR"]), np.ravel(m["best_w_AND"])

# MATLAB scores re-derived from the stored mean map: confirms how the stored numbers are defined
it, iv = m["idx_test"].astype(int) - 1, m["idx_val"].astype(int) - 1
zt, zv = np.ravel(d.z)[it], np.ravel(d.z)[iv]


def det(mean, ztrue):
    e = mean - ztrue
    return dict(RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), NSE=1 - np.sum(e**2) / np.sum((ztrue - ztrue.mean())**2),
                ME=np.mean(e))


mat_map = np.ravel(m["z_target_mean_pred"])
# MATLAB's grid order: x_target = x(:) (column-major); pyher's d.x is the same flattening
assert np.allclose(m["x_target"], d.x) and np.allclose(m["y_target"], d.y)
for nm, mean_, idx, key in (("test", mat_map[it], it, "test"), ("val", mat_map[iv], iv, "val")):
    r = det(mean_, np.ravel(d.z)[idx])
    print("MATLAB %-4s stored: RMSE %.4f MAE %.4f NSE %.4f ME %+.4f | recomputed from its mean map: RMSE %.4f MAE %.4f NSE %.4f ME %+.4f" % (
        nm, m[f"RMSE_{key}"], m[f"MAE_{key}"], m[f"NSE_{key}"], m[f"ME_{key}"], r["RMSE"], r["MAE"], r["NSE"], r["ME"]))

print("\npyher with the MATLAB weights and exponents:")
rows = {}
for nugget, factor in ((True, 1.0), (False, 1.0), (True, 3.0), (False, 3.0)):
    cfg = HerConfig(z_edges_factor=factor, lag_dist=float(m["lag_dist"]), binwidth=float(m["binwidth"]), binwidth_z=float(m["binwidth_z"]),
                    n_neighbor=int(m["n_neighbor"]), objective="bin", nugget=nugget)
    ig = fit_infogram(d.x_cal, d.y_cal, z, cfg)
    same_edges = len(ig.edges_z) == len(np.ravel(m["edges_z"])) and np.allclose(ig.edges_z, np.ravel(m["edges_z"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(d.x_cal, d.y_cal, z, ig, fixed_weights=(w_or, w_and),
                             alpha_beta=(float(m["best_alpha"]), float(m["best_beta"])))
        pmf, _ = Predictor(d.x_cal, d.y_cal, z, ig, w).predict(d.x, d.y)
    mean = np.nan_to_num(pmf) @ ig.bin_centers_edges_z
    cal = np.ones(len(d.x), bool)
    cal[d.idx_cal] = False  # at calibration nodes MATLAB/pyher differ only by the nugget treatment
    diff = mean - mat_map
    tag = ("nugget" if nugget else "NOnugget") + f", z-edge factor {factor:g}"
    print(f"  {tag:26s} (z edges as MATLAB: {same_edges}): classes {ig.n_classes_range} | grid mean vs MATLAB (non-calibration nodes): max |diff| {np.abs(diff[cal]).max():.2e}, "
          f"mean |diff| {np.abs(diff[cal]).mean():.2e}, corr {np.corrcoef(mean[cal], mat_map[cal])[0, 1]:.6f}")
    for nm, idx, key in (("test", it, "test"), ("val", iv, "val")):
        r = det(mean[idx], np.ravel(d.z)[idx])
        dkl = performance_prob(np.round(np.ravel(d.z)[idx], 5), pmf[idx], ig.edges_z)
        print("      %-4s RMSE %.4f (MATLAB %.4f)  MAE %.4f (%.4f)  NSE %.4f (%.4f)  DKL %.4f (%.4f)" % (
            nm, r["RMSE"], m[f"RMSE_{key}"], r["MAE"], m[f"MAE_{key}"], r["NSE"], m[f"NSE_{key}"], dkl, m[f"DKL_score_mean_{key}"]))
