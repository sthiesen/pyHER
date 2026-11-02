"""HER with nugget vs. HER without nugget (``HerConfig(nugget=...)``) on a synthetic field of Thiesen et al. (2020, HESS),
both with the class weights and (alpha, beta) of the paper's MATLAB run. Every node of the 100 x 100 grid is predicted,
including the calibration nodes, to check whether either version is an exact interpolator: difference maps
(prediction - truth), the error at the calibration nodes, the predicted spread there and the probability the PMF
puts on the bin of the true value. Test / validation scores of both versions are printed as well.

Usage: python her_nugget_vs_nonugget.py <folder with .mat files> --paper-mat <HER results .mat> [--field LR1 --size 600]
Output: examples/output/synthetic_compare/<field>_<size>_her_nugget_vs_nonugget.png"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from pyher import HerConfig, load_her_synthetic
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.weights import optimize_weights

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--paper-mat", required=True)
ap.add_argument("--field", default="LR1")
ap.add_argument("--size", type=int, default=600)
ap.add_argument("-k", type=int, default=12)
args = ap.parse_args()

out = Path(__file__).resolve().parent / "output" / "synthetic_compare"
d = load_her_synthetic(Path(args.folder) / f"{args.field}_input_randomfield_cal{args.size}.mat")
zc = np.round(d.z_cal, 5)
n = len(d.x)
assert d.y[1] - d.y[0] == 1 and d.x[1] == d.x[0], "grid is expected to be stored with y varying fastest (x(:) of a meshgrid)"
is_cal = np.zeros(n, bool)
is_cal[d.idx_cal] = True
pm = {k: np.squeeze(v) for k, v in loadmat(args.paper_mat, variable_names=["best_w_OR", "best_w_AND", "best_alpha", "best_beta"]).items()
      if not k.startswith("__")}
fixed = dict(fixed_weights=(np.ravel(pm["best_w_OR"]), np.ravel(pm["best_w_AND"])),
             alpha_beta=(float(pm["best_alpha"]), float(pm["best_beta"])))
ZG = np.linspace(-8, 6, 561)


def scores(pmf, edges, cen, z):
    m = pmf @ cen
    e = m - z
    cdf = np.array([np.interp(ZG, edges, r) for r in np.c_[np.zeros(len(pmf)), np.cumsum(pmf, 1)]])
    q = lambda a: np.array([np.interp(a, r, ZG) for r in cdf])
    r = dict(RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), NSE=1 - np.sum(e**2) / np.sum((z - z.mean())**2),
             CRPS=np.mean(np.sum((cdf - (ZG[None] >= z[:, None])) ** 2, 1) * (ZG[1] - ZG[0])))
    for lev in (0.5, 0.9):
        lo, hi = q((1 - lev) / 2), q((1 + lev) / 2)
        r[f"cov{int(lev * 100)}"] = np.mean((z >= lo) & (z <= hi))
    return r


res = {}
for name, nugget in (("with nugget", True), ("without nugget", False)):
    cfg = HerConfig(lag_dist=2.0, binwidth=0.2, binwidth_z=0.2, n_neighbor=args.k, objective="bin", nugget=nugget,
                    z_edges_factor=1.0)
    ig = fit_infogram(d.x_cal, d.y_cal, zc, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(d.x_cal, d.y_cal, zc, ig, **fixed)
        pmf, zero = Predictor(d.x_cal, d.y_cal, zc, ig, w).predict(d.x, d.y)
    bad = ~np.isfinite(pmf).all(1)
    pmf = np.nan_to_num(pmf)
    pmf /= np.maximum(pmf.sum(1, keepdims=True), 1e-12)
    c = ig.bin_centers_edges_z
    mean = pmf @ c
    std = np.sqrt(np.maximum(pmf @ c**2 - mean**2, 0))
    bin_true = np.clip(np.searchsorted(ig.edges_z, np.round(d.z, 5), side="right") - 1, 0, pmf.shape[1] - 1)
    res[name] = dict(pmf=pmf, mean=mean, std=std, p_true_bin=pmf[np.arange(n), bin_true], edges=ig.edges_z, cen=c, bad=bad.sum())
    print(f"{name}: classes {ig.n_classes_range}, alpha {w.best_alpha:.2f}, beta {w.best_beta:.2f}, rows with NaN: {bad.sum()}", flush=True)

print(f"\n{args.field}-{args.size}: error (prediction - truth), predicted spread and probability of the true bin")
print("%-16s | %-31s | %-18s | %s" % ("", "calibration nodes (600)", "other nodes", "calibration nodes"))
print("%-16s | %9s %9s %9s | %9s %8s | %10s %10s" % ("", "mean|err|", "max|err|", "RMSE", "mean|err|", "RMSE", "mean sd", "P(true bin)"))
for name, r in res.items():
    e = r["mean"] - d.z
    print("%-16s | %9.3f %9.3f %9.3f | %9.3f %8.3f | %10.3f %10.3f" % (
        name, np.abs(e[is_cal]).mean(), np.abs(e[is_cal]).max(), np.sqrt(np.mean(e[is_cal]**2)), np.abs(e[~is_cal]).mean(),
        np.sqrt(np.mean(e[~is_cal]**2)), r["std"][is_cal].mean(), r["p_true_bin"][is_cal].mean()))
print("(bin width 0.2: a PMF concentrated on the right bin has |error| up to 0.1 even if the bin is right)")
for sname, idx in (("test", d.idx_test), ("validation", d.idx_val)):
    print(f"\n{sname} nodes ({len(idx)})")
    print("%-16s %6s %6s %6s %6s %6s %6s %9s" % ("", "RMSE", "MAE", "NSE", "CRPS", "cov50", "cov90", "mean sd"))
    for name, r in res.items():
        s = scores(r["pmf"][idx], r["edges"], r["cen"], d.z[idx])
        print("%-16s %6.3f %6.3f %6.3f %6.3f %6.2f %6.2f %9.3f" % (name, s["RMSE"], s["MAE"], s["NSE"], s["CRPS"], s["cov50"], s["cov90"], r["std"][idx].mean()))
np.savez(out / f"{args.field}_{args.size}_her_nugget_vs_nonugget.npz", is_cal=is_cal, truth=d.z,
         **{f"{k}__{kk}": v for k, r in res.items() for kk, v in r.items() if kk in ("mean", "std", "p_true_bin")})

# ---- figure
G = lambda v: np.asarray(v).reshape(100, 100, order="F")  # [y, x]
dif = {k: r["mean"] - d.z for k, r in res.items()}
lim = np.percentile(np.abs(np.concatenate([e[~is_cal] for e in dif.values()])), 99)
slim = np.percentile(np.concatenate([r["std"] for r in res.values()]), 99)
cx, cy = d.x_cal - 1, d.y_cal - 1
best, best_n = (0, 0), -1
for x0 in range(0, 76):
    for y0 in range(0, 76):
        cnt = np.sum((cx >= x0) & (cx < x0 + 25) & (cy >= y0) & (cy < y0 + 25))
        if cnt > best_n:
            best, best_n = (x0, y0), cnt
x0, y0 = best
vm = (np.percentile(d.z, 1), np.percentile(d.z, 99))
panels = [("true field", d.z, "viridis", vm),
          ("with nugget: prediction - truth", dif["with nugget"], "RdBu_r", (-lim, lim)),
          ("without nugget: prediction - truth", dif["without nugget"], "RdBu_r", (-lim, lim)),
          ("without - with nugget (mean)", res["without nugget"]["mean"] - res["with nugget"]["mean"], "RdBu_r", (-lim, lim)),
          ("with nugget: predictive sd", res["with nugget"]["std"], "magma", (0, slim)),
          ("without nugget: predictive sd", res["without nugget"]["std"], "magma", (0, slim))]
fig, axs = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 8.2), constrained_layout=True)
ims = []
for j, (title, v, cm, (lo, hi)) in enumerate(panels):
    for r_ in (0, 1):
        ax = axs[r_, j]
        im = ax.imshow(G(v), origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        if r_ == 0:
            ax.set_title(title, fontsize=8.5)
            ax.add_patch(plt.Rectangle((x0 - .5, y0 - .5), 25, 25, fill=False, ec="k", lw=1))
        else:
            ax.set_xlim(x0 - .5, x0 + 24.5); ax.set_ylim(y0 - .5, y0 + 24.5)
            ax.scatter(cx, cy, s=14, marker="s", facecolors="none", edgecolors="k", linewidths=.6)
            if j == 0:
                ax.set_title("zoom (squares = calibration nodes)", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)
fig.colorbar(ims[0], ax=axs[:, 0], shrink=.5, location="left", pad=0.02, label="z")
fig.colorbar(ims[1], ax=axs[:, 1:4], shrink=.5, location="right", pad=0.01, label="difference")
fig.colorbar(ims[4], ax=axs[:, 4:], shrink=.5, location="right", pad=0.01, label="predictive sd")
fig.savefig(out / f"{args.field}_{args.size}_her_nugget_vs_nonugget.png", dpi=90)
print("figure:", out / f"{args.field}_{args.size}_her_nugget_vs_nonugget.png")
