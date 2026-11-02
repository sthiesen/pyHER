"""HERs (sequential simulation) on a synthetic field of Thiesen et al. (2020, HESS): do the realisations
reproduce the properties of the field? Checks (ergodic fluctuations, as in the commented block of
sHER_E_0304_...m):

 1. data are honoured at the calibration nodes;
 2. marginal distribution: pooled realisations vs. the true field (KS distance, mean, sd);
 3. spatial structure: infogram (entropy of the delta-z PMF by distance class) of every realisation vs. the
    true field, the calibration set and the E-type map;
 4. ensemble vs. analytical HER prediction: mean and spread at the test nodes, and the coverage of the
    true test values by the central 50 % / 90 % intervals of the ensemble.

Usage: python hers_hess2020.py <folder with .mat files> [--field LR1] [--size 600] [--nreal 50] [--nugget]
Output: examples/output/hess2020/hers_<field>_<size>.{png,npz}
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pyher import HerConfig, load_her_synthetic
from pyher.infogram import fit_infogram
from pyher.predict import Predictor, simulate
from pyher.stats import extract_pmf_statistics
from pyher.weights import optimize_weights

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--field", default="LR1")
ap.add_argument("--size", type=int, default=600)
ap.add_argument("--nreal", type=int, default=50)
ap.add_argument("--nugget", action="store_true", help="HESS-2020 code version (default: NOnugget, as used for HERs)")
ap.add_argument("--reuse", action="store_true", help="reuse saved realisations of a previous run")
ap.add_argument("--nsub", type=int, default=1500, help="nodes used for the infograms of the fields")
args = ap.parse_args()

out = Path(__file__).resolve().parent / "output" / "hess2020"
out.mkdir(parents=True, exist_ok=True)
d = load_her_synthetic(Path(args.folder) / f"{args.field}_input_randomfield_cal{args.size}.mat")
z = np.round(d.z_cal, 5)
cfg = HerConfig(lag_dist=2.0, binwidth=0.2, binwidth_z=0.2, n_neighbor=12, objective="bin", nugget=args.nugget)
ig = fit_infogram(d.x_cal, d.y_cal, z, cfg)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w = optimize_weights(d.x_cal, d.y_cal, z, ig)
print(f"{args.field}-{args.size}: range {2 * (ig.n_classes_range - 1)}, alpha {w.best_alpha:.2f}, beta {w.best_beta:.2f}")

# ---- the ensemble on the whole 100 x 100 grid
saved = out / f"hers_{args.field}_{args.size}.npz"
if args.reuse and saved.exists():
    sims = np.load(saved)["sims"].astype(float)
    args.nreal = len(sims)
else:
    rng = np.random.default_rng(1)
    sims = np.empty((args.nreal, len(d.x)))
    for i in range(args.nreal):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sims[i], _ = simulate((d.x_cal, d.y_cal, z, ig, w), d.x, d.y, rng=rng)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.nreal} realisations", flush=True)

# ---- 1. data honoured
print("\n1. honoured data: max |realisation - data| at the calibration nodes =",
      f"{np.abs(sims[:, d.idx_cal] - z).max():.2e}")

# ---- 2. marginal distribution
ks = [stats.ks_2samp(s, d.z).statistic for s in sims]
ks_cal = stats.ks_2samp(d.z_cal, d.z).statistic
print(f"2. marginal: true field mean {d.z.mean():.3f} sd {d.z.std():.3f} | calibration set mean {d.z_cal.mean():.3f} "
      f"sd {d.z_cal.std():.3f} | realisations mean {sims.mean(axis=1).mean():.3f} (+-{sims.mean(axis=1).std():.3f}) "
      f"sd {sims.std(axis=1).mean():.3f} (+-{sims.std(axis=1).std():.3f})")
print(f"   KS distance to the true field: realisations {np.mean(ks):.3f} (range {np.min(ks):.3f}-{np.max(ks):.3f}); "
      f"calibration set {ks_cal:.3f}")

# ---- ensemble vs analytical HER prediction at the test nodes
P = Predictor(d.x_cal, d.y_cal, z, ig, w)
pm, _ = P.predict(d.x_test, d.y_test)
st = extract_pmf_statistics(pm, ig.edges_z, ig.bin_centers_edges_z, None)
sd_pmf = np.sqrt((pm * ig.bin_centers_edges_z ** 2).sum(axis=1) - st.mean ** 2)
ens = sims[:, d.idx_test]
zt = np.round(d.z_test, 5)
q05, q25, q75, q95 = np.quantile(ens, [0.05, 0.25, 0.75, 0.95], axis=0)
print(f"4. ensemble ({args.nreal} realisations) vs analytical HER at the {len(zt)} test nodes:")
print(f"   mean: r = {np.corrcoef(ens.mean(axis=0), st.mean)[0, 1]:.3f}, MAE(ensemble mean, E-type) = {np.mean(np.abs(ens.mean(axis=0) - st.mean)):.3f}"
      f" | spread: r = {np.corrcoef(ens.std(axis=0, ddof=1), sd_pmf)[0, 1]:.3f}, mean sd ensemble {ens.std(axis=0, ddof=1).mean():.3f} vs PMF {sd_pmf.mean():.3f}")
print(f"   coverage of the true test values: 50% interval {np.mean((zt >= q25) & (zt <= q75)):.2f}, "
      f"90% interval {np.mean((zt >= q05) & (zt <= q95)):.2f}  (nominal 0.50 / 0.90)")
cdf = np.concatenate((np.zeros((len(pm), 1)), np.cumsum(pm, axis=1)), axis=1)
lo_hi = {p: (np.array([np.interp(a, c, ig.edges_z) for c in cdf]), np.array([np.interp(1 - a, c, ig.edges_z) for c in cdf])) for a in (0.25, 0.05) for p in [1 - 2 * a]}
print("   coverage of the true test values by the analytical HER PMF intervals: "
      + ", ".join(f"{int(100 * p)}% interval {np.mean((zt >= lo) & (zt <= hi)):.2f}" for p, (lo, hi) in lo_hi.items()))
print(f"   MAE of the ensemble mean vs truth {np.mean(np.abs(ens.mean(axis=0) - zt)):.3f}; of the E-type {np.mean(np.abs(st.mean - zt)):.3f}")

# ---- 3. infograms (entropy of the delta-z PMF by distance class) on a random subset of nodes
sub = np.random.default_rng(0).choice(len(d.x), args.nsub, replace=False)
cfg_i = HerConfig(lag_dist=2.0, binwidth=0.2, binwidth_z=0.2)


def curve(x, y, zz, n=26):
    h = fit_infogram(x, y, np.round(zz, 5), cfg_i).H_diff_z_by_class
    return np.pad(h[:n], (0, max(0, n - len(h))), constant_values=np.nan)


H_true = curve(d.x[sub], d.y[sub], d.z[sub])
H_cal = curve(d.x_cal, d.y_cal, d.z_cal)
H_sims = np.array([curve(d.x[sub], d.y[sub], s[sub]) for s in sims[: min(args.nreal, 30)]])
etype = np.empty(len(d.x))
Pe, _ = Predictor(d.x_cal, d.y_cal, z, ig, w).predict(d.x, d.y)
etype = extract_pmf_statistics(Pe, ig.edges_z, ig.bin_centers_edges_z, None).mean
H_et = curve(d.x[sub], d.y[sub], etype[sub])
k = slice(0, 25)
print(f"3. infogram (entropy by class, first 25 classes): mean |H - H_true|: realisations {np.nanmean(np.abs(H_sims[:, k] - H_true[k])):.3f} bit, "
      f"calibration set {np.nanmean(np.abs(H_cal[k] - H_true[k])):.3f}, E-type map {np.nanmean(np.abs(H_et[k] - H_true[k])):.3f}")
print(f"   class 1: true {H_true[0]:.2f}, realisations {np.nanmean(H_sims[:, 0]):.2f} (+-{np.nanstd(H_sims[:, 0]):.2f}), "
      f"E-type {H_et[0]:.2f}, calibration {H_cal[0]:.2f} bit")

np.savez_compressed(out / f"hers_{args.field}_{args.size}.npz", sims=sims.astype(np.float32), x=d.x, y=d.y, z=d.z)

# ---- figures
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 5)
shape = (100, 100)
fields = [("true field", d.z, "viridis"), ("E-type of HER", etype, "viridis"), ("one realisation", sims[0], "viridis"),
          (f"mean of {args.nreal} realisations", sims.mean(axis=0), "viridis"), (f"sd of {args.nreal} realisations", sims.std(axis=0, ddof=1), "magma")]
for j, (title, v, cmap) in enumerate(fields):
    ax = fig.add_subplot(gs[0, j])
    lim = dict(vmin=d.z.min(), vmax=d.z.max()) if cmap == "viridis" else {}
    m = ax.pcolormesh(np.reshape(d.x, shape, order="F"), np.reshape(d.y, shape, order="F"), np.reshape(v, shape, order="F"), cmap=cmap, shading="auto", **lim)
    if j in (0, 1, 2):
        ax.scatter(d.x_cal, d.y_cal, s=2, c="k", marker=".")
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    plt.colorbar(m, ax=ax, shrink=0.7)
ax = fig.add_subplot(gs[1, 0:2])
xs = np.sort(d.z)
ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), "k", lw=3, label="true field")
ax.plot(np.sort(d.z_cal), np.arange(1, len(d.z_cal) + 1) / len(d.z_cal), "r", lw=2, label="calibration set")
for i, s in enumerate(sims):
    ax.plot(np.sort(s), np.arange(1, len(s) + 1) / len(s), color="0.6", lw=0.4, label="realisations" if i == 0 else None)
ax.set(xlabel="z", ylabel="cumulative probability", title="marginal CDF")
ax.legend()
ax = fig.add_subplot(gs[1, 2:4])
c = np.arange(1, 27)
for i, h in enumerate(H_sims):
    ax.plot(c, h, color="0.6", lw=0.4, label="realisations" if i == 0 else None)
ax.plot(c, H_true, "k", lw=3, label="true field")
ax.plot(c, H_cal, "r", lw=2, label="calibration set")
ax.plot(c, H_et, "b", lw=2, label="E-type map")
ax.axvline(ig.n_classes_range - 1, color="k", ls=":", lw=0.8)
ax.set(xlabel="distance class (width 2)", ylabel="entropy of the delta-z PMF [bit]", title="infogram")
ax.legend(fontsize=8)
ax = fig.add_subplot(gs[1, 4])
ax.scatter(sd_pmf, ens.std(axis=0, ddof=1), s=3, alpha=0.4)
lim = [0, max(sd_pmf.max(), ens.std(axis=0).max())]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set(xlabel="sd of the HER PMF", ylabel="sd of the ensemble", title="local spread (test nodes)")
fig.suptitle(f"HERs on {args.field}-{args.size} ({'nugget' if args.nugget else 'NOnugget'}, {args.nreal} realisations)")
fig.tight_layout()
fig.savefig(out / f"hers_{args.field}_{args.size}.png", dpi=80)
print("saved to", out)
