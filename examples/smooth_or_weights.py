"""How to smooth the OR class weights? Compares regularisation strategies against the paper's (MATLAB
fmincon) weights and the unconstrained optimum (``pyher.smooth``). AND weights are fixed to MATLAB's so
only the OR weights change; hyper-parameters are chosen by 5-fold cross-validation over the targets.

Output: examples/output/smooth_or/ (figures) and a printed table.
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyher import HerConfig, load_jura_basin, load_jura_dataset, plots
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W
from pyher.smooth import FastOrObjective, SmoothSpec, cross_validate, fit_or_weights
from pyher.stats import extract_pmf_statistics, validate
from pyher.weights import Weights, _slopes, evaluate_alpha_beta

here = Path(__file__).resolve().parent
data = here.parents[1] / "HERs" / "1.0.raw data"
out = here / "output" / "smooth_or"
out.mkdir(parents=True, exist_ok=True)

ds = load_jura_dataset(data / "Jura_dataset_Pb_log10.mat")
basin = load_jura_basin(data / "jura_basin.shp")
z, zv = np.round(ds.z_cal, 5), np.round(ds.z_val, 5)
thr = float(np.round(np.log10(50), 5))
cfg = HerConfig(n_neighbor=30, z_thresh=thr, nugget=False)
ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
fast = FastOrObjective(z, ig)
w_and = np.asarray(W["w_and"])
paper_or = np.asarray(W["w_or"]) / np.sum(W["w_or"])
edges = ig.edges_distance_classes_range
gx, gy = np.meshgrid(np.round(np.arange(0, 6.0001, 0.05), 2), np.round(np.arange(0, 6.0001, 0.05), 2))
inside = plots.inside_basin(gx, gy, basin)

# ---------------------------------------------------------------- cross-validation of the strength
grids = {
    "ratio": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "diff": [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    "prior": [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    "logprior": [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1],
}
cvs = {m: cross_validate(fast, m, g, k=5, seed=0) for m, g in grids.items()}
print("5-fold CV of the held-out threshold DKL (lower is better); [*] min, [1se] smoothest within 1 SE")
for m, cv in cvs.items():
    print(f"\n{m}")
    for i, p in enumerate(cv.params):
        tag = "[*]" if i == cv.best else ("[1se]" if i == cv.one_se else "")
        print(f"  {p:6g}  CV={cv.cv_mean[i]:.4f} +- {cv.cv_se[i]:.4f}   train={cv.train[i]:.4f} {tag}")
print(f"\nreference: paper weights train loss = {fast(paper_or):.4f}")

# ---------------------------------------------------------------- candidates
cands = {"paper (fmincon)": paper_or,
         "unconstrained": fit_or_weights(fast, SmoothSpec("none"))[0]}
for m, cv in cvs.items():
    cands[f"{m}={cv.params[cv.best]:g} (CV min)"] = fit_or_weights(fast, SmoothSpec(m, cv.params[cv.best]))[0]
    if cv.one_se != cv.best:
        cands[f"{m}={cv.params[cv.one_se]:g} (1-SE)"] = fit_or_weights(fast, SmoothSpec(m, cv.params[cv.one_se]))[0]
cands["powerlaw (1+d/a)^-p"] = fit_or_weights(fast, SmoothSpec("powerlaw"))[0]

# which logprior strength reproduces the training loss of the paper's weights (bisection in log lambda)?
target = fast(paper_or)
lo_l, hi_l = np.log(1e-6), np.log(10.0)
for _ in range(25):
    mid = 0.5 * (lo_l + hi_l)
    if fast(fit_or_weights(fast, SmoothSpec("logprior", np.exp(mid)))[0]) < target:
        lo_l = mid
    else:
        hi_l = mid
lam_paper = float(np.exp(0.5 * (lo_l + hi_l)))
cands[f"logprior={lam_paper:.2g} (matches paper LOO)"] = fit_or_weights(fast, SmoothSpec("logprior", lam_paper))[0]
print(f"\nlogprior strength with the same training loss as the paper's weights: lambda = {lam_paper:.3g}")


def smoothness(w):
    big = w > 1e-3 * w.max()
    r = w[1:][big[:-1]] / w[:-1][big[:-1]]
    return float(r.min()), float(np.exp(-np.sum(w[w > 0] * np.log(w[w > 0]))))


def model(w_or, alpha, beta, kind="andor"):
    wt = Weights(w_or, w_and, _slopes(w_or, edges, False), _slopes(w_and, edges, False), alpha, beta,
                 0, 0, 0, 0, None, None, None, None)
    ig_k = dataclasses.replace(ig, cfg=dataclasses.replace(ig.cfg, aggregation_type=kind))
    return Predictor(ds.x_cal, ds.y_cal, z, ig_k, wt)


def area_fractions(pmf):
    H = extract_pmf_statistics(pmf, ig.edges_z, ig.bin_centers_edges_z, thr).entropy
    H = H.reshape(gx.shape, order="F")[inside]
    return np.mean(H < 5) * 100, np.mean(H > 6.5) * 100, np.median(H)


rows, maps = [], {}
for name, w in cands.items():
    a, b, dkl_ab, *_ = evaluate_alpha_beta(z, ig, w, w_and)
    min_ratio, n_eff = smoothness(w)
    far = np.mean(np.abs(np.log(np.maximum(w[:10], 1e-12) / paper_or[:10])))
    pv_final, _ = model(w, a, b).predict(ds.x_val, ds.y_val)
    vf = validate(pv_final, zv, z, ig.edges_z, ig.bin_centers_edges_z, thr)
    P_or = model(w, 0.0, 1.0, "or")
    pv_or, _ = P_or.predict(ds.x_val, ds.y_val)
    vo = validate(pv_or, zv, z, ig.edges_z, ig.bin_centers_edges_z, thr)
    pg_or, _ = P_or.predict(gx.ravel("F"), gy.ravel("F"))
    maps[name] = pg_or
    pink, yel, medH = area_fractions(pg_or)
    rows.append((name, fast(w), min_ratio, n_eff, far, a, b, vf.DKL_score_01_largersmaller_thres_mean, vf.G,
                 vo.DKL_score_01_largersmaller_thres_mean, vo.G, pink, yel, medH))
hdr = ("candidate", "LOO", "minratio", "n_eff", "dlogw", "alpha", "beta", "val DKL", "val G", "ORval DKL",
       "ORval G", "OR H<5%", "OR H>6.5%", "OR medH")
print("\n" + " | ".join(f"{h:>9s}" if i else f"{h:<24s}" for i, h in enumerate(hdr)))
for r in rows:
    print(f"{r[0]:<24s} | " + " | ".join(f"{v:9.3f}" for v in r[1:]))
print("\nLOO: leave-one-out OR objective; minratio: smallest w[i+1]/w[i] among significant classes;"
      "\nn_eff: effective number of classes; dlogw: mean |log(w/w_paper)| over the first 10 classes;"
      "\nval *: final model (alpha/beta re-searched, AND fixed); ORval *: pure OR model on the validation set;"
      "\nOR H: entropy map of the pure OR model (area share below 5 bit / above 6.5 bit, median).")

# ---------------------------------------------------------------- figures
x = edges[1:]
unc = cands["unconstrained"]
plot_c = {}
for name, w in cands.items():  # drop copies of the unconstrained curve (strength 0)
    if name == "unconstrained" or not np.allclose(w, unc, rtol=1e-3, atol=1e-6):
        plot_c[name] = w
plot_c = {("unconstrained = any method with strength 0" if n == "unconstrained" else n): w
          for n, w in plot_c.items()}
colors = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for i, (name, w) in enumerate(plot_c.items()):
    if name.startswith("paper"):
        style = dict(lw=3.5, color="k", zorder=5)
    elif "matches paper" in name:
        style = dict(lw=2.2, color="tab:red", ls="--", zorder=6)
    else:
        style = dict(lw=1.4, color=colors(i % 10), ls=["-", "-.", ":"][i // 10])
    for ax in axes:
        ax.plot(x, w, "o-" if "ls" not in style else "o", ms=3, label=name, **style)
axes[0].set(xlim=(0, 1.4), ylim=(0, 0.55), xlabel="Euclidean distance (km)", ylabel="OR weight (sum = 1)")
axes[1].set(xlim=(0, 1.4), yscale="log", ylim=(1e-5, 1), xlabel="Euclidean distance (km)")
axes[0].legend(fontsize=7)
fig.suptitle("OR class weights: paper (black) vs. regularised solutions; red dashed = logprior shrinkage matched to the paper")
fig.tight_layout()
fig.savefig(out / "or_weights.png", dpi=90)

fig, axes = plt.subplots(1, len(cvs), figsize=(5 * len(cvs), 3.8))
for ax, (m, cv) in zip(axes, cvs.items()):
    xs = np.arange(len(cv.params))
    ax.errorbar(xs, cv.cv_mean, cv.cv_se, fmt="o-", label="held-out (5-fold CV)")
    ax.plot(xs, cv.train, "s--", label="train (all targets)")
    ax.axhline(fast(paper_or), color="k", ls=":", label="paper weights (LOO)")
    ax.set_xticks(xs, [f"{p:g}" for p in cv.params], fontsize=7)
    ax.set(title=m, xlabel="strength", ylabel="threshold DKL [bit]")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "cross_validation.png", dpi=90)

show = [n for n in cands if n.startswith(("paper", "unconstrained", "ratio=0.6", "logprior"))][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for ax, name in zip(axes.ravel(), show):
    H = extract_pmf_statistics(maps[name], ig.edges_z, ig.bin_centers_edges_z, thr).entropy
    m = ax.pcolormesh(gx, gy, np.where(inside, H.reshape(gx.shape, order="F"), np.nan), cmap="spring",
                      vmin=4, vmax=9, shading="auto")
    for part in basin:
        ax.plot(part[:, 0], part[:, 1], "k", lw=1)
    ax.set(aspect="equal", title=name, xticks=[], yticks=[])
    plt.colorbar(m, ax=ax, shrink=0.7)
fig.suptitle("Entropy of the pure-OR model (bit), 30 neighbours")
fig.tight_layout()
fig.savefig(out / "entropy_or_model.png", dpi=80)
print("figures in", out)
