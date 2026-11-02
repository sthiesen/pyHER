"""Single effect of the number of neighbours: same infogram, same class weights (MATLAB/paper),
alpha = 0.65, beta = 0, NOnugget version; only ``n_neighbor`` changes.

Usage: python neighbors_effect.py [n1 n2 ...]   (default 9 30 150)
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyher import HerConfig, load_jura_basin, load_jura_dataset, plots
from pyher.core import entropy
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.reference import JURA_PB_PAPER_ALPHA_BETA
from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W
from pyher.stats import extract_pmf_statistics, validate
from pyher.weights import optimize_weights

here = Path(__file__).resolve().parent
data = here.parents[1] / "HERs" / "1.0.raw data"
out = here / "output"
out.mkdir(exist_ok=True)
nns = [int(a) for a in sys.argv[1:]] or [9, 30, 150]

ds = load_jura_dataset(data / "Jura_dataset_Pb_log10.mat")
basin = load_jura_basin(data / "jura_basin.shp")
z, zv = np.round(ds.z_cal, 5), np.round(ds.z_val, 5)
thr = float(np.round(np.log10(50), 5))
gx, gy = np.meshgrid(np.round(np.arange(0, 6.0001, 0.05), 2), np.round(np.arange(0, 6.0001, 0.05), 2))
inside = plots.inside_basin(gx, gy, basin)
shape = gx.shape

res = {}
print(f"{'n_neighbor':>10} {'time[s]':>8} | entropy area <4 | 4-5 | 5-5.35 | 5.35-6.5 | 6.5-7.5 | >7.5 [%]   "
      f"| median H | validation: G   DKL01  misclass@marg  RMSE")
for nn in nns:
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=nn, z_thresh=thr,
                    nugget=False)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, fixed_weights=(W["w_or"], W["w_and"]),
                         alpha_beta=JURA_PB_PAPER_ALPHA_BETA)
    t0 = time.time()
    P = Predictor(ds.x_cal, ds.y_cal, z, ig, w)
    pmf, _ = P.predict(gx.ravel("F"), gy.ravel("F"))
    dt = time.time() - t0
    st = extract_pmf_statistics(pmf, ig.edges_z, ig.bin_centers_edges_z, thr)
    pv, _ = P.predict(ds.x_val, ds.y_val)
    v = validate(pv, zv, z, ig.edges_z, ig.bin_centers_edges_z, thr)
    H = st.entropy.reshape(shape, order="F")
    frac = np.histogram(H[inside], bins=[0, 4, 5, 5.348, 6.5, 7.5, 99])[0] / inside.sum() * 100
    print(f"{nn:>10d} {dt:8.1f} | " + " | ".join(f"{f:5.1f}" for f in frac)
          + f"   | {np.median(H[inside]):5.2f}    | {v.G:.3f}  {v.DKL_score_01_largersmaller_thres_mean:.3f}"
          f"  {v.misclass_marginal:5.1f}%        {v.deterministic.RMSE:.3f}")
    res[nn] = dict(mean=st.mean.reshape(shape, order="F"), H=H, prob=st.prob_exceed.reshape(shape, order="F"))
    h = np.histogram(z, bins=ig.edges_z)[0]
    fig = plots.plot_paper_maps(gx, gy, res[nn]["mean"], H, res[nn]["prob"], ds.x_cal, ds.y_cal, z, thr,
                                entropy(h / h.sum()), basin, label=f"HER, {nn} neighbours")
    fig.savefig(out / f"paper_maps_nonugget_nn{nn}.png", dpi=85)
    plt.close(fig)

# side-by-side comparison, and differences with respect to the first setting
ref = nns[0]
rows = [("E-type", "mean", "viridis", (1.2, 2.5), (-0.15, 0.15)),
        ("Entropy [bit]", "H", "spring", (4, 9), (-2, 2)),
        ("P(z > z_c)", "prob", "Greys", (0, 1), (-0.3, 0.3))]
fig, axes = plt.subplots(len(rows), 2 * len(nns) - 1, figsize=(4.0 * (2 * len(nns) - 1), 4.0 * len(rows)))
for i, (name, key, cmap, lim, dlim) in enumerate(rows):
    col = 0
    for nn in nns:
        for kind in ("abs", "diff"):
            if kind == "diff" and nn == ref:
                continue
            ax = axes[i, col]
            col += 1
            vals = res[nn][key] if kind == "abs" else res[nn][key] - res[ref][key]
            vals = np.where(inside, vals, np.nan)
            m = ax.pcolormesh(gx, gy, vals, cmap=cmap if kind == "abs" else "RdBu_r", shading="auto",
                              vmin=(lim if kind == "abs" else dlim)[0], vmax=(lim if kind == "abs" else dlim)[1])
            plt.colorbar(m, ax=ax, shrink=0.7)
            for part in basin:
                ax.plot(part[:, 0], part[:, 1], "k", lw=1)
            ax.set(aspect="equal", xticks=[], yticks=[],
                   title=f"{name}: {nn} neighbours" if kind == "abs" else f"{nn} minus {ref} neighbours")
fig.tight_layout()
fig.savefig(out / "neighbors_effect.png", dpi=75)
print("figures in", out)
