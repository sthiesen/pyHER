"""Jura lead example: HER1-HERs4, PMF statistics, validation metrics and figures.

Settings of the paper (Thiesen & Ehret 2021): NOnugget variant, 30 neighbours, lag 0.07 km, bin width 0.015.

* default: class weights and alpha/beta are optimised in Python; the OR weights are shrunk towards the
  1/distance guess (``or_shrinkage``), which reproduces the smooth OR weights of the paper;
* ``--no-shrinkage``: the exact optimum of the objective instead (OR weights collapse onto the first
  two classes, see the README);
* ``--bin``: optimise the single-bin DKL (sHER_E_03, HESS 2020 paper) instead of the threshold DKL;
* ``--paper-weights``: use the MATLAB/paper class weights from ``pyher.reference`` with alpha = 0.65,
  beta = 0 (no optimisation at all).

Figures go to examples/output/{optimized,no_shrinkage,paper_weights}/."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from pyher import HerConfig, load_jura_basin, load_jura_dataset, plots
from pyher.core import entropy
from pyher.pipeline import run_her
from pyher.reference import JURA_PB_PAPER_ALPHA_BETA, JURA_PB_THRESHOLD_WEIGHTS
from pyher.stats import extract_pmf_statistics

data = Path(__file__).resolve().parents[2] / "HERs" / "1.0.raw data"


ds = load_jura_dataset(data / "Jura_dataset_Pb_log10.mat")
basin = load_jura_basin(data / "jura_basin.shp")
cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=30,
                z_thresh=float(np.round(np.log10(50), 5)), aggregation_type="andor",
                objective="bin" if "--bin" in sys.argv else "threshold", nugget=False)
paper = "--paper-weights" in sys.argv
no_shrink = "--no-shrinkage" in sys.argv
out = Path(__file__).resolve().parent / "output" / (
    "paper_weights" if paper else "no_shrinkage" if no_shrink else "optimized")
out.mkdir(parents=True, exist_ok=True)
fixed = (JURA_PB_THRESHOLD_WEIGHTS["w_or"], JURA_PB_THRESHOLD_WEIGHTS["w_and"]) if paper else None

res = run_her(ds.x_cal, ds.y_cal, ds.z_cal, cfg, grid=(0, 6, 0.05), nsim=1, nsim_val=100, seed=1,
              validation=(ds.x_val, ds.y_val, ds.z_val), fixed_weights=fixed,
              alpha_beta=JURA_PB_PAPER_ALPHA_BETA if paper else None,
              or_shrinkage=None if no_shrink else "auto")
ig, w = res.infogram, res.weights
print(f"alpha={w.best_alpha:.2f} beta={w.best_beta:.2f}  DKL OR/AND/alpha-beta = "
      f"{w.DKL_w_OR:.3f}/{w.DKL_w_AND:.3f}/{w.DKL_w_alpha_beta:.3f}")
for name, v in (("HER ", res.validation), ("HERs", res.validation_sim)):
    print(f"{name} validation: RMSE={v.deterministic.RMSE:.3f} MAE={v.deterministic.MAE:.3f} "
          f"NSE={v.deterministic.NSE:.3f} G={v.G:.3f} misclass@marginal={v.misclass_marginal:.1f}%")

# statistics of the predicted PMFs on the grid
ny, nx, nb = res.pmf_pred.shape
st = extract_pmf_statistics(res.pmf_pred.reshape(-1, nb), ig.edges_z, ig.bin_centers_edges_z,
                            cfg.z_thresh)
shape = (ny, nx)
figs = {}
figs["infogram"] = plots.plot_infogram(ig, ds.txt)
figs["weights"] = plots.plot_weights(w, ig)
figs["paper_maps"] = [plots.plot_paper_maps(
    res.grid_x, res.grid_y, st.mean.reshape(shape), st.entropy.reshape(shape),
    st.prob_exceed.reshape(shape), ds.x_cal, ds.y_cal, np.round(ds.z_cal, 5), cfg.z_thresh,
    entropy(np.histogram(np.round(ds.z_cal, 5), bins=ig.edges_z)[0] / len(ds.z_cal)), basin)]
figs["prediction"] = plots.plot_prediction(
    st.mean.reshape(shape), st.entropy.reshape(shape), res.pmf_pred, res.grid_x, res.grid_y, ds, ig,
    [3.6, 4.0], [4.4, 4.7], basin, txt=ds.txt)
figs["probability_map"] = plots.plot_probability_map(
    st.prob_exceed.reshape(shape), cfg.z_thresh, res.grid_x, res.grid_y, ds, basin, ds.txt)
figs["realization"] = [plots.plot_realization(res.z_realiz[:, :, 0], res.grid_x, res.grid_y, ds, basin,
                                               title="HERs realization #1")]
for name, fl in figs.items():
    for i, fig in enumerate(fl):
        fig.savefig(out / f"{name}_{i + 1}.png", dpi=110)
print("figures in", out)
