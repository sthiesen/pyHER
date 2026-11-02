"""Paper-style figures: (1) class weights, (2) the four maps of Fig. 17 with MATLAB's AND/OR weights.

Usage: python paper_figures.py <folder with the MATLAB map file> [--nonugget] [--nn N] [--alpha A] [--beta B]

Default: nugget version (sHER_E_04_0, 9 neighbours; MATLAB file maps_matlab_a065.mat from gen_maps_a065.m).
--nonugget --nn 30: the version and settings of the paper (sHER_E_04_1_NOnugget; MATLAB file
maps_matlab_nonugget.mat from gen_nonugget.m).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.io

from pyher import HerConfig, load_jura_basin, load_jura_dataset, plots
from pyher.core import entropy
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.stats import extract_pmf_statistics
from pyher.weights import Weights, _slopes, optimize_weights

here = Path(__file__).resolve().parent
repo = here.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--nonugget", action="store_true")
ap.add_argument("--nn", type=int, default=9)
ap.add_argument("--alpha", type=float, default=0.65)
ap.add_argument("--beta", type=float, default=0.0)
args = ap.parse_args()
src, alpha, beta = Path(args.src), args.alpha, args.beta
tag_v = "nonugget" if args.nonugget else "nugget"
mat_file = "maps_matlab_nonugget.mat" if args.nonugget else "maps_matlab_a065.mat"
out = here / "output"
out.mkdir(exist_ok=True)

ds = load_jura_dataset(repo / "HERs" / "1.0.raw data" / "Jura_dataset_Pb_log10.mat")
basin = load_jura_basin(repo / "HERs" / "1.0.raw data" / "jura_basin.shp")
z = np.round(ds.z_cal, 5)
cfg = HerConfig(n_neighbor=args.nn, z_thresh=float(np.round(np.log10(50), 5)), nugget=not args.nonugget)
ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)

r = scipy.io.loadmat(here.parent / "tests" / "golden" / "her2.mat", squeeze_me=True,
                     struct_as_record=False)["her2"]
w_ml = Weights(
    best_w_OR=np.atleast_1d(r.best_w_OR), best_w_AND=np.atleast_1d(r.best_w_AND),
    w_OR_slope=_slopes(np.atleast_1d(r.best_w_OR), ig.edges_distance_classes_range, cfg.nugget),
    w_AND_slope=_slopes(np.atleast_1d(r.best_w_AND), ig.edges_distance_classes_range, cfg.nugget),
    best_alpha=alpha, best_beta=beta, DKL_w_OR=float(r.DKL_w_OR), DKL_w_AND=float(r.DKL_w_AND),
    DKL_w_alpha_beta=float(r.DKL_w_alpha_beta), n_neighbor_aggreg=9, idx_opt_nn=None,
    n_pairs_by_class_obs=None, nmean_pairs_by_class=None, target_idx_zero_neigh=None)
w_py = optimize_weights(ds.x_cal, ds.y_cal, z, ig)

# --- 1. weights
fig = plots.plot_weights_paper(ig, {
    "MATLAB (fmincon)  DKL_OR=%.3f  DKL_AND=%.3f" % (w_ml.DKL_w_OR, w_ml.DKL_w_AND):
        (w_ml.best_w_OR, w_ml.best_w_AND),
    "Python (SLSQP)  DKL_OR=%.3f  DKL_AND=%.3f" % (w_py.DKL_w_OR, w_py.DKL_w_AND):
        (w_py.best_w_OR, w_py.best_w_AND)})
fig.savefig(out / f"weights_paper_style_{tag_v}.png", dpi=100)
print("class edges (km):", np.round(ig.edges_distance_classes_range[:-1], 2))
print("MATLAB OR :", np.round(w_ml.best_w_OR[:8], 3))
print("MATLAB AND:", np.round(w_ml.best_w_AND[:8], 3))
print("Python OR :", np.round(w_py.best_w_OR[:8], 3))
print("Python AND:", np.round(w_py.best_w_AND[:8], 3))

# --- 2. maps with the MATLAB weights
m = scipy.io.loadmat(src / mat_file, squeeze_me=True, struct_as_record=False)
gx, gy = m["gx"], m["gy"]
pmf, _ = Predictor(ds.x_cal, ds.y_cal, z, ig, w_ml).predict(gx.ravel("F"), gy.ravel("F"))
st = extract_pmf_statistics(pmf, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
print(f"\nalpha={alpha} beta={beta}: max|PMF diff| = {np.abs(pmf - m['pmf']).max():.1e}, "
      f"max|E-type diff| = {np.abs(st.mean - m['zmean']).max():.1e}, "
      f"max|entropy diff| = {np.abs(st.entropy - m['H']).max():.1e}, "
      f"max|prob diff| = {np.abs(st.prob_exceed - m['zprob']).max():.1e}")
h = np.histogram(z, bins=ig.edges_z)[0]
H_marg = entropy(h / h.sum())
shape = gx.shape
for tag, (mean, ent, prob) in {
        "python": (st.mean, st.entropy, st.prob_exceed),
        "matlab": (m["zmean"], m["H"], m["zprob"])}.items():
    f = plots.plot_paper_maps(gx, gy, mean.reshape(shape, order="F"), ent.reshape(shape, order="F"),
                              prob.reshape(shape, order="F"), ds.x_cal, ds.y_cal, z, cfg.z_thresh,
                              H_marg, basin, label="HER (%s)" % tag)
    f.savefig(out / f"paper_maps_{tag_v}_nn{args.nn}_{tag}.png", dpi=85)
print("H marginal =", round(H_marg, 3), " marginal prob =", round(float(np.mean(z > cfg.z_thresh)), 3))
