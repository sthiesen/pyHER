"""Compare MATLAB and Python maps built from the SAME infogram classes and AND/OR weights.

MATLAB side: ``tests/golden_gen/gen_maps.m`` runs the original f_her3_predict / f_her4_HERs with the
weights, alpha/beta and classes stored in ``tests/golden/her2.mat``. Python side: the infogram is
refitted by ``pyher`` (it reproduces MATLAB's to 1e-12, see tests), and the *MATLAB* weights are
injected instead of re-optimising. Usage: ``python compare_with_matlab.py <folder with *_matlab.mat>``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from pyher import HerConfig, load_jura_basin, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.plots import inside_basin
from pyher.predict import Predictor, simulate
from pyher.stats import extract_pmf_statistics
from pyher.weights import Weights

here = Path(__file__).resolve().parent
repo = here.parents[1]
golden = here.parent / "tests" / "golden"
src = Path(sys.argv[1])
out = here / "output"
out.mkdir(exist_ok=True)


def mat(path):
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


ds = load_jura_dataset(repo / "HERs" / "1.0.raw data" / "Jura_dataset_Pb_log10.mat")
basin = load_jura_basin(repo / "HERs" / "1.0.raw data" / "jura_basin.shp")
z = np.round(ds.z_cal, 5)
cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=9,
                z_thresh=float(np.round(np.log10(50), 5)))
ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)

# --- the shared model: MATLAB's weights on the (identical) classes
r = mat(golden / "her2.mat")["her2"]
w = Weights(
    best_w_OR=np.atleast_1d(r.best_w_OR), best_w_AND=np.atleast_1d(r.best_w_AND),
    w_OR_slope=np.atleast_1d(r.w_OR_slope), w_AND_slope=np.atleast_1d(r.w_AND_slope),
    best_alpha=float(r.best_alpha), best_beta=float(r.best_beta), DKL_w_OR=float(r.DKL_w_OR),
    DKL_w_AND=float(r.DKL_w_AND), DKL_w_alpha_beta=float(r.DKL_w_alpha_beta),
    n_neighbor_aggreg=int(r.n_neighbor_aggreg), idx_opt_nn=None, n_pairs_by_class_obs=None,
    nmean_pairs_by_class=None, target_idx_zero_neigh=None)
print("shared inputs  (Python infogram vs MATLAB her2 struct):")
for name in ("edges_distance_classes_range", "edges_z", "pmf_diff_z_by_class_obs_range_shift"):
    a, b = np.asarray(getattr(ig, name)), np.asarray(getattr(r, name))
    print(f"  {name:42s} max|diff| = {np.abs(a - b.reshape(a.shape)).max():.2e}")
print(f"  alpha={w.best_alpha:.2f} beta={w.best_beta:.2f}, {len(w.best_w_OR)} classes")

# --- HER3 maps
m = mat(src / "maps_matlab.mat")
gx, gy = m["gx"], m["gy"]
ny, nx = gx.shape
pmf_py, zero_py = Predictor(ds.x_cal, ds.y_cal, z, ig, w).predict(gx.ravel("F"), gy.ravel("F"))
st = extract_pmf_statistics(pmf_py, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
d_pmf = np.abs(pmf_py - m["pmf"])
print(f"\nHER3 on {nx}x{ny} grid: max|PMF diff| = {d_pmf.max():.2e}  "
      f"(zero-neighbour targets equal: {np.array_equal(zero_py + 1, np.atleast_1d(m['idx_zero']))})")
fields = {
    "E-type (mean)": (m["zmean"], st.mean, "viridis", (1.2, 2.5)),
    "Entropy [bit]": (m["H"], st.entropy, "spring", None),
    "P(z > thresh)": (m["zprob"], st.prob_exceed, "Greys", (0, 1)),
}
inside = inside_basin(gx, gy, basin).ravel("F")
rows = list(fields.items())

# --- HERs4 replay
s = mat(src / "sim_matlab.mat")
sx, sy = s["sx"], s["sy"]
z_py, _ = simulate((ds.x_cal, ds.y_cal, z, ig, w), sx.ravel("F"), sy.ravel("F"),
                   perm=np.atleast_1d(s["rec"].perm) - 1, p_values=np.atleast_1d(s["rec"].p_value))
z_ml = np.atleast_1d(s["z_realiz"])
print(f"HERs4 on {sx.shape[1]}x{sx.shape[0]} grid (replaying MATLAB's random numbers): "
      f"max|diff| = {np.abs(z_py - z_ml).max():.2e}")
sim_inside = inside_basin(sx, sy, basin).ravel("F")

fig, axes = plt.subplots(len(rows) + 1, 3, figsize=(13, 4.2 * (len(rows) + 1)))


def draw(ax, X, Y, vals, ins, cmap, lim, title):
    v = np.where(ins, vals, np.nan).reshape(X.shape, order="F")
    kw = {} if lim is None else dict(vmin=lim[0], vmax=lim[1])
    mesh = ax.pcolormesh(X, Y, v, cmap=cmap, shading="auto", **kw)
    for part in basin:
        ax.plot(part[:, 0], part[:, 1], "k", lw=1.2)
    ax.set(aspect="equal", title=title, xticks=[], yticks=[])
    plt.colorbar(mesh, ax=ax, shrink=0.75)


for i, (name, (a, b, cmap, lim)) in enumerate(rows):
    lim = lim or (np.nanmin(a[inside]), np.nanmax(a[inside]))
    draw(axes[i, 0], gx, gy, a, inside, cmap, lim, f"MATLAB: {name}")
    draw(axes[i, 1], gx, gy, b, inside, cmap, lim, f"Python: {name}")
    diff = b - a
    dmax = np.nanmax(np.abs(diff[inside]))
    draw(axes[i, 2], gx, gy, diff, inside, "coolwarm", (-max(dmax, 1e-16), max(dmax, 1e-16)),
         f"Python - MATLAB (max {dmax:.1e})")
    print(f"  {name:15s} max|diff| inside basin = {dmax:.2e}")
draw(axes[-1, 0], sx, sy, z_ml, sim_inside, "viridis", (1.2, 2.5), "MATLAB: HERs realization")
draw(axes[-1, 1], sx, sy, z_py, sim_inside, "viridis", (1.2, 2.5), "Python: HERs realization (replay)")
dz = z_py - z_ml
dm = max(np.nanmax(np.abs(dz[sim_inside])), 1e-16)
draw(axes[-1, 2], sx, sy, dz, sim_inside, "coolwarm", (-dm, dm), f"Python - MATLAB (max {dm:.1e})")
fig.suptitle("Same infogram classes and AND/OR weights (MATLAB optimum), alpha/beta = "
             f"{w.best_alpha:.2f}/{w.best_beta:.2f}", y=0.995)
fig.tight_layout()
fig.savefig(out / "compare_matlab_python_maps.png", dpi=90)
print("figure:", out / "compare_matlab_python_maps.png")
