"""Test pyher on the synthetic data sets of Thiesen et al. (2020, HESS) with the paper's parameters.

Data: https://github.com/KIT-HYD/HER/tree/master/datasets (SR0, SR1, LR0, LR1; 100 x 100 grid; calibration
sizes 200 ... 2000; the same 2000 validation and 2000 test points for every size).
Parameters (paper, Sect. 3-4 and Supplement S2): lag (class width) 2 grid units, bin width 0.2 for z and
delta z, 12 neighbours, objective = DKL of the single bin (``objective="bin"``), HESS version of the code
(``nugget=True``), AND/OR aggregation. The performance is computed on the test set.

Usage: python reproduce_hess2020.py <folder with the .mat files> [--fields LR1 ...] [--sizes 200 600 ...]
                                    [--shrinkage 0.0003]   (default: unshrunk OR weights)
Output: examples/output/hess2020/ (csv + figure)
"""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from pyher import HerConfig, load_her_synthetic
from pyher.core import performance_prob
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.stats import extract_pmf_statistics, performance_det
from pyher.weights import optimize_weights

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--fields", nargs="*", default=["SR0", "SR1", "LR0", "LR1"])
ap.add_argument("--sizes", nargs="*", type=int, default=[200, 400, 600, 800, 1000, 1500, 2000])
ap.add_argument("--shrinkage", type=float, default=None)
ap.add_argument("--paper-params", default=None, help="json with the alpha/beta/range of Supplement S2")
args = ap.parse_args()

folder = Path(args.folder)
out = Path(__file__).resolve().parent / "output" / "hess2020"
out.mkdir(parents=True, exist_ok=True)
paper = json.load(open(args.paper_params)) if args.paper_params else {}
all_sizes = [200, 400, 600, 800, 1000, 1500, 2000]


def nn_idw(d, power=2):
    tree = cKDTree(np.column_stack((d.x_cal, d.y_cal)))
    dist, idx = tree.query(np.column_stack((d.x_test, d.y_test)), k=len(d.x_cal))
    nn = d.z_cal[idx[:, 0]]
    w = 1.0 / np.maximum(dist, 1e-12) ** power
    return nn, (w * d.z_cal[idx]).sum(axis=1) / w.sum(axis=1)


rows = []
for field in args.fields:
    for n in args.sizes:
        t0 = time.time()
        d = load_her_synthetic(folder / f"{field}_input_randomfield_cal{n}.mat")
        z, zt = np.round(d.z_cal, 5), np.round(d.z_test, 5)
        cfg = HerConfig(lag_dist=2.0, binwidth=0.2, binwidth_z=0.2, n_neighbor=12, objective="bin", nugget=True)
        ig = fit_infogram(d.x_cal, d.y_cal, z, cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = optimize_weights(d.x_cal, d.y_cal, z, ig, or_shrinkage=args.shrinkage)
        pm, _ = Predictor(d.x_cal, d.y_cal, z, ig, w).predict(d.x_test, d.y_test)
        st = extract_pmf_statistics(pm, ig.edges_z, ig.bin_centers_edges_z, None)
        det = performance_det(st.mean, zt)
        nn, idw = nn_idw(d)
        i = all_sizes.index(n)
        p = paper.get(field, {})
        row = dict(field=field, n=n, range=2 * (ig.n_classes_range - 1), alpha=w.best_alpha, beta=w.best_beta,
                   EMA=det.MAE, ENS=det.NSE, DKL=performance_prob(zt, pm, ig.edges_z),
                   NN_EMA=np.mean(np.abs(nn - zt)), NN_ENS=1 - np.sum((nn - zt) ** 2) / np.sum((zt - zt.mean()) ** 2),
                   IDS_EMA=np.mean(np.abs(idw - zt)), IDS_ENS=1 - np.sum((idw - zt) ** 2) / np.sum((zt - zt.mean()) ** 2),
                   paper_range=p.get("range", [np.nan] * 7)[i], paper_alpha=p.get("alpha", [np.nan] * 7)[i],
                   paper_beta=p.get("beta", [np.nan] * 7)[i])
        rows.append(row)
        print(f"{field}-{n:<4d} {time.time() - t0:4.0f}s  range {row['range']:4.0f} (paper {row['paper_range']:4.0f})  "
              f"alpha {row['alpha']:.2f} (paper {row['paper_alpha']:.2f})  beta {row['beta']:.2f} (paper {row['paper_beta']:.2f})  "
              f"EMA {row['EMA']:.3f} ENS {row['ENS']:.3f} DKL {row['DKL']:.2f} | NN {row['NN_EMA']:.3f}/{row['NN_ENS']:.3f} "
              f"IDS {row['IDS_EMA']:.3f}/{row['IDS_ENS']:.3f}", flush=True)

tag = "unshrunk" if args.shrinkage is None else f"shrink{args.shrinkage:g}"
with open(out / f"benchmark_{tag}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

fields = args.fields
fig, axes = plt.subplots(3, len(fields), figsize=(4.2 * len(fields), 9), squeeze=False)
for j, field in enumerate(fields):
    r = [x for x in rows if x["field"] == field]
    ns = [x["n"] for x in r]
    for i, (key, label) in enumerate((("EMA", "mean absolute error"), ("ENS", "Nash-Sutcliffe"), ("DKL", "mean DKL [bit]"))):
        ax = axes[i, j]
        ax.plot(ns, [x[key] for x in r], "o-", label="HER (AND/OR)")
        if key != "DKL":
            ax.plot(ns, [x["NN_" + key] for x in r], "s--", label="NN")
            ax.plot(ns, [x["IDS_" + key] for x in r], "^--", label="IDS")
        ax.set(title=field if i == 0 else "", xlabel="calibration size" if i == 2 else "", ylabel=label)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"pyher on the HESS 2020 synthetic data (OR weights: {tag})")
fig.tight_layout()
fig.savefig(out / f"benchmark_{tag}.png", dpi=90)
print("saved to", out)
