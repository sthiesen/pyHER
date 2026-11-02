"""Bin-width selection by cross-entropy convergence (pyher.binwidth), applied to the two data sets of the papers.

Two readings are given. (A) excess at the actual class size (smooth bootstrap, no subsample fraction), (B) the
subsample-convergence curves. Procedure of (B), for several candidate bin widths:
 1. delta-z of all point pairs of a distance class = the "full learning set" p of that class (empirical PMF with a
    pseudo-count of 1 in every bin);
 2. draw random subsamples without replacement of growing size n (500 repetitions), build their PMFs q in the same
    way and compute the excess of the cross entropy Hpq - H(p) = DKL(p || q) [bit];
 3. a bin width is reasonable when this excess is small already for small samples.
The class that is checked is the one with the FEWEST pairs among the first 10 classes (the closest classes are the
most used in the prediction; the class with the fewest pairs is the most demanding one). As a check, all of the first
10 classes are also analysed one by one. The papers chose 0.015 (Jura lead, log10 Pb) and 0.2 (HESS synthetic fields).

Usage: python binwidth_selection.py [folder with the HESS .mat files]   (the HESS part is skipped without it)
Output: examples/output/binwidth/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyher import load_her_synthetic, load_jura_dataset, plots
from pyher.binwidth import actual_size_excess, classwise_cross_entropy, limiting_class_cross_entropy

here = Path(__file__).resolve().parent
out = here / "output" / "binwidth"
out.mkdir(parents=True, exist_ok=True)
N_FIRST = 10
FRACTION = 0.2  # subsample size at which the excess is read: 20 % of the values of the class

cases = []
jura = load_jura_dataset(here.parents[1] / "HERs" / "1.0.raw data" / "Jura_dataset_Pb_log10.mat")
cases.append(("Jura log10(Pb), 259 points", jura.x_cal, jura.y_cal, np.round(jura.z_cal, 5), 0.07, 0.015,
              [0.0075, 0.015, 0.03, 0.06, 0.12, 0.24]))
if len(sys.argv) > 1:
    folder = Path(sys.argv[1])
    for n in (200, 600):
        d = load_her_synthetic(folder / f"LR1_input_randomfield_cal{n}.mat")
        cases.append((f"HESS LR1, {n} points", d.x_cal, d.y_cal, np.round(d.z_cal, 5), 2.0, 0.2,
                      [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]))

for name, x, y, z, lag, chosen, bws in cases:
    k, res = limiting_class_cross_entropy(x, y, z, lag, bws, n_first=N_FIRST, n_rep=500, seed=0)
    n_target = int(res.sample_sizes[np.argmin(np.abs(res.sample_sizes - FRACTION * res.n_data))])
    print(f"\n=== {name}; the papers used bin width {chosen:g}")
    print(f"limiting class = class {k} (fewest pairs of the first {N_FIRST} classes): {res.n_data} delta-z values")
    print("DKL(p||q) [bit] = Hpq - H(p), mean of 500 random subsamples of size n")
    need = res.samples_needed(0.1)
    show = res.sample_sizes[::2]
    print(f"{'bin width':>10} {'bins':>5} {'H(p)':>6} | " + " ".join(f"n={n:<5d}" for n in show) + " | n for 0.1 bit")
    for i, bw in enumerate(res.binwidths):
        mark = "  <- papers" if np.isclose(bw, chosen) else ""
        print(f"{bw:10g} {res.n_bins[i]:5d} {res.H_p[i]:6.2f} | " + " ".join(f"{v:7.3f}" for v in res.dkl_mean[i, ::2])
              + f" | {need[i]:8.0f}" + mark)
    for tol in (0.05, 0.1, 0.2):
        print(f"  smallest bin width with DKL <= {tol:g} bit at n = {n_target} (20% of the class): "
              f"{res.recommend(n_target, tol)}")
    tag = f"{name.split(',')[0].replace(' ', '_').replace('(', '').replace(')', '')}_{len(x)}"
    fig = plots.plot_binwidth_selection(res, n_target=n_target, dkl_tol=0.1, title=f"{name}, class {k}:")
    fig.savefig(out / f"binwidth_{tag}.png", dpi=90)
    plt.close(fig)

    # ---- excess at the ACTUAL class size (smooth bootstrap): what the class PMF used by HER loses
    ac = actual_size_excess(x, y, z, lag, bws, n_first=N_FIRST, n_rep=200)
    print(f"\n  excess DKL [bit] of the class PMF built from ALL its values (smooth bootstrap), first {N_FIRST} classes")
    print("  class  pairs | " + " ".join(f"bw={b:<6g}" for b in ac.binwidths))
    for j in range(len(ac.class_index)):
        flag = "  <- fewest pairs" if ac.class_index[j] == ac.limiting_class else ""
        print(f"  {ac.class_index[j]:5d} {ac.n_values[j]:6d} | " + " ".join(f"{v:9.3f}" for v in ac.excess[j]) + flag)
    for tol in (0.05, 0.1, 0.2):
        print(f"  tolerance {tol:g} bit: smallest bin width: limiting class {ac.recommend(tol, [ac.limiting_class])}, "
              f"all {N_FIRST} classes {ac.recommend(tol)}")
    fig = plots.plot_binwidth_actual(ac, dkl_tol=0.1, chosen=chosen, title=name)
    fig.savefig(out / f"binwidth_actual_{tag}.png", dpi=90)
    plt.close(fig)

    # ---- check: every one of the first classes on its own (20 % of each class)
    cw = classwise_cross_entropy(x, y, z, lag, N_FIRST, bws, fraction=FRACTION, n_rep=300)
    print(f"\n  check, first {N_FIRST} classes one by one (DKL [bit] at 20% of each class)")
    print("  class  pairs | " + " ".join(f"bw={b:<6g}" for b in cw.binwidths))
    for j in range(len(cw.class_index)):
        flag = "  <- fewest pairs" if cw.class_index[j] == k else ""
        print(f"  {cw.class_index[j]:5d} {cw.n_values[j]:6d} | " + " ".join(f"{v:9.3f}" for v in cw.dkl_at_target[j]) + flag)
    for tol in (0.1, 0.2):
        print(f"  tolerance {tol:g}: limiting class alone -> {cw.recommend(tol, [k])}; "
              f"all {N_FIRST} classes -> {cw.recommend(tol)}")
    fig = plots.plot_binwidth_classwise(cw, dkl_tol=0.1, title=name)
    fig.savefig(out / f"binwidth_classwise_{tag}.png", dpi=90)
    plt.close(fig)
print("\nfigures in", out)
