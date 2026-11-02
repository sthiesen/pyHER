"""Matplotlib versions of the ``f_plot_*`` functions and of the figures in
``sHER_E_0304_Extract_pmf_statistics_performance_and_plot.m``.

Requires the ``plot`` extra (``matplotlib``). Every function returns the created figure(s) and never
calls ``plt.show()``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.path import Path

from pyher.infogram import Infogram
from pyher.io import Dataset
from pyher.stats import ValidationMetrics
from pyher.weights import Weights

_CLIMS_Z = (1.2, 2.5)  # colour limits of the Jura log10(Pb) maps in the MATLAB code


def inside_basin(gx, gy, basin_parts) -> np.ndarray:
    """Mask of grid nodes inside (or on the border of) the basin polygon (MATLAB ``inpolygon``)."""
    pts = np.column_stack((np.ravel(gx), np.ravel(gy)))
    mask = np.zeros(len(pts), dtype=bool)
    for part in basin_parts:
        mask |= Path(part).contains_points(pts, radius=1e-9) | Path(part).contains_points(pts, radius=-1e-9)
    return mask.reshape(np.shape(gx))


def _outline(ax, basin_parts):
    for part in basin_parts or []:
        ax.plot(part[:, 0], part[:, 1], color="k", lw=2)


def _clip(values, gx, gy, basin_parts):
    out = np.array(values, dtype=float, copy=True)
    if basin_parts:
        out[~inside_basin(gx, gy, basin_parts)] = np.nan
    return out


def _data_markers(ax, ds: Dataset, size=50):
    z = ds.z[ds.idx_cal]
    span = np.ptp(z)
    ax.scatter(ds.x[ds.idx_cal], ds.y[ds.idx_cal], s=1 + 70 * (z - z.min()) / (span or 1),
               c="k", marker="+", lw=1)
    ax.scatter(ds.x[ds.idx_val], ds.y[ds.idx_val], s=size, facecolors="none", edgecolors="r")


# ----------------------------------------------------------------------------------------------
# spatial characterisation / weights
# ----------------------------------------------------------------------------------------------
def plot_infogram(ig: Infogram, txt: str = ""):
    """Infogram cloud, infogram, number of pairs, delta-z PMFs per class and normalised
    infogram + variogram (``f_plot_infogram.m``). Returns a list of figures."""
    figs = []
    n_cal = ig.mat_euc_distance_xy.shape[0]
    edges = ig.edges_distance_classes

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ig.mat_euc_distance_xy.ravel(), ig.mat_diff_z.ravel(), "xb", ms=3)
    for e in edges[1:]:
        ax.axvline(e, color="r", ls="--", lw=0.5)
    ax.set(xlabel="Euclidean distance", ylabel=r"$\Delta z$",
           title=f"Infogram cloud / cal. set: {n_cal} obs.")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ig.bin_centers_distance_classes, ig.H_diff_z_by_class, ".r", ms=12,
            label=r"Entropy of $\Delta z$ PMF by class")
    ax.hlines(ig.H_diff_z, 0, ig.mat_euc_distance_xy.max(), lw=2, label=r"Entropy of the full-set $\Delta z$ PMF")
    for e in edges[1:]:
        ax.axvline(e, color="k", ls="--", lw=0.5)
    ax.set(xlabel="Euclidean distance", ylabel="Entropy [bit]", title=f"Infogram / {txt}")
    ax.legend(loc="lower left")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    classes = np.arange(1, ig.n_lag + 1)
    colors = ["#0000e6" if c < ig.n_classes_range else "#808080" for c in classes]
    ax.bar(classes, ig.n_pairs_by_class, color=colors)
    ax.set(xlabel="Distance class", ylabel="Number of pairs",
           title=r"Infogram & number of pairs (in $\Delta z$ histogram) by class")
    ax2 = ax.twinx()
    ax2.plot(classes, ig.H_diff_z_by_class, ".r", ms=10)
    ax2.set_ylabel("Entropy [bit]")
    in_range = ig.n_pairs_by_class[: ig.n_classes_range - 1].sum()
    total = ig.n_pairs_by_class.sum()
    ax.text(0.55, 0.62, f"# observations: {n_cal} (cal. set)\n# pairs: {total}\n"
            f"# pairs into the range: {in_range} ({100 * in_range / total:.1f}%)",
            transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", ec="0.7", alpha=0.9))
    figs.append(fig)

    R = ig.n_classes_range
    ncols = 5
    nrows = int(np.ceil(R / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows), squeeze=False)
    ymax = ig.pmf_diff_z_by_class_obs_range.max() * 1.02
    for i, ax in enumerate(axes.ravel()):
        if i >= R:
            ax.axis("off")
            continue
        ax.bar(ig.bin_centers_edges_diff_z, ig.pmf_diff_z_by_class_obs_range[i], width=ig.cfg.binwidth)
        full = i == R - 1
        lo, hi = ig.edges_distance_classes_range[i], ig.edges_distance_classes_range[i + 1]
        ax.set_title("full dataset PMF" if full else f"class {i + 1} / dist.: ({lo:.2f}, {hi:.2f}]",
                     fontsize=8)
        ax.set(xlim=(-1, 1), ylim=(0, ymax))
    fig.suptitle(r"$\Delta z$ PMF by class")
    fig.tight_layout()
    figs.append(fig)

    # experimental variogram next to the normalised infogram
    gamma = np.full(ig.n_lag, np.nan)
    sq = ig.mat_diff_z ** 2
    for i in range(ig.n_lag):
        m = (ig.mat_euc_distance_xy > edges[i]) & (ig.mat_euc_distance_xy <= edges[i + 1])
        if m.any():
            gamma[i] = 0.5 * sq[m].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ig.bin_centers_distance_classes, ig.H_diff_z_by_class / np.nanmax(ig.H_diff_z_by_class),
            ".-", color="#000066", label="normalised entropy")
    ax.plot(ig.bin_centers_distance_classes, gamma / np.nanmax(gamma), ".-", color="#b30000",
            label=r"normalised $\frac{1}{2}\mathrm{mean}(\Delta z^2)$")
    for e in edges[1:]:
        ax.axvline(e, color="k", ls="--", lw=0.4)
    ax.set(xlim=(0, ig.bin_centers_distance_classes[-2]), ylim=(0, 1), xlabel="lag",
           title="Normalised infogram and experimental variogram")
    ax.legend(loc="lower right")
    figs.append(fig)
    return figs


def plot_weights(w: Weights, ig: Infogram):
    """Optimum OR / AND class weights and the alpha / beta mixing weights (``f_plot_weights.m``)."""
    left = ig.edges_distance_classes_range[:-1]
    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.plot(left, w.best_w_OR, ".-", ms=10, lw=2, label=f"OR weights (DKL_OR: {w.DKL_w_OR:.2f})")
    ax.plot(left, w.best_w_AND, ".-", ms=10, lw=2, label=f"AND weights (DKL_AND: {w.DKL_w_AND:.2f})")
    for e in left:
        ax.axvline(e, color="k", ls="--", lw=0.5)
    ax.set(ylim=(0, 1), xlabel="Euclidean distance", ylabel="Optimum weights",
           title="Optimum weights by class")
    ax.legend()

    fig2, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Beta", "Alpha"], [w.best_beta, w.best_alpha], color=["C0", "#800080"])
    ax.set(ylim=(0, 1.2), ylabel="Optimum weights",
           title=f"AND/OR mix | DKL_αβ: {w.DKL_w_alpha_beta:.2f}")
    return [fig1, fig2]


# ----------------------------------------------------------------------------------------------
# prediction maps
# ----------------------------------------------------------------------------------------------
def plot_prediction(z_mean, z_entropy, pmfs, gx, gy, ds: Dataset, ig: Infogram, x_pmf, y_pmf,
                    basin_parts=None, clims=_CLIMS_Z, clims_h=None, txt=""):
    """Original data, E-type (mean) map and entropy map, plus the PMFs at selected locations
    (``f_plot_prediction.m``). ``pmfs`` is ``[ny, nx, n_bins_z]`` (or ``[T, n_bins_z]`` in the
    column-major order of the grid); ``z_mean`` and ``z_entropy`` are ``[ny, nx]``.
    The MATLAB code fixes the entropy colour range to [0.5, 5] bit, which does not fit every choice
    of binwidths, so by default the range of the data inside the basin is used (``clims_h`` overrides)."""
    pmfs = np.asarray(pmfs)
    if pmfs.ndim == 2:
        pmfs = pmfs.reshape(gx.shape[1], gx.shape[0], -1).transpose(1, 0, 2)
    x_pmf, y_pmf = np.atleast_1d(x_pmf), np.atleast_1d(y_pmf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(ds.x[ds.idx_cal], ds.y[ds.idx_cal], s=50, c=ds.z[ds.idx_cal], cmap="viridis",
               vmin=clims[0], vmax=clims[1], edgecolors="k")
    sc = ax.scatter(ds.x[ds.idx_val], ds.y[ds.idx_val], s=50, c=ds.z[ds.idx_val], cmap="viridis",
                    vmin=clims[0], vmax=clims[1], edgecolors="r")
    ax.set_title("Original")
    fig.colorbar(sc, ax=ax, shrink=0.7)

    h_vals = _clip(z_entropy, gx, gy, basin_parts)
    if clims_h is None:
        clims_h = (np.nanmin(h_vals), np.nanmax(h_vals))
    for ax, field, title, cmap, norm in (
        (axes[1], z_mean, "HER E-type", "viridis", None),
        (axes[2], z_entropy, "Entropy map [bits]", "spring", LogNorm(*clims_h)),
    ):
        vals = _clip(field, gx, gy, basin_parts)
        kw = dict(norm=norm) if norm else dict(vmin=clims[0], vmax=clims[1])
        mesh = ax.pcolormesh(gx, gy, vals, cmap=cmap, shading="auto", **kw)
        _data_markers(ax, ds)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    for ax in axes:
        ax.scatter(x_pmf, y_pmf, s=100, marker="s", facecolors="none", edgecolors="0.8", lw=1.5)
        _outline(ax, basin_parts)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(f"Maps | Dataset: {txt}")
    fig.tight_layout()

    ncols = 2
    nrows = int(np.ceil(len(x_pmf) / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    xs, ys = gx[0, :], gy[:, 0]
    from pyher.core import entropy
    for ax, xp, yp in zip(axes2.ravel(), x_pmf, y_pmf):
        j, i = int(np.argmin(np.abs(xs - xp))), int(np.argmin(np.abs(ys - yp)))
        ax.bar(ig.bin_centers_edges_z, pmfs[i, j], width=ig.cfg.binwidth_z, color="k")
        ax.set(xlim=(-10, 10), ylim=(0, 0.2), xlabel="z",
               title=f"x={xs[j]:.2f}; y={ys[i]:.2f}\nH = {entropy(pmfs[i, j]):.2f} bits")
    for ax in axes2.ravel()[len(x_pmf):]:
        ax.axis("off")
    fig2.suptitle(f"Predicted z PMF / {txt}")
    fig2.tight_layout()
    return [fig, fig2]


def plot_probability_map(z_prob, z_thresh, gx, gy, ds: Dataset, basin_parts=None, txt=""):
    """Map of P(z > z_thresh) with the data classified by the threshold
    (``f_plot_probabilitymap.m``)."""
    # grey classes of the MATLAB colormap: limits 0,15,30,45,60,75,100 (% probability)
    limits = [0, 15, 30, 45, 60, 75, 100]
    greys = [0.9, 0.75, 0.6, 0.4, 0.3, 0.1]
    cmap = ListedColormap(np.vstack([[[g, g, g]] * (limits[i + 1] - limits[i] + 1)
                                     for i, g in enumerate(greys)]))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    mesh = ax.pcolormesh(gx, gy, _clip(z_prob, gx, gy, basin_parts), cmap=cmap, vmin=0, vmax=1,
                         shading="auto")
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    for name, idx, size, marker in (("cal. set", ds.idx_cal, 60, "o"), ("val. set", ds.idx_val, 35, "s")):
        hi = ds.z[idx] > z_thresh
        ax.scatter(ds.x[idx][hi], ds.y[idx][hi], s=size, marker=marker, facecolors="none",
                   edgecolors="r", label=f"{name} > thresh.")
        ax.scatter(ds.x[idx][~hi], ds.y[idx][~hi], s=size, marker=marker, facecolors="none",
                   edgecolors="k", label=f"{name} <= thresh.")
    _outline(ax, basin_parts)
    ax.set(aspect="equal", xlabel="x", ylabel="y",
           title=f"Probability map [Probability Z>{z_thresh:g}] / {txt}")
    ax.legend(loc="upper left", fontsize=7)
    return [fig]


# ----------------------------------------------------------------------------------------------
# performance
# ----------------------------------------------------------------------------------------------
def plot_performance(val: ValidationMetrics, z_val, z_pred, txt=""):
    """True-vs-predicted and residual scatter, ROC curve, contamination and misclassification
    versus the probability threshold (figures of the MATLAB performance block)."""
    z_val, z_pred = np.asarray(z_val), np.asarray(z_pred)
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(z_val, z_pred, s=12)
    axes[0].set(xlabel="True Z", ylabel="Expected Z", xlim=(-2, 2), ylim=(-2, 2))
    axes[1].scatter(z_val, z_pred - z_val, s=12)
    axes[1].set(xlabel="True Z", ylabel="Residue", xlim=(-3, 3), ylim=(-3, 3))
    fig1.suptitle("Correlation")

    fig2, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(val.roc["FPR"], val.roc["TPR"], "o-", ms=3)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curve")

    fig3, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(val.prob_thres, val.contamination, "o-", ms=3)
    axes[0].set(xlabel="Probability p", ylabel="Locations declared contaminated [%]",
                title="Proportion of locations declared contaminated")
    axes[1].plot(val.prob_thres, val.misclassication, "o-", ms=3)
    axes[1].axvline(val.marginal_prob_thres_cal, color="k", ls="--")
    axes[1].text(val.marginal_prob_thres_cal, 50, "Marginal prob. →", ha="right")
    axes[1].set(xlabel="Probability p", ylabel="Misclassification [%]",
                title="Proportion of locations wrongly classified")
    fig3.suptitle(f"Probability threshold p / {txt}")
    return [fig1, fig2, fig3]


def plot_realization(z_realiz, gx, gy, ds: Dataset, basin_parts=None, clims=_CLIMS_Z, title=""):
    """One simulated realisation (log colour scale like the MATLAB figure)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    mesh = ax.pcolormesh(gx, gy, _clip(z_realiz, gx, gy, basin_parts), cmap="viridis",
                         vmin=clims[0], vmax=clims[1], shading="auto")
    z = ds.z[ds.idx_cal]
    ax.scatter(ds.x[ds.idx_cal], ds.y[ds.idx_cal], s=1 + 70 * (z - z.min()) / np.ptp(z), c="k",
               marker="+", lw=1)
    _outline(ax, basin_parts)
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    ax.set(aspect="equal", xlabel="x", ylabel="y", title=title)
    return fig


# ----------------------------------------------------------------------------------------------
# paper-style figures (Thiesen & Ehret 2021, Fig. 17 and weights figure)
# ----------------------------------------------------------------------------------------------
def plot_weights_paper(ig: Infogram, sets: dict, xmax=1.4, ymax=0.5, at="right"):
    """Class weights against distance like the paper's weight figure: OR (blue) and AND (orange)
    weights of every distance class, dashed lines at the class edges, y-range 0-0.5.

    ``sets`` maps a panel title to ``(w_or, w_and)``. Points are placed at the right edge of their
    class (``at="right"``, the paper's positions) or at the left edge (``at="left"``, what
    ``f_plot_weights.m`` does).
    """
    edges = ig.edges_distance_classes_range
    x = edges[1:] if at == "right" else edges[:-1]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.4), squeeze=False)
    for ax, (title, (w_or, w_and)) in zip(axes[0], sets.items()):
        for e in edges[:-1]:
            ax.axvline(e, color="0.3", ls="--", lw=0.7)
        ax.plot(x, w_or, "o-", color="#1f5fbf", label=r"$w_{OR}$")
        ax.plot(x, w_and, "o-", color="#c8602a", label=r"$w_{AND}$")
        ax.set(xlim=(0, xmax), ylim=(0, ymax), xlabel="Euclidean distance (km)",
               ylabel="Optimum weights", title=title)
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


_EST_LEVELS = [1.3, 1.4, 1.5, 1.6, 1.699, 1.8, 1.9, 2.1, 2.4]
_EST_COLORS = ["#2b3990", "#3f4fe0", "#5b8def", "#6fb4e8", "#52b788", "#a3b48a", "#e8a95a", "#f4e04d"]
_ENT_LEVELS = [1.2, 2, 3, 4, 5, 5.348, 6.5, 7.5, 9]
_ENT_COLORS = ["#e236f5", "#da4fd0", "#dc5aa8", "#e5708f", "#ea8d85", "#e9b062", "#f2d65a", "#fbf24b"]
_PROB_LEVELS = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
_PROB_COLORS = ["#e6e6e6", "#bdbdbd", "#999999", "#6b6b6b", "#444444", "#161616"]


def _banded(ax, fig, X, Y, vals, levels, colors, ticklabels=None, label=None):
    from matplotlib.colors import BoundaryNorm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    mesh = ax.pcolormesh(X, Y, np.clip(vals, levels[0], levels[-1] - 1e-9), cmap=cmap, norm=norm,
                         shading="auto")
    # evenly spaced bands in the colour bar, labelled with the band limits (as in the paper)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.85, spacing="uniform", ticks=levels)
    cb.ax.set_yticklabels(ticklabels or [f"{v:g}" for v in levels])
    if label:
        cb.set_label(label)
    return cb


def plot_paper_maps(gx, gy, z_mean, z_entropy, z_prob, z_cal_x, z_cal_y, z_cal, z_thresh, H_marginal,
                    basin_parts=None, label="HER"):
    """Four-panel figure like Fig. 17 of the paper: (a) estimate, (b) local entropy, (c) probability of
    exceeding ``z_thresh``, (d) classification into contaminated / safe using the marginal
    probability of contamination as decision threshold.

    All maps are ``[ny, nx]`` arrays on the grid ``gx, gy``; ``H_marginal`` is the entropy of the
    calibration-set PMF (the 5.348 bit break of the entropy colour bar).
    """
    z_cal = np.asarray(z_cal)
    marginal = float(np.mean(z_cal > z_thresh))
    levels_est = list(_EST_LEVELS)
    levels_est[4] = round(z_thresh, 3)
    levels_ent = list(_ENT_LEVELS)
    levels_ent[5] = round(H_marginal, 3)
    maps = [
        ("(a) %s | Estimate map" % label, _clip(z_mean, gx, gy, basin_parts)),
        ("(b) %s | Entropy map" % label, _clip(z_entropy, gx, gy, basin_parts)),
        ("(c) %s | Probability map ($z_c$)" % label, _clip(z_prob, gx, gy, basin_parts)),
        ("(d) %s | Classification map" % label, _clip(z_prob, gx, gy, basin_parts)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.5))
    for ax, (title, vals) in zip(axes.ravel(), maps):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    cb = _banded(axes[0, 0], fig, gx, gy, maps[0][1], levels_est, _EST_COLORS,
                 [f"{v:g}" if i != 4 else f"{v:g}" for i, v in enumerate(levels_est)], r"$\log_{10}$(Pb)")
    cb.ax.axhline(4, color="k", lw=0.6)
    _banded(axes[0, 1], fig, gx, gy, maps[1][1], levels_ent, _ENT_COLORS, label="bits")
    _banded(axes[1, 0], fig, gx, gy, maps[2][1], _PROB_LEVELS, _PROB_COLORS, label="Probability")
    # classification
    cls = np.where(np.isnan(maps[3][1]), np.nan, (maps[3][1] > marginal).astype(float))
    cmap = ListedColormap(["#b8cbe6", "#5d6a85"])
    mesh = axes[1, 1].pcolormesh(gx, gy, cls, cmap=cmap, vmin=-0.5, vmax=1.5, shading="auto")
    cb = fig.colorbar(mesh, ax=axes[1, 1], shrink=0.85, ticks=[0, 1])
    cb.ax.set_yticklabels(["safe", "contaminated"], rotation=90, va="center")
    for ax in axes.ravel():
        hi = z_cal > z_thresh
        ax.scatter(z_cal_x[~hi], z_cal_y[~hi], s=22, facecolors="none", edgecolors="k", lw=0.8)
        ax.scatter(z_cal_x[hi], z_cal_y[hi], s=22, facecolors="none", edgecolors="r", lw=0.8)
        _outline(ax, basin_parts)
        ax.set(aspect="equal", xlim=(0, 6), ylim=(0, 6), xlabel="x (km)", ylabel="y (km)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec="k", label=r"$\log_{10}$Pb $\leq z_c$"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="r", label=r"$\log_{10}$Pb $> z_c$")],
        title=f"calibration set ({len(z_cal)} obs.)", loc="lower center", ncol=2, frameon=True,
        bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.085, f"$z_c$ = {z_thresh:.3f};  marginal probability of contamination = "
             f"{marginal:.3f}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def plot_ergodic_fluctuations(data, fields, *, lag_dist, binwidth, max_points=1500, labels=None, seed=0):
    """Ergodic-fluctuation check: CDF and infogram of simulated fields against the data
    (the commented block at the end of ``sHER_E_0304_...m``).

    ``data = (x, y, z)`` of the calibration set (red). ``fields`` is a list of ``(x, y, z)`` arrays of
    realisations (grey) followed optionally by the E-type map as the last element when
    ``labels = {"etype": True}``. Returns the CDF figure and the infogram figure.
    """
    from pyher.infogram import infogram_entropy

    etype = (labels or {}).get("etype", False)
    sims, et = (fields[:-1], fields[-1]) if etype else (fields, None)
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))

    def cdf(ax, z, **kw):
        z = np.sort(np.asarray(z)[np.isfinite(z)])
        ax.plot(z, np.arange(1, len(z) + 1) / len(z), **kw)

    for i, (x, y, z) in enumerate(sims):
        cdf(ax1, z, color="0.6", lw=0.4, label="realisations" if i == 0 else None)
        c, h, _ = infogram_entropy(x, y, z, lag_dist, binwidth, max_points=max_points, seed=seed)
        ax2.plot(c, h, color="0.6", lw=0.4, label="realisations" if i == 0 else None)
    if et is not None:
        cdf(ax1, et[2], color="b", lw=1, label="E-type")
        c, h, _ = infogram_entropy(*et, lag_dist, binwidth, max_points=max_points, seed=seed)
        ax2.plot(c, h, color="b", lw=1.5, label="E-type")
    cdf(ax1, data[2], color="r", lw=2, label="calibration set")
    c, h, _ = infogram_entropy(*data, lag_dist, binwidth)
    ax2.plot(c, h, color="r", lw=2, label="calibration set")
    ax1.set(xlabel="z", ylabel="probability", title="CDF")
    ax2.set(xlabel="Euclidean distance", ylabel="Entropy [bit]", title="Infogram")
    ax1.legend()
    ax2.legend()
    return fig1, fig2


def plot_binwidth_selection(res, n_target=None, dkl_tol=0.1, title=""):
    """Cross-entropy convergence for several bin widths (``pyher.binwidth``): the excess of the cross entropy over
    the entropy of the full set, DKL(p || q), against the sample size (left) and against the bin width at
    ``n_target`` with the tolerance of ``BinwidthResult.recommend`` (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("viridis")
    dkl = res.dkl_mean
    shown = np.where(dkl > 1e-6, dkl, np.nan)  # the full data set has zero excess by construction (log axis)
    for i, bw in enumerate(res.binwidths):
        ax1.plot(res.sample_sizes, shown[i], "o-", ms=3, color=cmap(i / max(1, len(res.binwidths) - 1)),
                 label=f"bw = {bw:g} ({res.n_bins[i]} bins)")
    ax1.axhline(dkl_tol, color="k", ls=":", lw=1)
    ax1.set(xscale="log", yscale="log", xlabel="sample size n", ylabel="DKL(p || q) = Hpq - H(p) [bit]",
            title=f"{title} cross-entropy excess")
    ax1.legend(fontsize=7)
    j = 0 if n_target is None else int(np.argmin(np.abs(res.sample_sizes - n_target)))
    ax2.plot(res.binwidths, dkl[:, j], "o-")
    ax2.axhline(dkl_tol, color="k", ls=":", lw=1, label=f"tolerance {dkl_tol:g} bit")
    rec = res.recommend(n_target, dkl_tol)
    if rec is not None:
        ax2.axvline(rec, color="r", ls="--", lw=1, label=f"smallest bin width within tolerance: {rec:g}")
    ax2.set(xscale="log", yscale="log", xlabel="bin width", ylabel=f"DKL at n = {res.sample_sizes[j]} [bit]",
            title=f"{res.n_data} data values, {res.n_rep} repetitions, {res.strategy} sampling")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_binwidth_classwise(cw, n_range=None, dkl_tol=0.1, title=""):
    """Class-by-class bin-width check (``pyher.binwidth.classwise_cross_entropy``): number of delta-z values per
    distance class (left) and the excess DKL(p || q) at ``cw.fraction`` of each class (right; one curve per bin
    width). ``n_range`` marks the last class inside the range."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    k = cw.class_index
    colors = ["#1f4fa0" if n_range is None or c <= n_range else "0.7" for c in k]
    ax1.bar(k, cw.n_values, color=colors)
    ax1.set(xlabel="distance class", ylabel="delta-z values (pairs)", yscale="log",
            title=f"{title} data per class" + ("" if n_range is None else " (blue: inside the range)"))
    cmap = plt.get_cmap("viridis")
    for i, bw in enumerate(cw.binwidths):
        shade = cmap(i / max(1, len(cw.binwidths) - 1))
        ax2.plot(k, cw.dkl_at_target[:, i], "o-", ms=3, color=shade, label=f"bw = {bw:g}")
    ax2.axhline(dkl_tol, color="k", ls=":", lw=1, label=f"tolerance {dkl_tol:g} bit")
    if n_range is not None:
        ax2.axvline(n_range + 0.5, color="r", ls="--", lw=1, label="end of the range")
    ax2.set(xlabel="distance class", yscale="log", title="cross-entropy excess by class",
            ylabel=f"DKL(p || q) at {int(cw.fraction * 100)}% of the class [bit]")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_binwidth_actual(res, dkl_tol=0.1, chosen=None, title=""):
    """Excess DKL at the actual class size against the bin width, one thin line per class and a thick one for the
    limiting class (``pyher.binwidth.actual_size_excess``). ``chosen`` marks the bin width used in a paper."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    others_labelled = False
    for j, k in enumerate(res.class_index):
        lim = k == res.limiting_class
        label = f"limiting class {k} ({res.n_values[j]} values)" if lim else None
        if not lim and not others_labelled:
            label, others_labelled = "other classes", True
        ax.plot(res.binwidths, res.excess[j], "o-" if lim else "-", ms=4, lw=3 if lim else 0.8,
                color="tab:red" if lim else "0.65", label=label)
    ax.axhline(dkl_tol, color="k", ls=":", lw=1, label=f"tolerance {dkl_tol:g} bit")
    rec = res.recommend(dkl_tol)
    if rec is not None:
        ax.axvline(rec, color="k", ls="--", lw=1, label=f"smallest bin width within tolerance: {rec:g}")
    if chosen is not None:
        ax.axvline(chosen, color="tab:blue", lw=4, alpha=0.3, label=f"bin width of the paper: {chosen:g}")
    ax.set(xscale="log", yscale="log", xlabel="bin width", ylabel="expected DKL(P || q) of the class PMF [bit]",
           title=f"{title} excess at the actual class size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
