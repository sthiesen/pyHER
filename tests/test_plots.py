"""Smoke tests: every figure function runs on the real example and the basin mask is sane."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import JURA_MAT, JURA_SHP

from pyher import HerConfig, load_jura_basin, load_jura_dataset, plots
from pyher.infogram import fit_infogram
from pyher.stats import validate
from pyher.weights import optimize_weights


@pytest.fixture(scope="module")
def run():
    ds = load_jura_dataset(JURA_MAT)
    cfg = HerConfig(n_neighbor=9, z_thresh=float(np.round(np.log10(50), 5)))
    z = np.round(ds.z_cal, 5)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig)
    return ds, cfg, ig, w


def test_basin_mask():
    basin = load_jura_basin(JURA_SHP)
    gx, gy = np.meshgrid(np.arange(0, 6.01, 0.5), np.arange(0, 6.01, 0.5))
    inside = plots.inside_basin(gx, gy, basin)
    assert inside.any() and not inside.all()
    assert not inside[0, 0]  # corner (0, 0) is outside the Jura basin


def test_figures(run):
    ds, cfg, ig, w = run
    figs = plots.plot_infogram(ig, ds.txt) + plots.plot_weights(w, ig)
    # performance figures from synthetic PMFs
    pm = np.full((20, ig.n_bins_z), 1 / ig.n_bins_z)
    val = validate(pm, ds.z_val[:20], ds.z_cal, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
    figs += plots.plot_performance(val, ds.z_val[:20], np.zeros(20), ds.txt)
    assert len(figs) == 10 and all(f.axes for f in figs)
    plt.close("all")


def test_ergodic_fluctuation_figures():
    rng = np.random.default_rng(0)
    x, y = rng.random(200) * 10, rng.random(200) * 10
    data = (x[:80], y[:80], rng.normal(size=80))
    sims = [(x, y, rng.normal(size=200)) for _ in range(3)]
    f1, f2 = plots.plot_ergodic_fluctuations(data, sims + [(x, y, np.zeros(200) + 0.01 * rng.normal(size=200))],
                                             lag_dist=1.0, binwidth=0.3, max_points=100, labels={"etype": True})
    assert f1.axes and f2.axes
    plt.close("all")
