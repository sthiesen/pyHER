"""Reproduction of numbers published in Thiesen et al. (2020, HESS) and its supplement, using the
synthetic data sets of https://github.com/KIT-HYD/HER/tree/master/datasets (download them to
``<repo>/HER_datasets``; the tests are skipped when the folder is missing)."""

import dataclasses
import warnings

import numpy as np
import pytest
from conftest import REPO

from pyher import HerConfig, load_her_synthetic
from pyher.core import entropy, performance_prob
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.stats import extract_pmf_statistics, performance_det
from pyher.weights import optimize_weights

DATA = REPO / "HER_datasets"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="HER synthetic data sets not downloaded")


def _cfg():
    # parameters of the paper: class width 2, bin width 0.2, 12 neighbours, single-bin DKL, HESS code version
    return HerConfig(lag_dist=2.0, binwidth=0.2, binwidth_z=0.2, n_neighbor=12, objective="bin", nugget=True)


@pytest.mark.parametrize("field,n,stats", [
    # (mean, sd, entropy, max, median, min) of the learning sets, Supplement S1
    ("SR0", 200, (-0.57, 1.05, 4.27, 1.76, -0.42, -3.68)),
    ("SR0", 600, (-0.58, 1.02, 4.34, 1.92, -0.51, -3.68)),
    ("LR1", 600, (-0.99, 1.01, 4.34, 1.87, -0.97, -3.65)),
    ("LR1", 2000, (-1.01, 1.03, 4.39, 2.00, -0.98, -3.95)),
])
def test_learning_set_statistics_match_supplement(field, n, stats):
    d = load_her_synthetic(DATA / f"{field}_input_randomfield_cal{n}.mat")
    ig = fit_infogram(d.x_cal, d.y_cal, np.round(d.z_cal, 5), _cfg())
    h = np.histogram(d.z_cal, bins=ig.edges_z)[0]
    got = (d.z_cal.mean(), d.z_cal.std(ddof=1), entropy(h / h.sum()), d.z_cal.max(), np.median(d.z_cal), d.z_cal.min())
    np.testing.assert_allclose(got, stats, atol=0.011)


@pytest.fixture(scope="module")
def lr1_600():
    d = load_her_synthetic(DATA / "LR1_input_randomfield_cal600.mat")
    z = np.round(d.z_cal, 5)
    ig = fit_infogram(d.x_cal, d.y_cal, z, _cfg())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(d.x_cal, d.y_cal, z, ig, or_shrinkage=None)
    return d, z, ig, w


def test_lr1_600_spatial_characterisation(lr1_600):
    """Paper Sect. 4.1: range at 44 grid units and 142 512 of 359 400 pairs inside it."""
    _, _, ig, _ = lr1_600
    assert ig.edges_distance_classes_range[-2] == 44
    assert ig.n_pairs_by_class.sum() == 359400
    assert ig.n_pairs_by_class[: ig.n_classes_range - 1].sum() == 142512


def test_lr1_600_alpha(lr1_600):
    assert lr1_600[3].best_alpha == pytest.approx(1.0)  # paper: alpha = 1 (beta = 0.95, a flat optimum)


def test_lr1_600_pure_and_model_reproduces_table_1(lr1_600):
    """Model 2 of Table 1 (pure AND): EMA 0.43, ENS 0.72, DKL 3.58, median entropy 3.75, targets A-D."""
    d, z, ig, w = lr1_600
    P = Predictor(d.x_cal, d.y_cal, z, ig, dataclasses.replace(w, best_alpha=1.0, best_beta=0.0))
    pm, _ = P.predict(d.x_test, d.y_test)
    st = extract_pmf_statistics(pm, ig.edges_z, ig.bin_centers_edges_z, None)
    det = performance_det(st.mean, np.round(d.z_test, 5))
    assert det.MAE == pytest.approx(0.43, abs=0.006)
    assert det.NSE == pytest.approx(0.72, abs=0.006)
    assert performance_prob(np.round(d.z_test, 5), pm, ig.edges_z) == pytest.approx(3.58, abs=0.03)
    assert np.median(st.entropy) == pytest.approx(3.75, abs=0.01)
    pa, _ = P.predict(np.array([10.0, 25, 47, 49]), np.array([42.0, 63, 16, 73]))
    sa = extract_pmf_statistics(pa, ig.edges_z, ig.bin_centers_edges_z, None)
    np.testing.assert_allclose(sa.mean, [-2.06, -1.33, -1.00, 0.66], atol=0.02)
    assert sa.entropy[1] == pytest.approx(3.52, abs=0.02)  # target B
    # targets C and D coincide with calibration points but keep a non-zero uncertainty (HESS version)
    assert (sa.entropy[2:] > 3).all()


@pytest.mark.slow
def test_and_optimisation_not_stuck_at_start_lr1_1500():
    """The 1/distance start is so sharp here that the loss is flat and a single SLSQP run stopped at
    11.8 bit (alpha came out 0.45); with several starts the AND loss is ~3.4 bit and alpha = 1 as in the paper."""
    d = load_her_synthetic(DATA / "LR1_input_randomfield_cal1500.mat")
    z = np.round(d.z_cal, 5)
    ig = fit_infogram(d.x_cal, d.y_cal, z, _cfg())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(d.x_cal, d.y_cal, z, ig, or_shrinkage=None)
    assert w.DKL_w_AND < 4.0
    assert w.best_alpha == pytest.approx(1.0)


@pytest.mark.slow
def test_hers_realisations_honour_data_and_follow_the_prediction():
    """Sequential simulation on a sub-grid of LR1-600: data honoured, marginal moments close to the true
    field, and the ensemble mean close to the analytical E-type (reproduction of the ergodic checks of
    examples/hers_hess2020.py, with few realisations)."""
    from pyher.predict import simulate

    d = load_her_synthetic(DATA / "LR1_input_randomfield_cal600.mat")
    z = np.round(d.z_cal, 5)
    cfg = dataclasses.replace(_cfg(), nugget=False)
    ig = fit_infogram(d.x_cal, d.y_cal, z, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(d.x_cal, d.y_cal, z, ig)
        keep = np.flatnonzero((d.x % 2 == 0) & (d.y % 2 == 0))
        keep = np.union1d(keep, d.idx_cal[:50])  # also some calibration nodes
        xt, yt = d.x[keep], d.y[keep]
        rng = np.random.default_rng(3)
        sims = np.array([simulate((d.x_cal, d.y_cal, z, ig, w), xt, yt, rng=rng)[0] for _ in range(6)])
        mean_pmf, _ = Predictor(d.x_cal, d.y_cal, z, ig, w).predict(xt, yt)
    on_data = np.isin(keep, d.idx_cal)
    np.testing.assert_allclose(sims[:, on_data], np.tile(d.z[keep][on_data], (6, 1)), atol=1e-4)
    etype = extract_pmf_statistics(mean_pmf, ig.edges_z, ig.bin_centers_edges_z, None).mean
    assert np.corrcoef(sims.mean(axis=0), etype)[0, 1] > 0.9
    assert abs(sims.mean() - d.z.mean()) < 0.15
    assert 0.85 < sims.std(axis=1).mean() / d.z.std() < 1.1
