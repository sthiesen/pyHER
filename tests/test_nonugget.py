"""NOnugget version (the one used in the paper): weights reached at the upper class edge, 1/distance
inside the first class and 1e16 at distance 0."""

import numpy as np
import pytest
from conftest import JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.predict import Predictor, simulate
from pyher.weights import Weights, _slopes


@pytest.fixture(scope="module")
def setup(golden):
    ds = load_jura_dataset(JURA_MAT)
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=30,
                    z_thresh=float(np.round(np.log10(50), 5)), nugget=False)
    z = np.round(ds.z_cal, 5)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    r = golden("her2")["her2"]
    g = golden("nonugget")
    w = Weights(
        best_w_OR=np.atleast_1d(r.best_w_OR), best_w_AND=np.atleast_1d(r.best_w_AND),
        w_OR_slope=np.atleast_1d(g["w_OR_slope"]), w_AND_slope=np.atleast_1d(g["w_AND_slope"]),
        best_alpha=0.65, best_beta=0.0, DKL_w_OR=0, DKL_w_AND=0, DKL_w_alpha_beta=0,
        n_neighbor_aggreg=30, idx_opt_nn=None, n_pairs_by_class_obs=None, nmean_pairs_by_class=None,
        target_idx_zero_neigh=None, nugget=ig.cfg.nugget)
    return ds, z, ig, w, g


def test_slopes_match_matlab(setup):
    _, _, ig, w, g = setup
    for best, key in ((w.best_w_OR, "w_OR_slope"), (w.best_w_AND, "w_AND_slope")):
        got = _slopes(best, ig.edges_distance_classes_range, nugget=False)
        want = np.atleast_1d(g[key])
        assert np.isnan(got[0]) and np.isnan(want[0])
        np.testing.assert_allclose(got[1:], want[1:], rtol=1e-12)


def test_prediction_matches_matlab(setup):
    ds, z, ig, w, g = setup
    pmf, zero = Predictor(ds.x_cal, ds.y_cal, z, ig, w).predict(g["tx"], g["ty"])
    np.testing.assert_allclose(pmf, g["pmf_mat"], rtol=1e-9, atol=1e-14)
    np.testing.assert_array_equal(zero + 1, np.atleast_1d(g["idx_zero"]))


def test_no_uncertainty_at_sampled_locations(setup):
    ds, z, ig, w, g = setup
    pmf, _ = Predictor(ds.x_cal, ds.y_cal, z, ig, w).predict(ds.x_cal[:5], ds.y_cal[:5])
    # the 1e16 weight puts (almost) all mass into one bin
    assert (pmf.max(axis=1) > 0.2).all()


def test_simulation_replay_matches_matlab(setup):
    ds, z, ig, w, g = setup
    real, _ = simulate((ds.x_cal, ds.y_cal, z, ig, w), g["sx"], g["sy"],
                       perm=np.atleast_1d(g["rec"].perm) - 1,
                       p_values=np.atleast_1d(g["rec"].p_value))
    np.testing.assert_allclose(real, np.atleast_1d(g["z_realiz"]), rtol=1e-9, atol=1e-12)


def test_paper_infogram_numbers(setup):
    """Paper, Sect. 3.4 / Fig. 14: 20 classes up to 1.4 km; 20 294 of 66 822 pairs inside the range."""
    _, _, ig, _, _ = setup
    assert ig.n_classes_range - 1 == 20
    assert ig.edges_distance_classes_range[-2] == pytest.approx(1.4)
    assert ig.n_pairs_by_class.sum() == 66822
    assert ig.n_pairs_by_class[: ig.n_classes_range - 1].sum() == 20294
