import numpy as np
import pytest
from conftest import JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.predict import Predictor, simulate
from pyher.weights import Weights


@pytest.fixture(scope="module")
def setup(golden):
    ds = load_jura_dataset(JURA_MAT)
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=9,
                    z_thresh=float(np.round(np.log10(50), 5)), objective="threshold", nugget=True)
    z = np.round(ds.z_cal, 5)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    r = golden("her2")["her2"]
    empty = np.empty(0)
    w = Weights(
        best_w_OR=np.atleast_1d(r.best_w_OR), best_w_AND=np.atleast_1d(r.best_w_AND),
        w_OR_slope=np.atleast_1d(r.w_OR_slope), w_AND_slope=np.atleast_1d(r.w_AND_slope),
        best_alpha=float(r.best_alpha), best_beta=float(r.best_beta), DKL_w_OR=0, DKL_w_AND=0,
        DKL_w_alpha_beta=0, n_neighbor_aggreg=9, idx_opt_nn=empty, n_pairs_by_class_obs=empty,
        nmean_pairs_by_class=empty, target_idx_zero_neigh=empty)
    return ds, z, ig, w


def test_prediction_matches_matlab(setup, golden):
    ds, z, ig, w = setup
    g = golden("her3_pred")
    pred = Predictor(ds.x_cal, ds.y_cal, z, ig, w)
    pmf, zero = pred.predict(np.ravel(g["gx"], order="F"), np.ravel(g["gy"], order="F"))
    np.testing.assert_allclose(pmf, g["pmf_pred_mat"], rtol=1e-9, atol=1e-14)
    np.testing.assert_array_equal(zero + 1, np.atleast_1d(g["idx_zero"]))


def test_simulation_replay_matches_matlab(setup, golden):
    """Replay MATLAB's random numbers (visiting order and Monte Carlo p-values)."""
    ds, z, ig, w = setup
    g = golden("hers4")
    real, rec = simulate((ds.x_cal, ds.y_cal, z, ig, w), np.ravel(g["sx"], order="F"),
                         np.ravel(g["sy"], order="F"), perm=np.atleast_1d(g["rec"].perm) - 1,
                         p_values=np.atleast_1d(g["rec"].p_value))
    np.testing.assert_allclose(real, np.atleast_1d(g["z_realiz"]), rtol=1e-9, atol=1e-12)


def test_simulation_is_reproducible_and_honours_data(setup):
    ds, z, ig, w = setup
    xt, yt = ds.x_cal[:5], ds.y_cal[:5]
    a, _ = simulate((ds.x_cal, ds.y_cal, z, ig, w), xt, yt, rng=1)
    b, _ = simulate((ds.x_cal, ds.y_cal, z, ig, w), xt, yt, rng=1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, z[:5])  # targets on calibration points reproduce the data
