import numpy as np
import pytest

from pyher import stats


@pytest.fixture(scope="module")
def g(golden):
    return golden("stats")


@pytest.fixture(scope="module")
def edges(golden):
    r = golden("her1")["her1"]
    return np.atleast_1d(r.edges_z), np.atleast_1d(r.bin_centers_edges_z)


@pytest.fixture(scope="module")
def grid_pmfs(golden):
    return golden("her3_pred")["pmf_pred_mat"]


THRESH = float(np.round(np.log10(50), 5))


def test_extract_pmf_statistics(g, edges, grid_pmfs):
    e, c = edges
    st = stats.extract_pmf_statistics(grid_pmfs, e, c, THRESH)
    np.testing.assert_allclose(st.entropy, g["z_entropy"], rtol=1e-12)
    np.testing.assert_allclose(st.mean, g["z_mean"], rtol=1e-12)
    np.testing.assert_allclose(st.mode, g["z_mode"], rtol=1e-12)
    np.testing.assert_allclose(st.prob_exceed, g["z_prob"], rtol=1e-10, atol=1e-14)


def test_no_threshold_gives_nan(edges, grid_pmfs):
    e, c = edges
    assert np.isnan(stats.extract_pmf_statistics(grid_pmfs[:3], e, c, None).prob_exceed).all()


@pytest.fixture(scope="module")
def val(g, edges):
    e, c = edges
    return stats.validate(g["pmf_val"], g["z_val"], g["z_cal"], e, c, THRESH)


DET = ["RMSE", "ME", "MAE", "NSE", "correl_true_pred", "correl_true_residue"]


@pytest.mark.parametrize("name", DET)
def test_deterministic(val, g, name):
    assert getattr(val.deterministic, name) == pytest.approx(getattr(g["perf"], name), rel=1e-10)


SCALARS = ["marginal_prob_thres_cal", "DKL_score_mean", "DKL_score_01_largersmaller_thres_mean",
           "DKL_score_1_larger_thres_mean", "G", "ksi_fraction_uq_lq", "misclass_mean",
           "misclass_min", "misclass_marginal", "Accuracy_marginal", "F1_score_marginal",
           "MCC_marginal", "DistanceROC_marginal"]


@pytest.mark.parametrize("name", SCALARS)
def test_scalar_metrics(val, g, name):
    assert getattr(val, name) == pytest.approx(getattr(g["perf"], name), rel=1e-10, nan_ok=True)


def test_index_marg(val, g):
    assert val.index_marg + 1 == g["perf"].index_marg


@pytest.mark.parametrize("name", ["TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "Accuracy",
                                  "F1_score", "MCC", "DistanceROC"])
def test_roc_curves(val, g, name):
    np.testing.assert_allclose(val.roc[name], np.ravel(getattr(g["perf"], name)), rtol=1e-12,
                               atol=1e-14)


@pytest.mark.parametrize("name", ["misclassication", "contamination", "prob_thres", "ksi_fraction",
                                  "edges_interv", "interv_size", "DKL_score_01_largersmaller_thres",
                                  "DKL_score_1_larger_thres"])
def test_vectors(val, g, name):
    want = np.asarray(getattr(g["perf"], name))
    got = np.asarray(getattr(val, name))
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-10, atol=1e-14)


def test_f_perf_agrees_with_script_block(g):
    """The original f_perf (non-HER helper) must give the same metrics as the script block; this
    guards the assumption that the two code paths are interchangeable."""
    a, b = g["perf_f"], g["perf"]
    for name in ["RMSE", "ME", "MAE", "NSE", "DKL_score_largersmaller_thres_mean",
                 "misclass_marginal"]:
        other = {"DKL_score_largersmaller_thres_mean": "DKL_score_01_largersmaller_thres_mean"}
        assert getattr(a, name) == pytest.approx(getattr(b, other.get(name, name)), rel=1e-10)


def test_pmf_from_realizations():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    out = stats.pmf_from_realizations(np.array([[0.5, 0.7, 2.5, 5.0]]), edges)
    np.testing.assert_allclose(out, [[2 / 3, 0, 1 / 3]])
