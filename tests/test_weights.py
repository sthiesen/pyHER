import numpy as np
import pytest
from conftest import JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.weights import make_objective, optimize_weights


@pytest.fixture(scope="module")
def setup():
    ds = load_jura_dataset(JURA_MAT)
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=9,
                    z_thresh=float(np.round(np.log10(50), 5)), objective="threshold")
    z = np.round(ds.z_cal, 5)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    return ds, z, ig


@pytest.fixture(scope="module")
def ref(golden):
    return golden("her2")["her2"]


def test_objective_at_matlab_optimum(setup, ref):
    """The leave-one-out DKL evaluated at MATLAB's optimal weights must reproduce MATLAB's value.
    This checks the whole HER2 plumbing independently of the optimiser."""
    _, z, ig = setup
    obj = make_objective(z, ig)
    assert obj.f_or(np.atleast_1d(ref.best_w_OR)) == pytest.approx(ref.DKL_w_OR, rel=1e-10)
    assert obj.f_and(np.atleast_1d(ref.best_w_AND)) == pytest.approx(ref.DKL_w_AND, rel=1e-10)


@pytest.fixture(scope="module")
def fitted(setup):
    ds, z, ig = setup
    return optimize_weights(ds.x_cal, ds.y_cal, z, ig)


def test_neighbour_cardinality(fitted, ref):
    np.testing.assert_array_equal(fitted.n_pairs_by_class_obs, ref.n_pairs_by_class_obs)
    np.testing.assert_allclose(fitted.nmean_pairs_by_class, np.atleast_1d(ref.nmean_pairs_by_class))


def test_neighbour_indices(fitted, ref):
    np.testing.assert_array_equal(fitted.idx_opt_nn + 1, ref.idx_opt_nn)


def test_optimiser_not_worse_than_matlab(fitted, ref):
    """Solvers differ (fmincon interior-point vs SLSQP): require an objective at least as good as
    MATLAB's up to a small tolerance rather than identical weights."""
    assert fitted.DKL_w_OR <= ref.DKL_w_OR + 1e-4
    assert fitted.DKL_w_AND <= ref.DKL_w_AND + 1e-4


def test_alpha_beta_search_with_matlab_weights(setup, ref):
    from pyher.weights import evaluate_alpha_beta

    _, z, ig = setup
    alpha, beta, dkl, n_nn, _ = evaluate_alpha_beta(
        z, ig, np.atleast_1d(ref.best_w_OR), np.atleast_1d(ref.best_w_AND))
    assert (alpha, beta) == (pytest.approx(ref.best_alpha), pytest.approx(ref.best_beta))
    assert dkl == pytest.approx(ref.DKL_w_alpha_beta, rel=1e-10)
    assert n_nn == ref.n_neighbor_aggreg


# ---- fixed (paper / MATLAB) weights ------------------------------------------------------------
def test_fixed_weights_reproduce_matlab_objective(setup, ref):
    from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W

    ds, z, ig = setup  # n_neighbor = 9, as the golden run
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, fixed_weights=(W["w_or"], W["w_and"]))
    assert w.result_or is None and w.result_and is None  # nothing was optimised
    np.testing.assert_allclose(w.best_w_OR, np.atleast_1d(ref.best_w_OR), rtol=1e-12)
    np.testing.assert_allclose(w.best_w_AND, np.atleast_1d(ref.best_w_AND), rtol=1e-12)
    assert w.DKL_w_OR == pytest.approx(ref.DKL_w_OR, rel=1e-10)
    assert w.DKL_w_AND == pytest.approx(ref.DKL_w_AND, rel=1e-10)
    assert (w.best_alpha, w.best_beta) == (pytest.approx(ref.best_alpha), pytest.approx(ref.best_beta))
    assert w.DKL_w_alpha_beta == pytest.approx(ref.DKL_w_alpha_beta, rel=1e-10)


def test_alpha_beta_override(setup):
    from pyher.reference import JURA_PB_PAPER_ALPHA_BETA
    from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W

    ds, z, ig = setup
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, fixed_weights=(W["w_or"], W["w_and"]),
                         alpha_beta=JURA_PB_PAPER_ALPHA_BETA)
    assert (w.best_alpha, w.best_beta) == JURA_PB_PAPER_ALPHA_BETA
    assert np.isfinite(w.DKL_w_alpha_beta)


def test_fixed_weights_wrong_length_raises(setup):
    ds, z, ig = setup
    with pytest.raises(ValueError, match="one weight per class"):
        optimize_weights(ds.x_cal, ds.y_cal, z, ig, fixed_weights=(np.ones(3), np.ones(3)))


def test_degenerate_or_weights_warn():
    import warnings

    from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W
    from pyher.weights import warn_if_degenerate
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_degenerate(W["w_or"])  # the paper's smooth weights: silent
    collapsed = np.r_[0.4976, 0.4976, np.full(19, 0.0001)]
    with pytest.warns(UserWarning, match="collapsed"):
        warn_if_degenerate(collapsed / collapsed.sum())


# ---- shrinkage of the OR weights (default on, can be switched off) -----------------------------
def test_default_or_weights_are_smooth_and_paper_like(setup, ref):
    ds, z, ig = setup
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig)  # default: shrinkage on
    paper = np.atleast_1d(ref.best_w_OR)
    assert w.best_w_OR[2:].sum() > 0.3  # not collapsed onto the first two classes
    assert w.DKL_w_OR == pytest.approx(ref.DKL_w_OR, abs=2e-3)  # same loss as the fmincon solution
    assert np.mean(np.abs(np.log(w.best_w_OR[:10] / paper[:10]))) < 0.5


@pytest.mark.slow
def test_shrinkage_can_be_switched_off(setup, ref):
    ds, z, ig = setup
    for off in (None, 0):
        with pytest.warns(UserWarning, match="collapsed"):
            w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, or_shrinkage=off)
        assert w.DKL_w_OR == pytest.approx(0.8086, abs=2e-3)  # exact minimiser, below fmincon's 0.882
        assert w.best_w_OR[2:].sum() < 0.02


def test_shrinkage_does_not_touch_and_weights(setup, ref):
    ds, z, ig = setup
    w = optimize_weights(ds.x_cal, ds.y_cal, z, ig)
    assert w.DKL_w_AND == pytest.approx(ref.DKL_w_AND, abs=1e-4)
