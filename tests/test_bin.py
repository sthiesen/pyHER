"""Single-bin DKL objective (sHER_E_03): validated against the original f_her2_weight (03_0, nugget,
30 neighbours, continuous weights in the alpha/beta step) and f_her2_weight_NOnugget (03_1, 9 neighbours)."""

import warnings

import numpy as np
import pytest
from conftest import GOLDEN, JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.smooth import FastOrObjective
from pyher.weights import evaluate_alpha_beta, make_objective, optimize_weights

VARIANTS = {"03_0": dict(nugget=True, n_neighbor=30), "03_1": dict(nugget=False, n_neighbor=9)}


@pytest.fixture(scope="module", params=list(VARIANTS))
def case(request, golden):
    name = request.param
    if not (GOLDEN / f"her2_{name}.mat").exists():
        pytest.skip(f"golden file her2_{name}.mat missing")
    ds = load_jura_dataset(JURA_MAT)
    z = np.round(ds.z_cal, 5)
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, z_thresh=float(np.round(np.log10(50), 5)),
                    objective="bin", **VARIANTS[name])
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    return ds, z, ig, golden(f"her2_{name}")["her2"]


def test_infogram_unchanged(case):
    _, _, ig, ref = case
    assert ig.n_classes_range == ref.n_classes_range
    np.testing.assert_allclose(ig.edges_z, np.atleast_1d(ref.edges_z), rtol=1e-12)


def test_objective_at_matlab_weights(case):
    _, z, ig, ref = case
    full = make_objective(z, ig)
    fast = FastOrObjective(z, ig)
    w_or, w_and = np.atleast_1d(ref.best_w_OR), np.atleast_1d(ref.best_w_AND)
    assert full.f_or(w_or) == pytest.approx(ref.DKL_w_OR, rel=1e-9)
    assert fast(w_or) == pytest.approx(ref.DKL_w_OR, rel=1e-9)
    assert full.f_and(w_and) == pytest.approx(ref.DKL_w_AND, rel=1e-9)


def test_alpha_beta_search_at_matlab_weights(case):
    _, z, ig, ref = case
    a, b, dkl, n_nn, _ = evaluate_alpha_beta(z, ig, np.atleast_1d(ref.best_w_OR), np.atleast_1d(ref.best_w_AND))
    assert (a, b) == (pytest.approx(ref.best_alpha), pytest.approx(ref.best_beta))
    assert dkl == pytest.approx(ref.DKL_w_alpha_beta, rel=1e-9)
    assert n_nn == ref.n_neighbor_aggreg


@pytest.mark.parametrize("shrinkage", [None, 0.0043])
def test_python_optimiser_not_worse_than_matlab_and(case, shrinkage):
    ds, z, ig, ref = case
    with warnings.catch_warnings():  # the unshrunk OR weights are expected to collapse (warned about)
        warnings.filterwarnings("ignore", message="OR weights collapsed")
        w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, or_shrinkage=shrinkage)
    assert w.DKL_w_AND <= ref.DKL_w_AND + 1e-3


def test_unshrunk_weights_are_exactly_monotone_and_prediction_finite(case):
    """SLSQP only satisfies w[i+1] <= w[i] up to its tolerance; a tiny increase made the interpolated
    weights of the NOnugget predictor negative (NaN PMFs on part of the grid) before it was enforced."""
    from pyher.predict import Predictor

    ds, z, ig, ref = case
    with pytest.warns(UserWarning, match="OR weights collapsed"):
        w = optimize_weights(ds.x_cal, ds.y_cal, z, ig, or_shrinkage=None)
    assert (np.diff(w.best_w_OR) <= 0).all() and (np.diff(w.best_w_AND) <= 0).all()
    gx, gy = np.meshgrid(np.arange(0, 6.01, 0.25), np.arange(0, 6.01, 0.25))
    pmf, _ = Predictor(ds.x_cal, ds.y_cal, z, ig, w).predict(gx.ravel(), gy.ravel())
    assert np.isfinite(pmf).all()


def test_default_shrinkage_depends_on_objective():
    from pyher.weights import DEFAULT_OR_SHRINKAGE, resolve_or_shrinkage

    assert resolve_or_shrinkage("auto", "threshold") == DEFAULT_OR_SHRINKAGE
    assert resolve_or_shrinkage("auto", "bin") is None
    assert resolve_or_shrinkage(None, "threshold") is None
    assert resolve_or_shrinkage(0.001, "bin") == 0.001
    with pytest.raises(ValueError):
        resolve_or_shrinkage("yes", "bin")


def test_bin_default_reproduces_matlab_or_weights(case):
    """With the default ('auto') the bin objective is unshrunk and reproduces MATLAB's OR loss."""
    ds, z, ig, ref = case
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = optimize_weights(ds.x_cal, ds.y_cal, z, ig)
    assert w.DKL_w_OR == pytest.approx(ref.DKL_w_OR, abs=2e-3)
