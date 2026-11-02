import numpy as np
import pytest
from conftest import JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.reference import JURA_PB_THRESHOLD_WEIGHTS as W
from pyher.smooth import FastOrObjective, SmoothSpec, cross_validate, fit_or_weights
from pyher.weights import make_objective


@pytest.fixture(scope="module")
def setup():
    ds = load_jura_dataset(JURA_MAT)
    z = np.round(ds.z_cal, 5)
    cfg = HerConfig(n_neighbor=30, z_thresh=float(np.round(np.log10(50), 5)))
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)
    return z, ig, FastOrObjective(z, ig), make_objective(z, ig)


def test_fast_objective_equals_full_objective(setup):
    z, ig, fast, full = setup
    rng = np.random.default_rng(0)
    for w in [W["w_or"], *[rng.random(fast.R) for _ in range(5)]]:
        assert fast(w) == pytest.approx(full.f_or(w), rel=1e-10)


def test_fast_objective_at_matlab_optimum(setup):
    _, _, fast, _ = setup
    assert fast(W["w_or"]) == pytest.approx(0.8817033375734491, rel=1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
def test_ratio_constraint_holds(setup, rho):
    _, _, fast, _ = setup
    w, _ = fit_or_weights(fast, SmoothSpec("ratio", rho))
    assert (np.diff(w) <= 1e-7).all()  # monotone decay (absolute tolerance: weights can sit at 1e-6)
    big = w[:-1] > 1e-4
    assert (w[1:][big] / w[:-1][big] >= rho - 1e-5).all()  # no abrupt drops


def test_smoothing_costs_training_loss_monotonically(setup):
    _, _, fast, _ = setup
    losses = [fast(fit_or_weights(fast, SmoothSpec("ratio", r))[0]) for r in (0.0, 0.3, 0.6, 0.9)]
    assert all(b >= a - 1e-6 for a, b in zip(losses, losses[1:]))


def test_powerlaw_is_smooth_and_monotone(setup):
    _, _, fast, _ = setup
    w, _ = fit_or_weights(fast, SmoothSpec("powerlaw"))
    assert (np.diff(w) <= 1e-12).all()


def test_cross_validation_picks_valid_index(setup):
    _, _, fast, _ = setup
    cv = cross_validate(fast, "ratio", [0.0, 0.3, 0.6], k=3)
    assert 0 <= cv.best <= cv.one_se < 3


def test_logprior_limits(setup):
    """lambda -> 0 gives the unconstrained optimum, a large lambda the 1/distance guess."""
    from pyher.weights import _initial_weights

    fast = setup[2]
    free, _ = fit_or_weights(fast, SmoothSpec("none"))
    w0, _ = fit_or_weights(fast, SmoothSpec("logprior", 0.0))
    assert fast(w0) == pytest.approx(fast(free), abs=1e-4)
    w_big, _ = fit_or_weights(fast, SmoothSpec("logprior", 1e3))
    prior = _initial_weights(fast.centers)
    np.testing.assert_allclose(w_big, prior / prior.sum(), rtol=5e-2)


def test_logprior_reproduces_paper_like_weights(setup):
    """With the strength that matches the paper's training loss the weights stay close to the paper's."""
    _, _, fast, _ = setup
    w, _ = fit_or_weights(fast, SmoothSpec("logprior", 0.0043))
    paper = np.asarray(W["w_or"]) / np.sum(W["w_or"])
    assert fast(w) == pytest.approx(fast(paper), abs=0.005)
    assert np.mean(np.abs(np.log(w[:10] / paper[:10]))) < 0.5


def test_fast_bin_objective_equals_full_objective():
    ds = load_jura_dataset(JURA_MAT)
    z = np.round(ds.z_cal, 5)
    ig = fit_infogram(ds.x_cal, ds.y_cal, z, HerConfig(n_neighbor=30, objective="bin"))
    fast, full = FastOrObjective(z, ig), make_objective(z, ig)
    rng = np.random.default_rng(1)
    for w in [W["w_or"], *[rng.random(fast.R) for _ in range(5)]]:
        assert fast(w) == pytest.approx(full.f_or(w), rel=1e-10)
