"""Edge cases that used to fail silently or crash."""

import numpy as np
import pytest

from pyher import HerConfig
from pyher.infogram import fit_infogram
from pyher.predict import Predictor
from pyher.stats import extract_pmf_statistics, validate
from pyher.weights import optimize_weights


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.random(n) * 10, rng.random(n) * 10
    z = np.round(np.sin(x / 3) + 0.3 * rng.normal(size=n), 2)
    return x, y, z


CFG = dict(lag_dist=1.0, binwidth=0.4, binwidth_z=0.4, z_thresh=0.0, objective="bin", nugget=True)


def test_infogram_keeps_its_own_copy_of_the_config():
    x, y, z = _data()
    cfg = HerConfig(**CFG)
    ig = fit_infogram(x, y, z, cfg)
    cfg.n_neighbor = 3
    assert ig.cfg.n_neighbor == 30


def test_empty_class_inside_the_range_raises():
    x, y, z = _data()
    with pytest.raises(ValueError, match="Empty distance classes"):
        fit_infogram(x, y, z, HerConfig(**{**CFG, "lag_dist": 0.01}))


def test_more_neighbours_than_points_does_not_crash():
    x, y, z = _data(40)
    cfg = HerConfig(**{**CFG, "n_neighbor": 100})
    ig = fit_infogram(x, y, z, cfg)
    w = optimize_weights(x, y, z, ig, alpha_beta=(1.0, 0.0), fixed_weights=(np.ones(ig.n_classes_range),) * 2)
    assert w.n_neighbor_aggreg == len(x) - 1
    pmf, _ = Predictor(x, y, z, ig, w).predict([5.0], [5.0])
    assert np.isfinite(pmf).all() and pmf.sum() == pytest.approx(1.0)


def test_coincident_points_are_not_their_own_neighbours():
    x, y, z = _data(40)
    x[1], y[1] = x[0], y[0]  # point 1 sits exactly on point 0
    ig = fit_infogram(x, y, z, HerConfig(**{**CFG, "n_neighbor": 10}))
    R = ig.n_classes_range
    w = optimize_weights(x, y, z, ig, alpha_beta=(1.0, 0.0), fixed_weights=(np.ones(R), np.ones(R)))
    assert 1 not in w.idx_opt_nn[:, 0] and 0 not in w.idx_opt_nn[:, 1]
    assert all(t not in w.idx_opt_nn[:, t] for t in range(len(x)))


def test_weights_remember_the_nugget_rule_and_the_predictor_checks_it():
    x, y, z = _data()
    ig = fit_infogram(x, y, z, HerConfig(**CFG))
    R = ig.n_classes_range
    w = optimize_weights(x, y, z, ig, alpha_beta=(1.0, 0.0), fixed_weights=(np.ones(R), np.ones(R)))
    assert w.nugget is True and w.objective == "bin"
    ig.cfg.nugget = False
    with pytest.raises(ValueError, match="nugget"):
        Predictor(x, y, z, ig, w)


def test_validate_needs_a_threshold_and_stats_accept_a_threshold_above_the_edges():
    with pytest.raises(ValueError, match="threshold"):
        validate(np.full((2, 3), 1 / 3), [0.1, 0.2], [0.0, 1.0], np.arange(4.0), np.arange(3.0) + 0.5, None)
    pmfs = np.full((2, 3), 1 / 3)
    st = extract_pmf_statistics(pmfs, np.arange(4.0), np.arange(3.0) + 0.5, z_thresh=99.0)
    np.testing.assert_allclose(st.prob_exceed, 0.0, atol=1e-12)
