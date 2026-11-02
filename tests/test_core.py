import numpy as np
import pytest

from pyher import core


@pytest.fixture(scope="module")
def leaf(golden):
    return golden("leaf")["leaf"]


def test_entropy(leaf):
    assert core.entropy(leaf.pmfs[0]) == pytest.approx(leaf.entropy, rel=1e-12)


def test_linear_aggregation(leaf):
    np.testing.assert_allclose(core.linear_aggregation(leaf.pmfs, leaf.w), leaf.linear, rtol=1e-12)


def test_loglinear_aggregation(leaf):
    np.testing.assert_allclose(core.loglinear_aggregation(leaf.pmfs, leaf.w), leaf.loglinear, rtol=1e-12)


def test_monte_carlo_draw(leaf):
    got = core.monte_carlo_draw(leaf.pmfs[0], leaf.edges, float(leaf.mc_p))
    assert got == pytest.approx(leaf.montecarlo, rel=1e-12)


def test_performance_prob(leaf):
    got = core.performance_prob(leaf.perf_z, leaf.perf_pmf, leaf.perf_edges)
    assert got == pytest.approx(leaf.perf, rel=1e-12)
