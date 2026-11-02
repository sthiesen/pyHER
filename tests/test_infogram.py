import numpy as np
import pytest
from conftest import JURA_MAT

from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram


@pytest.fixture(scope="module")
def fit():
    ds = load_jura_dataset(JURA_MAT)
    cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=9,
                    z_thresh=float(np.round(np.log10(50), 5)))
    return fit_infogram(ds.x_cal, ds.y_cal, np.round(ds.z_cal, 5), cfg)


@pytest.fixture(scope="module")
def ref(golden):
    return golden("her1")["her1"]


SCALARS = ["n_bins_z_per_bin_shift", "binwidth_shift", "n_lag", "n_bins_shift", "prob_min",
           "H_diff_z", "H_lim", "n_classes_range", "n_bins_z"]
ARRAYS = ["mat_euc_distance_xy", "mat_diff_z", "edges_distance_classes", "edges_diff_z",
          "edges_diff_z_shift", "H_diff_z_by_class", "edges_distance_classes_range",
          "bin_centers_distance_classes", "bin_centers_distance_classes_range",
          "pmf_diff_z_by_class_obs_range", "pmf_diff_z_by_class_obs_range_shift",
          "H_diff_z_by_class_range", "edges_z", "bin_centers_edges_z", "bin_centers_edges_diff_z",
          "n_pairs_by_class"]


@pytest.mark.parametrize("name", SCALARS)
def test_scalars(fit, ref, name):
    assert getattr(fit, name) == pytest.approx(getattr(ref, name), rel=1e-12)


@pytest.mark.parametrize("name", ARRAYS)
def test_arrays(fit, ref, name):
    got, want = np.asarray(getattr(fit, name)), np.asarray(getattr(ref, name))
    assert got.size == want.size, (got.shape, want.shape)
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-12, atol=1e-14)


def test_empty_classes(fit, ref):
    np.testing.assert_array_equal(fit.emptyclass + 1, np.atleast_1d(ref.emptyclass))


def test_obs_diff_z(fit, ref):
    np.testing.assert_allclose(np.sort(fit.obs_diff_z), np.sort(ref.obs_diff_z), rtol=1e-12)


def test_infogram_entropy_equals_fit_infogram(fit):
    """f_infogram_realizplot = steps 1-6 of f_her1_infogram: same entropies by distance class."""
    from pyher.infogram import infogram_entropy

    ds = load_jura_dataset(JURA_MAT)
    _, H, H_all = infogram_entropy(ds.x_cal, ds.y_cal, np.round(ds.z_cal, 5), 0.07, 0.015)
    np.testing.assert_allclose(H, fit.H_diff_z_by_class, rtol=1e-12, equal_nan=True)
    assert H_all == pytest.approx(fit.H_diff_z, rel=1e-12)


def test_infogram_entropy_subsample_and_padding():
    from pyher.infogram import infogram_entropy

    rng = np.random.default_rng(0)
    x, y, z = rng.random(300) * 10, rng.random(300) * 10, rng.normal(size=300)
    c, H, _ = infogram_entropy(x, y, z, 1.0, 0.2, max_points=100, n_classes=30)
    assert len(H) == len(c) == 30
