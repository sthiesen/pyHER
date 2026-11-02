import numpy as np
import pytest

from pyher.binwidth import BinwidthResult, cross_entropy_convergence, delta_z_pairs, symmetric_edges


@pytest.fixture(scope="module")
def normal_data():
    return np.random.default_rng(0).normal(size=5000)


@pytest.mark.parametrize("strategy", ["random", "continuous"])
def test_full_sample_has_no_excess(normal_data, strategy):
    """With the whole data set as 'subsample' q equals p: Hpq = H(p), DKL = 0."""
    r = cross_entropy_convergence(normal_data, [0.1, 0.4], [100, 5000], n_rep=20, strategy=strategy)
    np.testing.assert_allclose(r.dkl_mean[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(r.hpq_std[:, -1], 0.0, atol=1e-12)


def test_excess_is_nonnegative_and_shrinks_with_sample_size(normal_data):
    r = cross_entropy_convergence(normal_data, [0.05, 0.2, 0.8], [20, 50, 200, 1000, 5000], n_rep=300)
    assert (r.dkl_mean >= -1e-12).all()  # Gibbs' inequality
    assert (np.diff(r.dkl_mean, axis=1) < 0.02).all()  # decreasing with n (up to Monte Carlo noise)
    assert (r.dkl_mean[:, 0] > r.dkl_mean[:, -1]).all()


def test_finer_bins_need_more_data(normal_data):
    r = cross_entropy_convergence(normal_data, [0.05, 0.2, 0.8], [30], n_rep=300)
    d = r.dkl_mean[:, 0]
    assert d[0] > d[1] > d[2]
    assert r.n_bins[0] > r.n_bins[1] > r.n_bins[2]
    assert r.H_p[0] > r.H_p[2]  # finer bins -> larger entropy (about one bit per halving)


def test_entropy_gain_per_halving_is_about_one_bit(normal_data):
    r = cross_entropy_convergence(normal_data, [0.4, 0.2], [5000], n_rep=2)
    assert r.H_p[1] - r.H_p[0] == pytest.approx(1.0, abs=0.1)


def test_continuous_strategy_against_direct_computation():
    """Two possible blocks of size N-1: the mean must lie between their exactly computed cross entropies."""
    data = np.random.default_rng(1).normal(size=60)
    bw, n = 0.5, 59
    r = cross_entropy_convergence(data, [bw], [n], n_rep=400, strategy="continuous", seed=3)
    edges = symmetric_edges(data, bw)
    B = len(edges) - 1
    p = (np.histogram(data, bins=edges)[0] + 1.0) / (len(data) + B)
    exact = []
    for block in (data[:n], data[1:]):
        q = (np.histogram(block, bins=edges)[0] + 1.0) / (n + B)
        exact.append(-(p * np.log2(q)).sum())
    assert min(exact) - 1e-9 <= r.hpq_mean[0, 0] <= max(exact) + 1e-9
    assert r.hpq_std[0, 0] <= abs(exact[0] - exact[1]) / 2 + 1e-9 + 0.01


def test_reproducible_with_seed_and_validation(normal_data):
    a = cross_entropy_convergence(normal_data, [0.2], [50, 500], n_rep=30, seed=5)
    b = cross_entropy_convergence(normal_data, [0.2], [50, 500], n_rep=30, seed=5)
    np.testing.assert_array_equal(a.hpq_mean, b.hpq_mean)
    with pytest.raises(ValueError):
        cross_entropy_convergence(normal_data, [0.2], [10_000])
    with pytest.raises(ValueError):
        cross_entropy_convergence(normal_data, [0.2], [10], strategy="bootstrap")


def test_recommend_picks_the_smallest_bin_width_within_tolerance():
    bws = np.array([0.01, 0.02, 0.05, 0.1])
    res = BinwidthResult(bws, np.array([20, 200]), np.array([400, 200, 80, 40]), np.zeros(4),
                         np.array([[2.0, 0.3], [0.4, 0.05], [0.09, 0.01], [0.03, 0.0]]), np.zeros((4, 2)), 1000, 10, "random")
    assert res.recommend(n_target=20, dkl_tol=0.1) == 0.05
    assert res.recommend(n_target=200, dkl_tol=0.1) == 0.02
    assert res.recommend(n_target=20, dkl_tol=0.01) is None


def test_delta_z_pairs_of_a_class():
    x = np.array([0.0, 1.0, 3.0])
    y = np.zeros(3)
    z = np.array([1.0, 2.0, 5.0])
    # distances: (0,1)=1, (1,2)=2, (0,2)=3 ; lag 1.5 -> class 1 = (0, 1.5], class 2 = (1.5, 3]
    c1 = delta_z_pairs(x, y, z, 1.5, 1)
    assert sorted(c1) == [-1.0, 1.0]  # both triangles kept
    c2 = delta_z_pairs(x, y, z, 1.5, 2, both_triangles=False)
    assert sorted(c2) == sorted([z[0] - z[2], z[1] - z[2]])  # upper triangle only: (0,2) and (1,2)


def test_samples_needed_interpolates_and_flags_unreachable():
    bws = np.array([0.1, 0.5])
    res = BinwidthResult(bws, np.array([10, 100, 1000]), np.array([40, 8]), np.zeros(2),
                         np.array([[1.0, 0.1, 0.01], [0.05, 0.005, 0.0005]]), np.zeros((2, 3)), 1000, 10, "random")
    need = res.samples_needed(0.1)
    assert need[0] == pytest.approx(100.0)  # exactly on a grid point
    assert need[1] == 10  # tolerance already met at the smallest size
    assert np.isnan(res.samples_needed(0.001)[0])  # not reached even with all data
    mid = res.samples_needed(0.316)[0]  # log-log midpoint between (10, 1.0) and (100, 0.1)
    assert 10 < mid < 100


def test_delta_z_by_class_counts_and_consistency():
    from pyher.binwidth import delta_z_by_class

    rng = np.random.default_rng(0)
    x, y, z = rng.random(120) * 10, rng.random(120) * 10, rng.normal(size=120)
    groups = delta_z_by_class(x, y, z, 1.0, 6)
    assert [len(g) for g in groups] == [len(delta_z_pairs(x, y, z, 1.0, k)) for k in range(1, 7)]
    assert sum(len(g) for g in groups) <= 120 * 119


def test_classwise_first_classes_have_fewest_pairs_and_limit_the_bin_width():
    from pyher.binwidth import classwise_cross_entropy

    rng = np.random.default_rng(2)
    x, y = rng.random(400) * 20, rng.random(400) * 20
    z = np.sin(x / 4) + 0.3 * rng.normal(size=400)
    cw = classwise_cross_entropy(x, y, z, 1.0, 6, [0.05, 0.1, 0.2, 0.4, 0.8], fraction=0.2, n_rep=60)
    assert (np.diff(cw.n_values) > 0).all()  # uniform points: pairs grow with the distance class
    assert cw.dkl_at_target.shape == (6, 5)
    # more data per class -> smaller excess at the same fraction of the class (on average over the bin widths)
    assert cw.dkl_at_target[0].mean() > cw.dkl_at_target[-1].mean()
    tol = 0.1
    first, allc = cw.recommend(tol, classes=[1]), cw.recommend(tol)
    assert first is not None and allc is not None and allc >= first  # the other classes can only add constraints
    assert cw.margin(tol).shape == (6, 5)


def test_classwise_skips_tiny_classes():
    from pyher.binwidth import classwise_cross_entropy

    rng = np.random.default_rng(3)
    x, y, z = rng.random(30) * 10, rng.random(30) * 10, rng.normal(size=30)
    cw = classwise_cross_entropy(x, y, z, 0.3, 8, [0.3, 0.6], n_rep=20, min_values=15)
    assert len(cw.class_index) < 8 and (cw.n_values >= 15).all()


def test_limiting_class_is_the_class_with_fewest_pairs_among_the_first():
    from pyher.binwidth import delta_z_by_class, limiting_class, limiting_class_cross_entropy

    rng = np.random.default_rng(4)
    x, y = rng.random(300) * 10, rng.random(300) * 10
    z = rng.normal(size=300)
    # a gap: no points between 3 and 4 units of a cluster -> make class 3 poor by removing pairs
    counts = [len(g) for g in delta_z_by_class(x, y, z, 1.0, 10)]
    k, data = limiting_class(x, y, z, 1.0, 10)
    assert k == int(np.argmin(counts)) + 1 and len(data) == min(counts)
    k5, _ = limiting_class(x, y, z, 1.0, 5)
    assert k5 == int(np.argmin(counts[:5])) + 1
    k2, res = limiting_class_cross_entropy(x, y, z, 1.0, [0.2, 0.5], n_first=10, n_rep=20)
    assert k2 == k and res.n_data == min(counts)
    with pytest.raises(ValueError):
        limiting_class(x[:6], y[:6], z[:6], 1.0, 5, min_values=100)


def test_excess_at_size_reproduces_the_true_excess_of_gaussian_data():
    """Smooth-bootstrap estimate vs the exact expectation for N(0,1) data (known truth)."""
    from scipy.stats import norm

    from pyher.binwidth import excess_at_size

    rng = np.random.default_rng(5)
    N, bw = 400, 0.2
    data = rng.normal(size=N)
    est, _ = excess_at_size(data, [bw], n_rep=200, seed=1)
    edges = symmetric_edges(data, bw)  # the bins HER uses: they cover the range of the data
    nb = len(edges) - 1
    P = np.diff(norm.cdf(edges))
    P /= P.sum()
    truth = []
    for _ in range(1500):
        c = np.histogram(rng.normal(size=N), bins=edges)[0]
        q = (c + 1.0) / (N + nb)
        truth.append(np.sum(P * np.log2(P / q)))
    assert est[0] == pytest.approx(np.mean(truth), rel=0.15)


def test_excess_at_size_decreases_with_sample_size_and_bin_width():
    from pyher.binwidth import excess_at_size

    data = np.random.default_rng(6).normal(size=500)
    fine, coarse = excess_at_size(data, [0.1, 0.8], n_rep=60)[0]
    assert fine > coarse
    small = excess_at_size(data, [0.2], n=100, n_rep=60)[0][0]
    large = excess_at_size(data, [0.2], n=500, n_rep=60)[0][0]
    assert small > large


def test_actual_size_excess_by_class_and_limiting_class():
    from pyher.binwidth import actual_size_excess, delta_z_by_class

    rng = np.random.default_rng(7)
    x, y = rng.random(250) * 10, rng.random(250) * 10
    z = np.sin(x / 3) + 0.3 * rng.normal(size=250)
    counts = [len(g) for g in delta_z_by_class(x, y, z, 1.0, 6)]
    res = actual_size_excess(x, y, z, 1.0, [0.1, 0.3, 0.9], n_first=6, n_rep=40)
    assert res.limiting_class == int(np.argmin(counts)) + 1
    assert res.excess.shape == (6, 3) and (np.diff(res.excess, axis=1) < 0).all()  # coarser bins: smaller excess
    only = actual_size_excess(x, y, z, 1.0, [0.1, 0.3, 0.9], n_first=6, only_limiting=True, n_rep=40)
    assert only.class_index.tolist() == [res.limiting_class]
    rec_all, rec_lim = res.recommend(0.05), only.recommend(0.05)
    assert rec_all is None or rec_lim is None or rec_all >= rec_lim


def test_class_assignment_matches_fit_infogram_at_exact_class_edges():
    """Pairs at a distance that is an exact multiple of the lag (integer grid, lag 1.4) must fall in the same class
    as in ``fit_infogram`` ((e_i, e_i+1] on the MATLAB colon edges)."""
    from pyher import HerConfig
    from pyher.binwidth import delta_z_by_class
    from pyher.infogram import fit_infogram

    gx, gy = np.meshgrid(np.arange(8.0), np.arange(6.0))
    x, y = gx.ravel(), gy.ravel()
    z = np.random.default_rng(0).normal(size=x.size)
    cfg = HerConfig(lag_dist=1.4, binwidth=0.5, binwidth_z=0.5, z_thresh=0.0)
    ig = fit_infogram(x, y, z, cfg)
    groups = delta_z_by_class(x, y, z, 1.4, ig.n_lag)
    assert [len(g) for g in groups] == ig.n_pairs_by_class.tolist()


def test_her_prob_min_is_one_over_the_number_of_ordered_pairs():
    from pyher.binwidth import her_prob_min

    x, y = np.array([0.0, 1.0, 2.0, 2.0]), np.zeros(4)
    assert her_prob_min(x, y) == pytest.approx(1 / 10)  # 12 ordered pairs minus the 2 at distance 0


def test_pair_excess_estimate_is_a_slightly_low_estimate_of_the_true_excess_on_random_fields():
    """Spatially dependent pairs: smooth bootstrap of the mirrored pairs vs the true expected excess of the first
    class of simulated Gaussian random fields (exponential covariance), with HER's smoothing."""
    from pyher.binwidth import delta_z_by_class, excess_at_size, her_prob_min

    rng = np.random.default_rng(10)
    n, lag, bw = 150, 0.7, 0.3
    fields = []
    for _ in range(60):
        xy = rng.random((n, 2)) * 10
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        z = np.linalg.cholesky(np.exp(-d / 2.0) + 1e-8 * np.eye(n)) @ rng.normal(size=n)
        fields.append((xy[:, 0], xy[:, 1], z))
    both = [delta_z_by_class(x, y, z, lag, 1)[0] for x, y, z in fields]
    pool = np.concatenate(both)
    edges = symmetric_edges(pool, bw)
    P = np.histogram(pool, bins=edges)[0] + 1e-9
    P /= P.sum()
    true, est = [], []
    for k, (x, y, z) in enumerate(fields):
        pm = her_prob_min(x, y)
        q = np.histogram(both[k], bins=edges)[0] / len(both[k]) + pm
        true.append(np.sum(P * np.log2(P / (q / q.sum()))))
        if k < 15:
            u = delta_z_by_class(x, y, z, lag, 1, both_triangles=False)[0]
            est.append(excess_at_size(u, [bw], n_rep=60, prob_min=pm, symmetrize=True, seed=k)[0][0])
    ratio = np.mean(est) / np.mean(true)
    assert 0.6 < ratio < 1.15, ratio
