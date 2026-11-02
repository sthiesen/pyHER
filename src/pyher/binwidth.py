"""Bin-width selection by cross-entropy convergence (stand-alone tool).

HER turns frequency distributions of delta-z into PMFs, so the bin width frames every number computed
afterwards (entropies, DKL). Following Thiesen et al. (2018/2019, HESS 23, 1015) and Sect. 2.2 of Thiesen et al.
(2020, HESS 24, 4523), the bin width is judged by comparing the cross entropy

    Hpq = H(p) + DKL(p || q) = - sum_x p(x) log2 q(x)

between the PMF ``p`` of the full learning set and the PMFs ``q`` of random subsamples of growing size. A bin
width is reasonable when this cross entropy is already stable (close to H(p)) for small sample sizes: bins that are
too fine make ``q`` of a small sample a poor description of ``p``, and ``Hpq`` is far above ``H(p)``.

This is an independent implementation based on that definition. It replaces ``f_binverificationTDE.m`` of the
MATLAB repository, which depends on two functions (``f_histcounts_anyd``, ``f_infomeasures_from_samples``) that are
not part of it, so the numbers are not expected to be identical to a MATLAB run.

Conventions (as in HER): equidistant bins, symmetric around zero (``n_bins = 2 floor(max|x| / bw + 1)``), distance
classes ``(e_i, e_i+1]`` on the MATLAB colon edges, and a small probability added to every bin so that no PMF has
empty bins. HER adds ``prob_min = 1 / (number of ordered point pairs)`` to every bin probability; the functions that
work on a point set use that by default (``prob_min="her"``). The 1-D functions take either ``prob_min`` or a
``pseudo_count`` (counts added per bin; ``prob_min`` p corresponds to ``n p`` counts for a sample of n values).
"""

from dataclasses import dataclass

import numpy as np

from pyher.core import matlab_colon, matlab_linspace


def _distances(x, y, z):
    x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
    dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    return dist, z[:, None] - z[None, :]


def _class_of(dist, lag_dist: float) -> np.ndarray:
    """0-based distance class of every pair, as ``fit_infogram``: classes ``(e_i, e_i+1]`` on the MATLAB colon edges
    ``0:lag:lag*ceil(max/lag)`` (so a pair at an exact edge belongs to the lower class). -1 for distance 0."""
    n_lag = int(np.ceil(dist.max() / lag_dist))
    edges = matlab_colon(0.0, lag_dist, lag_dist * n_lag)
    return np.searchsorted(edges, dist, side="left") - 1


def her_prob_min(x, y) -> float:
    """``prob_min`` of ``f_her1_infogram``: 1 / (number of ordered point pairs with distance > 0)."""
    x, y = (np.asarray(a, dtype=float).ravel() for a in (x, y))
    dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    return 1.0 / np.count_nonzero(dist > 0)


def delta_z_pairs(x, y, z, lag_dist: float, class_index: int = 1, *, both_triangles: bool = True) -> np.ndarray:
    """Delta-z of all point pairs whose distance lies in distance class ``class_index`` (1-based, as in the paper:
    class 1 is ``(0, lag_dist]``). ``both_triangles=True`` keeps (i, j) and (j, i) as ``f_her1_infogram`` does."""
    dist, diff = _distances(x, y, z)
    mask = _class_of(dist, lag_dist) == class_index - 1
    if not both_triangles:
        mask &= np.triu(np.ones_like(mask, dtype=bool), 1)
    return diff[mask]


def symmetric_edges(data, binwidth: float) -> np.ndarray:
    """Bin edges of width ``binwidth``, symmetric around zero, covering ``data`` (as ``f_her1_infogram``)."""
    n_bins = 2 * int(np.floor(np.abs(data).max() / binwidth + 1))
    half = n_bins / 2 * binwidth
    return matlab_linspace(-half, half, n_bins + 1)


@dataclass
class BinwidthResult:
    """Cross entropy of subsamples against the full set, for several bin widths and sample sizes."""

    binwidths: np.ndarray  # [nb]
    sample_sizes: np.ndarray  # [ns]
    n_bins: np.ndarray  # [nb]
    H_p: np.ndarray  # [nb] entropy of the full-set PMF [bit]
    hpq_mean: np.ndarray  # [nb, ns] mean cross entropy over the repetitions [bit]
    hpq_std: np.ndarray  # [nb, ns]
    n_data: int
    n_rep: int
    strategy: str

    @property
    def dkl_mean(self) -> np.ndarray:
        """Excess of the cross entropy over H(p), i.e. the mean DKL(p || q) [bit] ([nb, ns])."""
        return self.hpq_mean - self.H_p[:, None]

    def samples_needed(self, dkl_tol: float = 0.1) -> np.ndarray:
        """Sample size at which the mean DKL(p || q) falls to ``dkl_tol`` bit, per bin width (interpolated in
        log-log between the analysed sizes). The smallest analysed size is returned when the tolerance is met
        there already, ``nan`` when it is not met even by the full data set."""
        out = np.empty(len(self.binwidths))
        for i, d in enumerate(self.dkl_mean):
            if d[0] <= dkl_tol:
                out[i] = self.sample_sizes[0]
            elif d[-1] > dkl_tol:
                out[i] = np.nan
            else:
                out[i] = np.exp(np.interp(np.log(dkl_tol), np.log(np.maximum(d[::-1], 1e-12)),
                                          np.log(self.sample_sizes[::-1])))
        return out

    def recommend(self, n_target: int | None = None, dkl_tol: float = 0.1) -> float | None:
        """Heuristic: the smallest bin width whose mean DKL(p || q) at sample size ``n_target`` is at most
        ``dkl_tol`` bit, i.e. the finest resolution that a sample of that size still represents well.
        ``n_target`` defaults to the smallest sample size in the analysis. Returns ``None`` if no width qualifies.

        The excess DKL falls monotonically with coarser bins (fewer bins are easier to estimate), so this is a
        trade-off between resolution and the smallest sample you want to support: the answer depends on both
        ``n_target`` and ``dkl_tol``. The papers choose the width by eye ("stabilisation of the cross entropy for
        small sample sizes") and by the measurement uncertainty, so treat this as a guide and look at the curves
        (``plots.plot_binwidth_selection``) and at ``samples_needed``.
        """
        j = 0 if n_target is None else int(np.argmin(np.abs(self.sample_sizes - n_target)))
        ok = np.flatnonzero(self.dkl_mean[:, j] <= dkl_tol)
        if ok.size == 0:
            return None
        return float(self.binwidths[ok[np.argmin(self.binwidths[ok])]])


def _cross_entropy(p, counts, n, pseudo):
    """Cross entropy -sum p log2 q for sample count rows ``counts`` [reps, B] of size n, q = Laplace-type smoothing."""
    q = (counts + pseudo) / (n + pseudo * counts.shape[1])
    return -(p * np.log2(q)).sum(axis=1)


def cross_entropy_convergence(
    data,
    binwidths,
    sample_sizes=None,
    *,
    n_rep: int = 500,
    strategy: str = "random",
    pseudo_count: float = 1.0,
    prob_min: float | None = None,
    seed: int = 0,
) -> BinwidthResult:
    """Cross entropy ``Hpq = H(p) + DKL(p || q)`` between the full data set and subsamples, per bin width.

    ``data``: 1-D sample (e.g. the delta-z of the pairs of the first distance class, ``delta_z_pairs``).
    ``sample_sizes``: sizes of the subsamples (default: 12 geometrically spaced sizes from 10 to the full size).
    ``strategy``: ``"random"`` draws subsamples without replacement (exact, via the multivariate hypergeometric
    distribution); ``"continuous"`` takes a block of consecutive values of ``data`` starting at a random position.
    ``n_rep`` repetitions are averaged (the MATLAB function used 500).
    ``prob_min``: HER-style smoothing, a probability added to every bin of every PMF (``n * prob_min`` counts for a
    sample of n values); when given it replaces ``pseudo_count``.
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]
    N = data.size
    binwidths = np.atleast_1d(np.asarray(binwidths, dtype=float))
    if sample_sizes is None:
        sample_sizes = np.unique(np.round(np.geomspace(10, N, 12)).astype(int))
    sample_sizes = np.asarray(sample_sizes, dtype=int)
    if sample_sizes.min() < 1 or sample_sizes.max() > N:
        raise ValueError(f"sample sizes must be in 1..{N}")
    if strategy not in ("random", "continuous"):
        raise ValueError("strategy must be 'random' or 'continuous'")
    rng = np.random.default_rng(seed)

    nb, ns = len(binwidths), len(sample_sizes)
    hpq_mean, hpq_std = np.empty((nb, ns)), np.empty((nb, ns))
    H_p, n_bins = np.empty(nb), np.empty(nb, dtype=int)
    for i, bw in enumerate(binwidths):
        edges = symmetric_edges(data, bw)
        B = len(edges) - 1
        n_bins[i] = B
        counts = np.histogram(data, bins=edges)[0].astype(np.int64)
        pc_full = pseudo_count if prob_min is None else N * prob_min
        p = (counts + pc_full) / (N + pc_full * B)
        H_p[i] = -(p * np.log2(p)).sum()
        if strategy == "continuous":
            bin_idx = np.clip(np.searchsorted(edges, data, side="right") - 1, 0, B - 1)
        for j, n in enumerate(sample_sizes):
            if strategy == "random":
                c = rng.multivariate_hypergeometric(counts, int(n), size=n_rep).astype(float)
            else:
                starts = rng.integers(0, N - n + 1, size=n_rep)
                c = np.array([np.bincount(bin_idx[s:s + n], minlength=B) for s in starts], dtype=float)
            h = _cross_entropy(p, c, n, pseudo_count if prob_min is None else n * prob_min)
            hpq_mean[i, j], hpq_std[i, j] = h.mean(), h.std(ddof=1) if n_rep > 1 else 0.0
    return BinwidthResult(binwidths, sample_sizes, n_bins, H_p, hpq_mean, hpq_std, N, n_rep, strategy)


# ----------------------------------------------------------------------------------------------
# class by class: the classes inside the range matter, the first ones most
# ----------------------------------------------------------------------------------------------
def delta_z_by_class(x, y, z, lag_dist: float, n_classes: int, *, both_triangles: bool = True) -> list[np.ndarray]:
    """Delta-z of the point pairs of each of the first ``n_classes`` distance classes
    (class k = ``((k-1) lag, k lag]``, the same assignment as ``fit_infogram``)."""
    dist, diff = _distances(x, y, z)
    cls = _class_of(dist, lag_dist)
    if not both_triangles:
        cls = np.where(np.triu(np.ones_like(cls, dtype=bool), 1), cls, -2)
    return [diff[cls == k - 1] for k in range(1, n_classes + 1)]


@dataclass
class ClasswiseResult:
    """Cross-entropy convergence class by class (``n_classes`` classes, usually those inside the range)."""

    binwidths: np.ndarray  # [nb]
    class_index: np.ndarray  # [K] 1-based distance class of each row
    n_values: np.ndarray  # [K] delta-z values (pairs) of each class
    fraction: float  # the subsample size analysed in each class is fraction * n_values[k]
    n_target: np.ndarray  # [K]
    dkl_at_target: np.ndarray  # [K, nb] mean DKL(p || q) [bit] at n_target
    results: list  # BinwidthResult per class

    def samples_needed(self, dkl_tol: float = 0.1) -> np.ndarray:
        """[K, nb]: values needed in each class for the excess DKL to drop to ``dkl_tol`` bit."""
        return np.array([r.samples_needed(dkl_tol) for r in self.results])

    def margin(self, dkl_tol: float = 0.1) -> np.ndarray:
        """[K, nb]: values available in the class divided by the values needed
        (>= 1: the class supports the bin width)."""
        return self.n_values[:, None] / self.samples_needed(dkl_tol)

    def recommend(self, dkl_tol: float = 0.1, classes=None) -> float | None:
        """Smallest bin width for which the excess DKL at ``fraction`` of the data is within ``dkl_tol`` in *all*
        the given classes (1-based; default all analysed classes, e.g. ``range(1, 6)`` for the first five)."""
        sel = np.ones(len(self.n_values), dtype=bool) if classes is None else np.isin(self.class_index, list(classes))
        worst = self.dkl_at_target[sel].max(axis=0)
        ok = np.flatnonzero(worst <= dkl_tol)
        return None if ok.size == 0 else float(self.binwidths[ok].min())


def classwise_cross_entropy(
    x, y, z, lag_dist: float, n_classes: int, binwidths, *, fraction: float = 0.2, n_rep: int = 200,
    strategy: str = "random", prob_min: float | str | None = "her", seed: int = 0, min_values: int = 20,
) -> ClasswiseResult:
    """Run ``cross_entropy_convergence`` on the delta-z of each of the first ``n_classes`` distance classes.

    In HER the delta-z PMF of a class is built from *all* its pairs, so what matters is whether the pairs of that
    class support the bin width. The classes inside the range are used for prediction, the first (closest) ones
    carry most of the weight and have the fewest pairs; beyond the range the PMF of the full data set is used and
    the bin width matters much less. Classes with fewer than ``min_values`` values are skipped (NaN).
    ``prob_min="her"`` uses HER's own smoothing (``her_prob_min``); a float is used as given.
    """
    prob_min = her_prob_min(x, y) if isinstance(prob_min, str) else prob_min
    groups = delta_z_by_class(x, y, z, lag_dist, n_classes)
    binwidths = np.atleast_1d(np.asarray(binwidths, dtype=float))
    n_values = np.array([len(g) for g in groups])
    n_target = np.maximum(10, np.round(fraction * n_values).astype(int))
    dkl = np.full((n_classes, len(binwidths)), np.nan)
    results = []
    for k, g in enumerate(groups):
        if len(g) < min_values:
            results.append(None)
            continue
        sizes = np.unique(np.concatenate((np.round(np.geomspace(10, len(g), 12)).astype(int), [n_target[k]])))
        r = cross_entropy_convergence(g, binwidths, sizes, n_rep=n_rep, strategy=strategy,
                                      prob_min=prob_min, seed=seed + k)
        results.append(r)
        dkl[k] = r.dkl_mean[:, int(np.flatnonzero(sizes == n_target[k])[0])]
    keep = np.array([r is not None for r in results])
    return ClasswiseResult(binwidths, np.arange(1, n_classes + 1)[keep], n_values[keep], fraction, n_target[keep],
                           dkl[keep], [r for r in results if r is not None])


def limiting_class(x, y, z, lag_dist: float, n_first: int = 10, *, min_values: int = 20):
    """The class with the fewest delta-z values among the first ``n_first`` distance classes, and its data.

    Those classes are the ones used (and weighted most) in the predictions; the class with the fewest pairs is
    the one that supports the finest bin width the least, so it is the one to check. Classes with fewer than
    ``min_values`` values are ignored. Returns ``(class_index (1-based), delta_z values of that class)``.
    """
    groups = delta_z_by_class(x, y, z, lag_dist, n_first)
    sizes = np.array([len(g) if len(g) >= min_values else np.iinfo(np.int64).max for g in groups])
    k = int(np.argmin(sizes))
    if sizes[k] == np.iinfo(np.int64).max:
        raise ValueError(f"none of the first {n_first} classes has {min_values} values")
    return k + 1, groups[k]


def limiting_class_cross_entropy(x, y, z, lag_dist: float, binwidths, *, n_first: int = 10, sample_sizes=None,
                                 n_rep: int = 500, strategy: str = "random", prob_min: float | str | None = "her",
                                 seed: int = 0):
    """``cross_entropy_convergence`` on the class with the fewest pairs among the first ``n_first`` classes.

    Returns ``(class_index, BinwidthResult)``. Use ``result.recommend(n_target, dkl_tol)`` and
    ``result.samples_needed(dkl_tol)`` as for any single class.
    """
    prob_min = her_prob_min(x, y) if isinstance(prob_min, str) else prob_min
    k, data = limiting_class(x, y, z, lag_dist, n_first)
    return k, cross_entropy_convergence(data, binwidths, sample_sizes, n_rep=n_rep, strategy=strategy,
                                        prob_min=prob_min, seed=seed)


# ----------------------------------------------------------------------------------------------
# excess at the ACTUAL class size (smooth bootstrap): no subsample fraction needed
# ----------------------------------------------------------------------------------------------
def excess_at_size(data, binwidths, *, n: int | None = None, bandwidth: float | None = None, n_rep: int = 300,
                   pseudo_count: float = 1.0, prob_min: float | None = None, symmetrize: bool = False,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Expected excess DKL(P || q) [bit] of a PMF q built from ``n`` values (default: all ``len(data)``) with respect
    to the underlying distribution P, for each bin width. Returns ``(mean, std)`` over ``n_rep`` synthetic data sets.

    The underlying distribution is unknown, so it is replaced by a Gaussian kernel density estimate of ``data``
    (Silverman bandwidth unless ``bandwidth`` is given), restricted to the bins of HER (symmetric, covering the data).
    ``n_rep`` synthetic samples of ``n`` values are drawn from it (smooth bootstrap), binned like HER with the
    smoothing of ``prob_min`` (or ``pseudo_count`` counts) and DKL(P || q) is averaged.

    ``symmetrize=True`` is for delta-z of point pairs: pass each pair once (``both_triangles=False``); every sample
    is then mirrored (z_i - z_j and z_j - z_i, as HER builds its PMFs), which halves the independent information of
    the class. Counting (i, j) and (j, i) as two independent values would underestimate the excess.

    Accuracy: with independent values from a known Gaussian the estimate follows the true excess (N = 200 and 1000,
    bin widths 0.1-0.8 sd) on average, not for every single data set. For pairs of simulated Gaussian random fields
    (200 points, exponential covariance, classes 1 and 3) it is a LOWER bound: about 90 % of the true excess for
    class 1 and 50-70 % for class 3 at coarse bins, because the spatial dependence between pairs of one field is not
    reproduced. Treat the result as optimistic, mainly for the finer bin widths near the tolerance.

    Unlike the subsample experiment of ``cross_entropy_convergence``, which compares a PMF of n values with that of
    N values, this estimates the loss of the PMF that is actually used (built from all N values of the class).
    """
    from scipy.special import ndtr

    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]
    N = data.size
    n = N if n is None else int(n)
    kde_data = np.concatenate((data, -data)) if symmetrize else data
    mult = 2 if symmetrize else 1
    binwidths = np.atleast_1d(np.asarray(binwidths, dtype=float))
    sd = kde_data.std(ddof=1)
    iqr = np.subtract(*np.percentile(kde_data, [75, 25]))
    h = bandwidth if bandwidth is not None else 1.06 * min(sd, iqr / 1.34 if iqr > 0 else sd) * kde_data.size ** (-0.2)
    rng = np.random.default_rng(seed)
    mean, std = np.empty(len(binwidths)), np.empty(len(binwidths))
    for i, bw in enumerate(binwidths):
        edges = symmetric_edges(data, bw)
        B = len(edges) - 1
        # bin probabilities of the kernel density estimate, truncated to the HER bins
        P = np.zeros(B)
        for chunk in np.array_split(kde_data, max(1, kde_data.size // 2000)):  # keep the [bins x data] matrix small
            cdf = ndtr((edges[:, None] - chunk[None, :]) / h)
            P += np.diff(cdf, axis=0).sum(axis=1)
        P /= P.sum()
        pos = P > 0
        pc = pseudo_count if prob_min is None else mult * n * prob_min
        vals = np.empty(n_rep)
        for r in range(n_rep):
            x = np.empty(0)
            while x.size < n:  # smooth bootstrap, rejecting draws outside the bins
                cand = rng.choice(data, n) + rng.normal(0, h, n)
                x = np.concatenate((x, cand[(cand >= edges[0]) & (cand < edges[-1])]))
            x = x[:n]
            if symmetrize:
                x = np.concatenate((x, -x))
            c = np.histogram(x, bins=edges)[0]
            q = (c + pc) / (mult * n + pc * B)
            vals[r] = np.sum(P[pos] * np.log2(P[pos] / q[pos]))
        mean[i], std[i] = vals.mean(), (vals.std(ddof=1) if n_rep > 1 else 0.0)
    return mean, std


@dataclass
class ActualSizeResult:
    """Excess DKL at the actual class size, for several distance classes and bin widths."""

    binwidths: np.ndarray  # [nb]
    class_index: np.ndarray  # [K] 1-based
    n_values: np.ndarray  # [K]
    excess: np.ndarray  # [K, nb] mean DKL(P || q) [bit]
    excess_std: np.ndarray  # [K, nb] over the synthetic data sets
    limiting_class: int  # class with the fewest values among those analysed

    def recommend(self, dkl_tol: float = 0.1, classes=None) -> float | None:
        """Smallest bin width whose excess is within ``dkl_tol`` bit in all the given classes (default: all)."""
        sel = np.ones(len(self.n_values), dtype=bool) if classes is None else np.isin(self.class_index, list(classes))
        ok = np.flatnonzero(self.excess[sel].max(axis=0) <= dkl_tol)
        return None if ok.size == 0 else float(self.binwidths[ok].min())

    def finest_per_class(self, dkl_tol: float = 0.1) -> np.ndarray:
        """[K] smallest bin width within the tolerance for each class (nan if none)."""
        return np.array([self.binwidths[e <= dkl_tol].min() if (e <= dkl_tol).any() else np.nan for e in self.excess])


def actual_size_excess(x, y, z, lag_dist: float, binwidths, *, n_first: int = 10, only_limiting: bool = False,
                       n_rep: int = 300, prob_min: float | str | None = "her", seed: int = 0,
                       min_values: int = 20) -> ActualSizeResult:
    """``excess_at_size`` for each of the first ``n_first`` distance classes (or only for the class with the fewest
    pairs, ``only_limiting=True``). The result's ``limiting_class`` is that class in both cases.

    Pairs are handled as HER does: each pair once, mirrored into (i, j) and (j, i) (``symmetrize=True``), with
    HER's smoothing (``prob_min="her"``, see ``her_prob_min``). ``n_values`` are the ordered pairs of each class.
    The estimate is a lower bound for spatially dependent pairs, see ``excess_at_size``."""
    prob_min = her_prob_min(x, y) if isinstance(prob_min, str) else prob_min
    groups = delta_z_by_class(x, y, z, lag_dist, n_first, both_triangles=False)
    idx = [k for k, g in enumerate(groups) if 2 * len(g) >= min_values]
    if not idx:
        raise ValueError(f"none of the first {n_first} classes has {min_values} values")
    limiting = min(idx, key=lambda k: len(groups[k]))
    run = [limiting] if only_limiting else idx
    binwidths = np.atleast_1d(np.asarray(binwidths, dtype=float))
    ex, sd = np.empty((len(run), len(binwidths))), np.empty((len(run), len(binwidths)))
    for r, k in enumerate(run):
        ex[r], sd[r] = excess_at_size(groups[k], binwidths, n_rep=n_rep, prob_min=prob_min, symmetrize=True,
                                      seed=seed + k)
    return ActualSizeResult(binwidths, np.array(run) + 1, np.array([2 * len(groups[k]) for k in run]), ex, sd,
                            limiting + 1)
