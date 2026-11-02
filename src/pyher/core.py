"""Small numerical building blocks (ports of the leaf ``f_*.m`` functions)."""

import numpy as np

_TOL = 1e-5


def entropy(pmf: np.ndarray) -> float:
    """Shannon entropy in bit (``f_entropy.m``). NaN in -> NaN out."""
    pmf = np.asarray(pmf, dtype=float).ravel()
    if np.isnan(pmf).any():
        return float("nan")
    if abs(pmf.sum() - 1) > _TOL:
        raise ValueError("Probabilities don't sum to 1.")
    p = pmf[pmf > 0]
    return float(-(p * np.log2(p)).sum())


def _check_pmfs(pmfs: np.ndarray, w: np.ndarray) -> None:
    if np.isnan(pmfs).any():
        raise ValueError("NaNs in pmfs.")
    if np.isnan(w).any():
        raise ValueError("NaNs in weights.")
    if (np.abs(pmfs.sum(axis=1) - 1) > _TOL).any():
        raise ValueError("Probabilities in pmfs don't sum to 1.")
    if pmfs.shape[0] != w.shape[0]:
        raise ValueError("# of rows in pmfs and weights must be equal.")


def linear_aggregation(pmfs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """OR combination: weighted mixture of the rows of ``pmfs`` (``f_linear_aggregation.m``)."""
    pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
    w = np.asarray(w, dtype=float).ravel()
    _check_pmfs(pmfs, w)
    if abs(w.sum() - 1) > _TOL:
        raise ValueError("Weights must sum to 1.")
    out = (pmfs * w[:, None]).sum(axis=0)
    return out / out.sum()


def loglinear_aggregation(pmfs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AND combination: weighted geometric pooling (``f_loglinear_aggregation.m``)."""
    pmfs = np.atleast_2d(np.asarray(pmfs, dtype=float))
    w = np.asarray(w, dtype=float).ravel()
    _check_pmfs(pmfs, w)
    if not pmfs.sum() > 0:
        raise ValueError("AND-combination resulted in empty set")
    with np.errstate(divide="ignore"):
        logp = (np.log(pmfs) * w[:, None]).sum(axis=0)
    out = np.exp(logp - logp.max())  # avoid underflow of very small numbers
    return out / out.sum()


def monte_carlo_draw(pmf: np.ndarray, edges_z: np.ndarray, p_value: float) -> float:
    """Draw from the piecewise-linear CDF of a binned PMF at probability ``p_value``
    (``f_montecarlosim.m`` with the random number passed in)."""
    ccdf = np.concatenate(([0.0], np.cumsum(pmf)))
    idx = int(np.argmax(ccdf >= p_value))  # first index with ccdf >= p
    if ccdf[idx] < p_value:  # p above the (rounded) total mass
        idx = len(ccdf) - 1
    idx = max(idx, 1)
    return float(np.interp(p_value, ccdf[idx - 1: idx + 1], edges_z[idx - 1: idx + 1]))


def performance_prob(z_true: np.ndarray, pmfs: np.ndarray, edges_z: np.ndarray) -> float:
    """Mean DKL (bit) between a point-mass truth and the predicted PMFs
    (``f_performance_prob.m`` for the non-cell ``PMF_true = 1`` case).

    The truth falls in bin i if ``edges[i] <= z < edges[i+1]``; values outside every bin give NaN,
    as in MATLAB.
    """
    z_true = np.asarray(z_true, dtype=float).ravel()
    pmfs = np.asarray(pmfs, dtype=float)
    edges_z = np.asarray(edges_z, dtype=float)
    bins = np.searchsorted(edges_z, z_true, side="right") - 1
    valid = (bins >= 0) & (bins < len(edges_z) - 1) & (z_true < edges_z[-1])
    prob = np.full(len(z_true), np.nan)
    prob[valid] = pmfs[np.flatnonzero(valid), bins[valid]]
    return float(np.mean(-np.log2(prob)))


def matlab_linspace(start: float, stop: float, num: int) -> np.ndarray:
    """``linspace`` evaluated like MATLAB (R2018b): ``start + k*(stop-start)/(num-1)``.

    NumPy computes ``start + k*((stop-start)/(num-1))``, which differs in the last bit for some
    points. That matters when data lie exactly on bin edges (delta-z of rounded z values often do),
    because a one-ulp edge shift moves the observation into the neighbouring histogram bin.
    """
    k = np.arange(num)
    out = start + k * (stop - start) / (num - 1)
    out[-1] = stop
    return out


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """``start:step:stop`` evaluated like MATLAB: the first half counts up from ``start``, the second
    half down from the last element (so e.g. ``0:0.01:0.5`` differs from ``k*0.01`` by one ulp at
    several points)."""
    n = int(np.floor((stop - start) / step + 1e-10))
    k = np.arange(n + 1)
    last = start + n * step
    if abs(last - stop) <= 4 * np.finfo(float).eps * max(abs(stop), abs(start), 1.0):
        last = stop
    return np.where(k <= n // 2, start + k * step, last - (n - k) * step)
