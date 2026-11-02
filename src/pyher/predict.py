"""HER3 (z PMF prediction) and HERs4 (sequential simulation).

Ports of ``f_her3_predict.m`` and ``f_her4_HERs.m`` / ``f_montecarlosim.m``.
"""

import warnings

import numpy as np

from pyher.core import monte_carlo_draw
from pyher.infogram import Infogram
from pyher.pmfs import bins_shift, window_pmf
from pyher.weights import Weights, classify_distances


class Predictor:
    """Predicts z PMFs from a (growing) set of known points using the fitted HER model."""

    def __init__(self, x_cal, y_cal, z_cal, ig: Infogram, w: Weights, *, reserve: int = 0):
        self.ig, self.w, self.cfg = ig, w, ig.cfg
        if w.nugget != ig.cfg.nugget or w.objective != ig.cfg.objective:
            raise ValueError(
                f"the weights were optimised with nugget={w.nugget}, objective={w.objective!r}, but the config says "
                f"nugget={ig.cfg.nugget}, objective={ig.cfg.objective!r}; fit the weights again after changing it")
        x_cal, y_cal, z_cal = (np.asarray(a, dtype=float).ravel() for a in (x_cal, y_cal, z_cal))
        n = len(x_cal)
        cap = n + reserve
        self._x = np.empty(cap)
        self._y = np.empty(cap)
        self._z = np.empty(cap)
        self._x[:n], self._y[:n], self._z[:n] = x_cal, y_cal, z_cal
        self.n = n
        self._pmf_cache: dict[tuple[int, int], np.ndarray] = {}
        self._lost = 0.0

    # -- data handling ----------------------------------------------------------------------
    @property
    def x(self):
        return self._x[: self.n]

    @property
    def y(self):
        return self._y[: self.n]

    @property
    def z(self):
        return self._z[: self.n]

    def add_point(self, x: float, y: float, z: float) -> None:
        if self.n == len(self._x):
            for name in ("_x", "_y", "_z"):
                setattr(self, name, np.concatenate((getattr(self, name), np.empty(max(16, self.n)))))
        self._x[self.n], self._y[self.n], self._z[self.n] = x, y, z
        self.n += 1

    # -- helpers ----------------------------------------------------------------------------
    def _neighbour_pmf(self, bs: int, c: int) -> np.ndarray:
        key = (bs, c)
        pmf = self._pmf_cache.get(key)
        if pmf is None:
            pmf, lost_l, lost_r = window_pmf(self.ig.pmf_diff_z_by_class_obs_range_shift[c], bs,
                                             self.ig)
            self._lost += lost_l + lost_r
            self._pmf_cache[key] = pmf
        return pmf

    def warn_if_truncated(self) -> None:
        if self._lost > 0:
            warnings.warn("Part of the delta-z PMFs fell outside the range of edges_z and was "
                          "discarded (summed mass over all neighbour PMFs: %.3g)." % self._lost)
            self._lost = 0.0

    def predict(self, x_t, y_t, *, chunk: int | None = None, warn: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """z PMFs of the targets: array ``[T, n_bins_z]`` and the 0-based indices of the targets
        whose neighbours all lie beyond the information range."""
        ig, w, cfg = self.ig, self.w, self.cfg
        x_t = np.asarray(x_t, dtype=float).ravel()
        y_t = np.asarray(y_t, dtype=float).ravel()
        T, R, nn = len(x_t), ig.n_classes_range, min(cfg.n_neighbor, self.n)  # fewer data than neighbours
        edges = ig.edges_distance_classes_range
        if chunk is None:  # keep the [chunk, n_neighbor, n_bins_z] work array around 150 MB
            chunk = int(max(16, min(512, 2e7 // (nn * ig.n_bins_z))))
        out = np.empty((T, ig.n_bins_z))
        counts_last = np.empty(T, dtype=int)
        counts_all = np.empty(T, dtype=int)
        bs_all = bins_shift(self.z, ig)

        for start in range(0, T, chunk):
            sl = slice(start, min(start + chunk, T))
            dist = np.sqrt((self.x[:, None] - x_t[None, sl]) ** 2
                           + (self.y[:, None] - y_t[None, sl]) ** 2)  # [n, t]
            cls = classify_distances(dist, edges, prediction=True)
            counts_last[sl] = (cls == R - 1).sum(axis=0)
            counts_all[sl] = cls.shape[0]
            idx_nn = np.argsort(dist, axis=0, kind="stable")[:nn]  # [nn, t]
            d_nn = np.take_along_axis(dist, idx_nn, axis=0)
            c_nn = np.take_along_axis(cls, idx_nn, axis=0)

            if cfg.nugget:
                # class weight + slope * (distance - lower class edge)
                w_or_cont = w.best_w_OR[c_nn] + w.w_OR_slope[c_nn] * (d_nn - edges[c_nn])
                w_and_cont = w.best_w_AND[c_nn] + w.w_AND_slope[c_nn] * (d_nn - edges[c_nn])
            else:
                # f_her3_predict_NOnugget.m: the class weight is reached at the *upper* class edge and
                # interpolated linearly towards the previous class; inside the first class the weight
                # grows with 1/distance, and is 1e16 at distance 0 (no nugget at sampled locations)
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv = edges[1] / d_nn
                    w_or_cont = np.where(
                        d_nn == 0, 1e16, np.where(
                            d_nn <= edges[1], w.best_w_OR[c_nn] * inv,
                            w.best_w_OR[c_nn] + w.w_OR_slope[c_nn] * (d_nn - edges[c_nn + 1])))
                    w_and_cont = np.where(
                        d_nn == 0, 1e16, np.where(
                            d_nn <= edges[1], w.best_w_AND[c_nn] * inv,
                            w.best_w_AND[c_nn] + w.w_AND_slope[c_nn] * (d_nn - edges[c_nn + 1])))
            w_or_norm = w_or_cont / w_or_cont.sum(axis=0, keepdims=True)

            # z PMF given each neighbour: [t, nn, n_bins_z]
            G = np.empty((dist.shape[1], nn, ig.n_bins_z))
            for t in range(dist.shape[1]):
                for k in range(nn):
                    G[t, k] = self._neighbour_pmf(int(bs_all[idx_nn[k, t]]), int(c_nn[k, t]))

            or_pmf = np.einsum("kt,tkb->tb", w_or_norm, G)
            or_pmf /= or_pmf.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                logG = np.log(G)
            lp_and = np.einsum("kt,tkb->tb", w_and_cont, logG)
            and_pmf = np.exp(lp_and - lp_and.max(axis=1, keepdims=True))
            and_pmf /= and_pmf.sum(axis=1, keepdims=True)

            kind = cfg.aggregation_type
            if kind == "and":
                out[sl] = and_pmf
            elif kind == "or":
                out[sl] = or_pmf
            elif kind == "andor":
                with np.errstate(divide="ignore"):
                    lp = w.best_alpha * np.log(and_pmf) + w.best_beta * np.log(or_pmf)
                e = np.exp(lp - lp.max(axis=1, keepdims=True))
                out[sl] = e / e.sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"unknown aggregation_type {kind!r}")

        if warn:
            self.warn_if_truncated()
        return out, np.flatnonzero(counts_last == counts_all)


def simulate(
    predictor_args: tuple, x_target, y_target, *, rng=None, perm=None, p_values=None,
    tol: float = 10e-6,
) -> tuple[np.ndarray, dict]:
    """HERs4: one sequential simulation (random path through the targets, Monte Carlo draw from the
    predicted PMF, simulated value added to the data).

    ``predictor_args = (x_cal, y_cal, z_cal, ig, weights)``. To replay a MATLAB run pass its
    ``perm`` (0-based visiting order) and ``p_values`` (indexed by target; NaN where the target
    honours a data point). Returns the realisation and a record of the random numbers used.
    """
    x_cal, y_cal, z_cal, ig, w = predictor_args
    x_target = np.asarray(x_target, dtype=float).ravel()
    y_target = np.asarray(y_target, dtype=float).ravel()
    T = len(x_target)
    rng = np.random.default_rng(rng)
    perm = rng.permutation(T) if perm is None else np.asarray(perm)
    pred = Predictor(x_cal, y_cal, z_cal, ig, w, reserve=T)
    z_real = np.full(T, np.nan)
    used = np.full(T, np.nan)
    for t in perm:
        d = np.sqrt((pred.x - x_target[t]) ** 2 + (pred.y - y_target[t]) ** 2)
        j = int(np.argmin(d))
        if d[j] <= tol:  # honour the calibration data
            z_real[t] = pred.z[j]
            continue
        pmf, _ = pred.predict(x_target[t: t + 1], y_target[t: t + 1], warn=False)
        p = rng.random() if p_values is None else p_values[t]
        used[t] = p
        z_real[t] = monte_carlo_draw(pmf[0], ig.edges_z, p)
        pred.add_point(x_target[t], y_target[t], z_real[t])
    pred.warn_if_truncated()
    return z_real, {"perm": perm, "p_value": used}
