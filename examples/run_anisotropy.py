"""Jura lead: test for geometric anisotropy of log10 Pb.

1. directional empirical variograms (Matheron, 4 sectors of +-22.5 deg);
2. an exponential variogram with nugget is fitted to the (distance class x 8 sectors) variogram cloud, once
   isotropic (nugget, sill, range) and once with geometric anisotropy (+ anisotropy ratio and major-axis angle);
3. permutation test: the directions of the pairs are shuffled within distance classes (this destroys any
   directional structure but keeps the distance structure); the improvement of the fit from allowing
   anisotropy is compared with the permutation distribution. The pairs are not independent, so the
   p-value is approximate;
4. does it matter? ordinary kriging with the isotropic and the anisotropic model (leave-one-out on the
   calibration points and the validation points).

Angles are measured counter-clockwise from the x axis (east); the azimuth from north is 90 - angle.
Figures go to examples/output/anisotropy/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pyher import load_jura_dataset

LAG, HMAX, NSEC = 0.15, 1.8, 8
rng = np.random.default_rng(1)


def pairs(x, y, z):
    i, j = np.triu_indices(len(z), 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    return np.hypot(dx, dy), np.mod(np.arctan2(dy, dx), np.pi), 0.5 * (z[i] - z[j]) ** 2


def model(h, ang, p):
    """Exponential variogram, p = (nugget, partial sill, major range [, ratio, major-axis angle])."""
    nug, sill, rng_major = p[:3]
    if len(p) == 5:
        ratio, phi = p[3:]
        d = ang - phi
        h = h * np.hypot(np.cos(d), np.sin(d) / ratio)  # distances along the minor axis are stretched by 1/ratio
    return nug + sill * (1 - np.exp(-3 * h / rng_major))


def classes(h, ang):
    k = np.digitize(h, np.arange(0, HMAX + LAG, LAG)) - 1
    s = np.floor(ang / (np.pi / NSEC) + 0.5).astype(int) % NSEC  # sector centred on multiples of 180/NSEC deg
    return k, s


def summarize(h, ang, g, k, s):
    """Mean h, mean g, count per (distance class, sector)."""
    ok = k < int(HMAX / LAG)
    cell = k[ok] * NSEC + s[ok]
    n = np.bincount(cell, minlength=int(HMAX / LAG) * NSEC)
    mh = np.bincount(cell, weights=h[ok], minlength=n.size) / np.maximum(n, 1)
    mg = np.bincount(cell, weights=g[ok], minlength=n.size) / np.maximum(n, 1)
    sec = np.tile(np.arange(NSEC), int(HMAX / LAG)) * np.pi / NSEC
    keep = n >= 20
    return mh[keep], sec[keep], mg[keep], n[keep]


def fit(mh, sec, mg, n, anisotropic):
    top = mg.max()
    starts = [(0.4 * top, 0.6 * top, 0.8)] if not anisotropic else [
        (0.4 * top, 0.6 * top, 0.8, r, f) for r in (0.5, 0.9) for f in np.pi * np.arange(4) / 4]
    lo = [0, 0, 0.05] + ([0.15, -0.5 * np.pi] if anisotropic else [])
    hi = [top * 2, top * 3, 6.0] + ([1.0, 1.5 * np.pi] if anisotropic else [])
    best = None
    for p0 in starts:
        r = least_squares(lambda p: np.sqrt(n) * (model(mh, sec, p) - mg), p0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x.copy()
    if anisotropic:
        p[4] = np.mod(p[4], np.pi)
    return p, 2 * best.cost


def krige(x, y, z, p, xt, yt, k=30, exclude_self=False):
    """Ordinary kriging with ``k`` neighbours; returns prediction and kriging variance."""
    tree = cKDTree(np.c_[x, y])
    _, idx = tree.query(np.c_[xt, yt], k + int(exclude_self))
    idx = idx[:, int(exclude_self):]
    est, var = np.empty(len(xt)), np.empty(len(xt))
    for t in range(len(xt)):
        ii = idx[t]
        dx, dy = x[ii][:, None] - x[ii][None], y[ii][:, None] - y[ii][None]
        gm = model(np.hypot(dx, dy), np.mod(np.arctan2(dy, dx), np.pi), p)
        gm[np.diag_indices(k)] = 0.0
        A = np.zeros((k + 1, k + 1))
        A[:k, :k], A[:k, k], A[k, :k] = gm, 1.0, 1.0
        d0x, d0y = x[ii] - xt[t], y[ii] - yt[t]
        b = np.r_[model(np.hypot(d0x, d0y), np.mod(np.arctan2(d0y, d0x), np.pi), p), 1.0]
        sol = np.linalg.solve(A, b)
        est[t], var[t] = sol[:k] @ z[ii], sol @ b
    return est, var


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[2] / "HERs" / "1.0.raw data"
    out = Path(__file__).resolve().parent / "output" / "anisotropy"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_jura_dataset(data / "Jura_dataset_Pb_log10.mat")
    nperm = 299

    for label, (x, y, z) in {"calibration (259)": (ds.x_cal, ds.y_cal, ds.z_cal),
                             "all data (359)": (ds.x, ds.y, ds.z)}.items():
        h, ang, g = pairs(x, y, z)
        k, s = classes(h, ang)
        cells = summarize(h, ang, g, k, s)
        p_iso, sse_iso = fit(*cells, False)
        p_an, sse_an = fit(*cells, True)
        stat = (sse_iso - sse_an) / sse_an
        # permutation: shuffle the pair directions within distance classes
        null = np.empty(nperm)
        ang_p = ang.copy()
        for b in range(nperm):
            for c in np.unique(k):
                m = np.flatnonzero(k == c)
                ang_p[m] = ang[rng.permutation(m)]
            kp, sp = classes(h, ang_p)
            cp = summarize(h, ang_p, g, kp, sp)
            null[b] = (lambda a, b_: (a[1] - b_[1]) / b_[1])(fit(*cp, False), fit(*cp, True))
        pval = (1 + np.sum(null >= stat)) / (nperm + 1)
        print(f"\n== {label}: {len(h)} pairs, {len(cells[0])} (distance x sector) cells")
        print("isotropic     : nugget %.4f sill %.4f range %.2f km" % tuple(p_iso))
        print("anisotropic   : nugget %.4f sill %.4f major range %.2f km ratio %.2f (minor range %.2f km) "
              "major axis %.0f deg from east (azimuth %.0f deg)" % (p_an[0], p_an[1], p_an[2], p_an[3], p_an[2] * p_an[3],
                                                                 np.degrees(p_an[4]), np.mod(90 - np.degrees(p_an[4]), 180)))
        print("relative SSE reduction %.3f, permutation p = %.3f (null 95%% quantile %.3f)" % (stat, pval, np.quantile(null, 0.95)))
        if label.startswith("calibration"):
            cal = dict(h=h, ang=ang, g=g, p_iso=p_iso, p_an=p_an, null=null, stat=stat, pval=pval)

    # ---------- does it matter? kriging with both models ----------
    for name, p in (("isotropic", cal["p_iso"]), ("anisotropic", cal["p_an"])):
        loo, v_loo = krige(ds.x_cal, ds.y_cal, ds.z_cal, p, ds.x_cal, ds.y_cal, exclude_self=True)
        val, v_val = krige(ds.x_cal, ds.y_cal, ds.z_cal, p, ds.x_val, ds.y_val)
        r = lambda e: (np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)))
        print("%-11s kriging  LOO(cal) RMSE %.4f MAE %.4f | validation RMSE %.4f MAE %.4f | mean kriging std %.3f" % (
            (name,) + r(loo - ds.z_cal) + r(val - ds.z_val) + (np.sqrt(v_val).mean(),)))

    # ---------- figures ----------
    h, ang, g, p_an, p_iso = cal["h"], cal["ang"], cal["g"], cal["p_an"], cal["p_iso"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ax = axs[0]
    edges = np.arange(0, HMAX + LAG, LAG)
    for th, c in zip((0, 45, 90, 135), ("C0", "C1", "C2", "C3")):
        d = np.abs((np.degrees(ang) - th + 90) % 180 - 90)
        m = d <= 22.5
        kk = np.digitize(h[m], edges) - 1
        ok = kk < len(edges) - 1
        n = np.bincount(kk[ok], minlength=len(edges) - 1)
        gm = np.bincount(kk[ok], weights=g[m][ok], minlength=len(edges) - 1) / np.maximum(n, 1)
        hm = np.bincount(kk[ok], weights=h[m][ok], minlength=len(edges) - 1) / np.maximum(n, 1)
        sel = n >= 20
        ax.plot(hm[sel], gm[sel], "o-", c=c, label=f"{th} deg (east = 0)")
        hh = np.linspace(0.01, HMAX, 100)
        ax.plot(hh, model(hh, np.full_like(hh, np.radians(th)), p_an), "--", c=c, lw=1)
    hh = np.linspace(0.01, HMAX, 100)
    ax.plot(hh, model(hh, np.zeros_like(hh), p_iso), "k-", lw=2, label="isotropic fit")
    ax.set(xlabel="lag (km)", ylabel="semivariance of log10 Pb", title="Directional variograms (dashed: anisotropic fit)")
    ax.legend(fontsize=8)
    ax = axs[1]
    a = np.linspace(0, 2 * np.pi, 200)
    R = lambda r_, f: (r_ * np.cos(a - f) , r_ * np.sin(a - f))
    rx = p_an[2] * np.hypot(np.cos(a - p_an[4]), np.sin(a - p_an[4]) / p_an[3]) ** -1
    ax.plot(rx * np.cos(a), rx * np.sin(a), "C1", label="anisotropic fit")
    ax.plot(p_iso[2] * np.cos(a), p_iso[2] * np.sin(a), "k", label="isotropic fit")
    ax.set_aspect("equal"); ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
    ax.set(title="Range ellipse (km, x = east)", xlabel="x"); ax.legend()
    ax = axs[2]
    ax.hist(cal["null"], bins=30, color="C0", alpha=.7, label="permutation null")
    ax.axvline(cal["stat"], c="r", lw=2, label=f"observed (p = {cal['pval']:.2f})")
    ax.set(xlabel="relative SSE reduction from anisotropy", title="Permutation test"); ax.legend()
    fig.savefig(out / "anisotropy.png", dpi=110)
    print("figure:", out / "anisotropy.png")
