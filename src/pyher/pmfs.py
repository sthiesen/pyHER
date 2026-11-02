"""z PMF that one neighbour contributes to a target.

A neighbour with value ``z_i`` at distance class ``c`` says: the target is ``z_i + delta_z`` with
``delta_z`` distributed like the class-``c`` delta-z PMF. The delta-z PMF lives on a fine grid
(``binwidth_shift``); it is placed at ``z_i`` on the fine grid of the z axis and then summed up
into the coarser z bins (``binwidth_z``). Mass falling outside ``edges_z`` is discarded and the
rest renormalised (as in ``f_her3_predict.m``).
"""

import numpy as np

from pyher.infogram import Infogram


def matlab_round(x):
    """MATLAB ``round``: halves away from zero (NumPy rounds halves to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def bins_shift(z, ig: Infogram):
    """Offset (in fine bins) of the delta-z PMF placed at ``z`` relative to the start of ``edges_z``."""
    return matlab_round((ig.edges_diff_z_shift[0] + np.asarray(z) - ig.edges_z[0])
                        / ig.binwidth_shift).astype(int)


def window_pmf(pmf_shift_row: np.ndarray, bs: int, ig: Infogram) -> tuple[np.ndarray, float, float]:
    """z PMF for one neighbour.

    ``pmf_shift_row`` is a row of ``pmf_diff_z_by_class_obs_range_shift`` (element 0 is the extra
    zero-mass bin placed in front, used to pad both sides). Returns the PMF over the z bins and
    the probability mass discarded on the left and on the right of ``edges_z``.
    """
    n_per, nbs, n_z = ig.n_bins_z_per_bin_shift, ig.n_bins_shift, ig.n_bins_z
    diff_n = n_z * n_per - nbs  # number of fine padding bins needed on both sides together
    trunc_l = max(0, -bs)
    trunc_r = max(0, bs - diff_n)
    seq = np.concatenate((
        np.full(max(bs, 0), pmf_shift_row[0]),
        pmf_shift_row[1 + trunc_l: nbs + 1 - trunc_r],
        np.full(max(diff_n - bs, 0), pmf_shift_row[0]),
    ))
    # MATLAB reshape is column-major: consecutive n_per fine bins form one z bin
    pmf = seq.reshape(n_z, n_per).sum(axis=1)
    lost_l = pmf_shift_row[1: 1 + trunc_l].sum()
    lost_r = pmf_shift_row[nbs + 1 - trunc_r: nbs + 1].sum()
    return pmf / pmf.sum(), lost_l, lost_r
