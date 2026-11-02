"""User-facing configuration of a HER run (the ``DEFINE`` lines of the MATLAB driver scripts)."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class HerConfig:
    """Settings of a HER run (the ``DEFINE`` lines of the MATLAB driver scripts).

    The defaults are those of the SERRA 2021/22 paper on the Jura lead data. ``z_thresh`` has no default: it is
    required for ``objective="threshold"`` (the default) and for validation, so ``HerConfig()`` alone is only
    usable with ``objective="bin"``. ``lag_dist``, ``binwidth`` and ``binwidth_z`` are in the units of the data
    and must be chosen for each data set (see ``pyher.binwidth``). The shrinkage of the OR weights is an argument
    of ``optimize_weights`` / ``run_her`` (``or_shrinkage``), not a field here.
    """

    lag_dist: float = 0.07  # width of the distance classes (same unit as x, y)
    binwidth: float = 0.015  # delta-z binwidth
    binwidth_z: float = 0.015  # binwidth of the z PMFs
    n_neighbor: int = 30  # neighbours used for prediction / alpha-beta search (paper: 30; the MATLAB scripts: 9)
    z_thresh: float | None = None  # threshold for the probability map / threshold objective
    aggregation_type: Literal["and", "or", "andor"] = "andor"
    # Objective of the weight optimisation: 'bin' = DKL of the z bin containing the truth
    # (sHER_E_03), 'threshold' = DKL of the CDF at z_thresh (sHER_E_04).
    objective: Literal["bin", "threshold"] = "threshold"
    # nugget=False (default): sHER_E_*_1_NOnugget scripts, the version used in the paper (weights reached
    # at the upper class edge, weight ~ 1/distance inside the first class, no uncertainty at sampled
    # locations); nugget=True: sHER_E_*_0 scripts (weight interpolated from the lower class edge)
    nugget: bool = False
    # alpha/beta search: use the distance-interpolated (continuous) class weights instead of the discrete
    # ones. None = as in the MATLAB scripts (only f_her2_weight, i.e. objective='bin' with nugget=True).
    continuous_alpha_beta: bool | None = None
    round_decimals: int = 5  # MATLAB scripts round z to 5 decimals "to avoid numerical problems"
    # extent of the z PMF bins: edges at z_min + factor * min(delta z) ... z_max + factor * max(delta z).
    # 3 = current f_her1_infogram.m; 1 = the version that produced the HESS 2020 results (still visible in the
    # commented lines of f_her2_weight_*.m), e.g. LR1: z edges from -9.2 to 7.6 with a bin width of 0.2.
    z_edges_factor: float = 3.0
