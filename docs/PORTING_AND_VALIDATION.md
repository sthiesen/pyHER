# Port notes and validation

How `pyher` was checked against the MATLAB code (HERs) and the two papers, where it deviates, and findings from
the port. For how to use the package see the [README](../README.md).

## Validation

The MATLAB references are `tests/golden/*.mat`, made with MATLAB R2018b (Optimization Toolbox) by the scripts in
`tests/golden_gen/` (the `fmincon` runs take 15-25 min each; `gen_golden_03.m` expects the MATLAB variable `VARIANT` (`'03_0'` or `'03_1'`) and the `gen_maps*.m`
scripts `OUTDIR` to be set before they run, see the header of each). Run the checks with

```bash
.venv/Scripts/python scripts/check.py          # ruff + all tests, about 3-3.5 min; --fast skips the slowest
```

| Stage | Compared with | Result |
| --- | --- | --- |
| entropy, aggregations, Monte Carlo draw, DKL score | MATLAB | 1e-12 |
| HER1 infogram (all fields); `infogram_entropy` (= `f_infogram_realizplot`) | MATLAB | 1e-12 |
| HER2: loss at MATLAB's weights and alpha/beta search (threshold 04_0; bin 03_0 and 03_1) | MATLAB | 1e-9 / identical |
| HER3 PMFs (nugget and NOnugget) | MATLAB | 1e-9 |
| Full 121 x 121 maps (`examples/compare_with_matlab.py`, needs MATLAB-generated map files; not a test) | MATLAB | 2e-13 |
| HERs4 realisation, replaying MATLAB's random numbers | MATLAB | 1e-9 |
| PMF statistics, RMSE/ME/MAE/NSE, DKL scores, ROC, misclassification, G | MATLAB | 1e-10 |
| SERRA paper: 20 classes up to 1.4 km, 20 294 of 66 822 pairs | paper | exact |
| HESS 2020: data statistics, model range, alpha, beta, Table 1 | paper and supplement | see below |

Limits of the comparison with MATLAB:

* The SLSQP optimiser reaches a lower leave-one-out loss than `fmincon` but different weights. For the `bin` objective
  the loss matches MATLAB 03 (5.262 vs 5.261 bit) while single OR weights differ by up to 0.06 and beta by 0.05 (03_1:
  0.45 vs 0.50).
* "Same alpha/beta as the paper" holds only at 30 neighbours (alpha 0.75, beta 0, against the paper's 0.65 / 0). With 9
  neighbours the search gives 0.65 / 0.1 against MATLAB's 0.70 / 0.

**HESS 2020 synthetic data** (`../HER_datasets`, read with `load_her_synthetic`). Parameters of the paper: class width
2, bin width 0.2, 12 neighbours, `objective="bin", nugget=True`; the weights are **optimised in Python**, not copied.

* Model range identical in 28 of 28 cases, alpha in 28 of 28, beta within 0.05 in 22 of 28 (15 identical). The beta
  optimum is flat: with our weights the paper's own (alpha, beta) pair is within 0.02 bit of our optimum in all cases.
* LR1-600, pure-AND model (Table 1, model 2): EMA 0.43, ENS 0.72, DKL 3.59 (paper 3.58), median entropy 3.75, E-type
  means of targets A-D -2.06, -1.34, -1.00, 0.66, as published (tests accept the paper values within 0.03 DKL). LR1-1500
  was a failed AND optimisation from the 1/distance start (loss stuck at 11.8 bit); several starts fixed it (alpha 1,
  beta 0.95 as in the paper).
* The pure-OR and AND/OR models of Table 1 depend on the OR weights (the paper's are smooth): unshrunk weights give
  EMA 0.47 / ENS 0.68 for pure OR, an OR shrinkage of about 3e-4 gives 0.44 / 0.71 as published but matches beta in
  fewer cases overall.
* HERs on LR1-600 (50 realisations): data honoured, marginal KS distance 0.03 to the true field, ensemble mean equals
  the E-type (r = 0.995), the infogram of the realisations follows the true field (class 1: 3.4 vs 3.85 bit; E-type map
  1.9 bit). The ensemble is somewhat under-dispersed against the analytical PMFs (central 50% / 90% intervals cover
  0.38 / 0.75 of the test values, the PMFs themselves 0.47 / 0.86).

## Findings worth knowing

* The SERRA paper was made with the **NOnugget** functions and **30 neighbours**; three of the four shipped scripts use
  9. More neighbours sharpen the AND pooling (area with entropy above 6.5 bit on the Jura map: 31 % at 9, 10 % at 30,
  0.5 % at 150). The class weights themselves do not depend on `n_neighbor`.
* With beta = 0 (SERRA paper) the OR weights do not enter the final maps at all.
* Bin width and entropy: when only `binwidth_z` is changed and the weights are kept fixed, the entropy shifts by about 1
  bit per halving and the map pattern stays (r > 0.995). If both bin widths are changed and everything is refitted, the
  median shift is only about 0.04-0.12 bit and the maps correlate less (r about 0.90-0.97 in an independent check).
  Do not read absolute entropies across different bin widths.
* The single-bin objective gives sharper but less calibrated predictions on the Jura validation set (G 0.85 vs 0.93).
* NOnugget weights can become negative for a target farther from all data than the last class edge, as in the MATLAB
  formula; the predictor is not clamped.

## Bit-level details that matter

* MATLAB `linspace` and `a:step:b` differ from NumPy by one ulp at some points (`core.matlab_linspace`,
  `core.matlab_colon`). Delta-z of rounded z sit on bin edges, so this moves pairs between bins.
* MATLAB `round` rounds halves away from zero (`pmfs.matlab_round`); arrays are column-major (`order="F"`).
* SLSQP only satisfies `w[i+1] <= w[i]` up to its tolerance; the weights are made exactly monotone, otherwise the
  interpolated weights of the NOnugget predictor can turn negative.
* Python's RNG cannot reproduce MATLAB's stream, hence the replay tests (`simulate(..., perm=, p_values=)`).

## Deviations from the MATLAB scripts

* The metrics block of `sHER_E_0304` reads the prediction grid at `idx_val`, which only works if the validation points
  are grid nodes. `run_her(..., validation=...)` predicts at the validation locations instead (for HERs, the
  realisations use them as targets).
* Empty distance classes inside the information range raise a `ValueError` (MATLAB stops there too); a point is never
  its own leave-one-out neighbour, and neither are points at exactly the same location.
* Entropy maps use the range of the data as colour limits; `viridis` replaces `parula`.
* `f_binverificationTDE.m` (bin-width selection) cannot be ported 1:1: it needs `f_histcounts_anyd` and
  `f_infomeasures_from_samples`, which are not in the repository. `pyher.binwidth` is an independent implementation of
  the idea, not a translation.

## Checking the bin width (`pyher.binwidth`)

The bin width frames every entropy and DKL number. Following Thiesen et al. (2019) the tool compares the cross entropy
`Hpq = H(p) + DKL(p || q)` between the delta-z PMF of a distance class built from all its pairs (p) and PMFs of
subsamples (q) of growing size. A bin width is supported by a class if `Hpq` is close to `H(p)` for the sample sizes
you have.

```python
from pyher.binwidth import cross_entropy_convergence, delta_z_pairs
data = delta_z_pairs(x_cal, y_cal, z_cal, lag_dist=0.07, class_index=1)      # delta-z of the first class
res = cross_entropy_convergence(data, [0.0075, 0.015, 0.03, 0.06], n_rep=500, prob_min=1 / (len(x_cal) * (len(x_cal) - 1)))
res.dkl_mean               # [bin widths, sample sizes] excess of the cross entropy over H(p), bit
res.samples_needed(0.1)    # sample size at which the excess drops to 0.1 bit, per bin width
```

Class assignment is the one of `fit_infogram` (classes `(e_i, e_i+1]` on MATLAB colon edges). The point-set functions
use HER's own smoothing by default (`prob_min="her"`: `1 / number of ordered pairs` added to every bin probability).

**Excess at the actual class size.** The subsample curves compare a PMF of n values with one of N values, a pessimistic
proxy for the PMF HER really uses, and the number depends on the subsample fraction.
`actual_size_excess(x, y, z, lag_dist, binwidths, n_first=10)` estimates the expected loss DKL(P || q) of the class PMF
built from **all** its values with a smooth bootstrap (kernel density estimate of the class, pairs mirrored as in HER,
HER smoothing). It is a **lower bound**: on simulated Gaussian random fields (200 points, exponential covariance) it
recovered roughly 90 % of the true excess in class 1 and 50-70 % in class 3 at coarse bins, because the dependence between
pairs of one field is not reproduced. `ac.limiting_class` is the class with the fewest pairs among the first 10, the
most demanding one (the closest classes carry most of the weight in the prediction).

Results for the data of the papers (`examples/binwidth_selection.py`; tolerance on the excess of the limiting class and
of all first 10 classes):

| Data | Paper bin width | Excess at the paper width (limiting class) | Smallest width within 0.2 / 0.1 / 0.05 bit |
| --- | --- | --- | --- |
| Jura log10(Pb), 259 points, lag 0.07 | 0.015 | 0.67 bit (class 2, 202 pairs) | 0.06 / 0.12 / 0.24 |
| HESS LR1, 600 points, lag 2 | 0.2 | below 0.1 bit | 0.1 / 0.2 / 0.4 |
| HESS LR1, 200 points, lag 2 | 0.2 | 0.5 bit (class 1, 42 pairs) | 0.4 / 0.8 / 1.6 |

With HER's tiny smoothing, empty bins of a class PMF cost a lot of DKL, so this criterion is strict: the Jura width and
the LR1-200 width of the papers do not pass 0.1 bit, while the papers' prediction skill evidently did not suffer. The
tolerance is a choice, the smoothing matters (a pseudo-count of 1 per bin gives much smaller excess), and we have not
tested how the skill of the final predictions changes with the bin width. Use the tool as a diagnostic of how well each
class PMF is determined, not as a rule.
