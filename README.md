# pyher: HER and HERs in Python

**HER** (histogram via entropy reduction) is a nonparametric geostatistical method: it predicts a full probability
distribution (a PMF) of a variable at every location, from the distances and value differences of the observations.
**HERs** adds sequential simulation, and HER can work with probability maps for a threshold. `pyher` is the Python
version of the MATLAB code [KIT-HYD/HERs](https://github.com/KIT-HYD/HERs), and implements the two studies:

> Thiesen, S., Vieira, D. M., Mälicke, M., Loritz, R., Wellmann, J. F., Ehret, U. (2020). HER: an information-theoretic
> alternative for geostatistics. *Hydrology and Earth System Sciences* 24, 4523.
> Loss of the weight optimisation = DKL of the **single bin** that contains the observation (`objective="bin"`).

> Thiesen, S., Ehret, U. (2021). Assessing local and spatial uncertainty with nonparametric geostatistics.
> *Stochastic Environmental Research and Risk Assessment* 36, 173.
> Loss = DKL of the **threshold-exceedance** probability (`objective="threshold"`), plus the HERs simulation.

The numerical core is checked against the MATLAB code and the papers; how is described in
[docs/PORTING_AND_VALIDATION.md](docs/PORTING_AND_VALIDATION.md).

## License agreement

`pyher` comes with ABSOLUTELY NO WARRANTY. You are welcome to modify and redistribute it within the license agreement:
the new Python code is published under the [MIT license](LICENSE).

It is an adaptation of the MATLAB HERs code, which is published under the Creative Commons
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license together with its sample data set. `pyher` follows its
algorithms and conventions, and `tests/golden/` holds outputs produced by running the original code on its data. The
attribution required by CC-BY-4.0: original method and code by Stephanie Thiesen and co-authors (KIT-HYD), see the papers
above; changes here are the translation to Python, the SciPy optimiser, extra objectives, tests and tools. The original
MATLAB code, its sample data and the golden outputs derived from them remain under CC-BY-4.0. Please cite the papers if
you use `pyher`.

## Requisites

* Python 3.11 or newer. The dependencies (numpy, scipy, h5py, pyshp) are installed automatically; matplotlib for plots.
* MATLAB is **not** needed (only to regenerate the test references in `tests/golden/`).

```bash
git clone https://github.com/sthiesen/pyHER.git
cd pyHER
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[plot]"      # Windows; use .venv/bin/python elsewhere
```

The Jura sample data used below comes with the MATLAB repository. Clone <https://github.com/KIT-HYD/HERs> **next to**
this folder (`../HERs`). The synthetic data sets of the 2020 study go in `../HER_datasets`
(<https://github.com/KIT-HYD/HER/tree/master/datasets>); they are only needed for the HESS examples and tests.

## Usage

### In one call

```python
import numpy as np
from pyher import HerConfig, load_jura_dataset
from pyher.pipeline import run_her

ds = load_jura_dataset("../HERs/1.0.raw data/Jura_dataset_Pb_log10.mat")
cfg = HerConfig(z_thresh=float(np.round(np.log10(50), 5)))        # settings of the 2021 study
res = run_her(ds.x_cal, ds.y_cal, ds.z_cal, cfg, grid=(0, 6, 0.05), nsim=1, seed=1,
              validation=(ds.x_val, ds.y_val, ds.z_val))
print("alpha, beta:", res.weights.best_alpha, res.weights.best_beta)
print("PMF map shape:", res.pmf_pred.shape)
print("validation RMSE: %.3f" % res.validation.deterministic.RMSE)
```

About 30 s. It prints `alpha, beta: 0.75 0.0`, `PMF map shape: (121, 121, 507)` and `validation RMSE: 0.187`, plus a few
harmless warnings (PMF mass outside the z range; empty classes beyond the information range).
`res.pmf_pred[iy, ix]` is the predicted PMF at a grid node, `res.z_realiz` the simulations, `res.infogram` and
`res.weights` the fitted model.

### Step by step

The same pipeline as the MATLAB script `sHER_E_04_x_Jura_struct.m`, one function per step:

```python
import numpy as np
from pyher import HerConfig, load_jura_dataset
from pyher.infogram import fit_infogram
from pyher.predict import Predictor, simulate
from pyher.stats import extract_pmf_statistics, validate
from pyher.weights import optimize_weights

# 1. Load dataset (any x, y, z vectors work; z is rounded like the MATLAB scripts do)
ds = load_jura_dataset("../HERs/1.0.raw data/Jura_dataset_Pb_log10.mat")

# 2. Define infogram and HER3 properties
cfg = HerConfig(lag_dist=0.07, binwidth=0.015, binwidth_z=0.015, n_neighbor=30, z_thresh=1.69897)
z = np.round(ds.z_cal, cfg.round_decimals)

# 3. HER1: spatial characterisation (infogram: delta-z PMF per distance class, information range)
ig = fit_infogram(ds.x_cal, ds.y_cal, z, cfg)

# 4. HER2: weights of the AND / OR aggregation and the exponents alpha, beta (leave-one-out cross validation)
w = optimize_weights(ds.x_cal, ds.y_cal, z, ig)

# 5. HER3: z PMFs at any locations
pred = Predictor(ds.x_cal, ds.y_cal, z, ig, w)
pmf, no_neighbour_idx = pred.predict(ds.x_val, ds.y_val)           # [n_locations, n_bins_z]

# 6. HERs4: one sequential simulation at the target locations
gx, gy = np.meshgrid(np.arange(0, 6.01, 0.25), np.arange(0, 6.01, 0.25))
z_sim, _ = simulate((ds.x_cal, ds.y_cal, z, ig, w), gx.ravel(), gy.ravel(), rng=1)
z_sim = z_sim.reshape(gx.shape)

# 7. PMF statistics and performance (the sHER_E_0304 script)
st = extract_pmf_statistics(pmf, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
st.entropy, st.mean, st.mode, st.prob_exceed                          # per location
val = validate(pmf, ds.z_val, z, ig.edges_z, ig.bin_centers_edges_z, cfg.z_thresh)
val.deterministic.RMSE, val.deterministic.NSE, val.DKL_score_mean, val.G
```

### Your own data

All you need are 1-D arrays `x`, `y`, `z` (calibration data) and a `HerConfig`:

| Option | Meaning |
| --- | --- |
| `lag_dist` | Width of the distance classes (same unit as x, y). Choose so that the first classes hold enough pairs |
| `binwidth` | Bin width of the delta-z PMFs of the infogram |
| `binwidth_z` | Bin width of the predicted z PMFs |
| `z_thresh` | Threshold of the exceedance probability. **Required** for `objective="threshold"` (the default) and for validation |
| `objective` | `"threshold"` (default): loss = DKL of the exceedance probability, as in the 2021 study. `"bin"`: loss = DKL of the bin containing the observation, as in the 2020 study |
| `nugget` | `False` (default, the 2021 study, scripts `_1`): the prediction honours the data (no uncertainty at sampled locations). `True` (2020 study, scripts `_0`): sampled locations keep uncertainty |
| `n_neighbor` | Neighbours used in prediction and in the alpha/beta search. Default 30 (2021 study); the 2020 study used 12 |
| `aggregation_type` | `"andor"` (default), `"and"` or `"or"`: how the neighbour PMFs are combined |
| `round_decimals` | z is rounded to this many decimals first (MATLAB scripts: 5) |

Rule of thumb: fit the first classes (`lag_dist`), then check `binwidth` with the tool below. Settings of the two studies:

| | `lag_dist` | `binwidth`, `binwidth_z` | `n_neighbor` | `objective` | `nugget` |
| --- | --- | --- | --- | --- | --- |
| 2021 study, Jura lead (log10 Pb) | 0.07 km | 0.015 | 30 | threshold | False |
| 2020 study, synthetic fields | 2 | 0.2 | 12 | bin | True |

The weight optimisation can be skipped or constrained:

```python
w = optimize_weights(x, y, z, ig, fixed_weights=(w_or, w_and), alpha_beta=(0.65, 0.0))   # given weights / exponents
w = optimize_weights(x, y, z, ig, or_shrinkage=None)       # plain optimum, see below
```

**OR-weight shrinkage.** For the threshold objective the exact optimum of the OR weights puts nearly all weight on the
first two classes. `or_shrinkage` (default `"auto"`: 0.0043 for the threshold objective, none for the bin objective)
shrinks log(OR weights) towards the 1/distance guess and gives the smooth, slowly decaying weights of the 2021 study.
`or_shrinkage=None` switches it off; a warning is issued when the OR weights collapse. The AND weights are never shrunk.
(`or_shrinkage` is an argument of `optimize_weights` / `run_her`, not part of `HerConfig`.)

### Plots

`pyher.plots` has the figures of the MATLAB `f_plot_*` functions: `plot_infogram(ig)`, `plot_weights(w, ig)`,
`plot_prediction(...)` (data, E-type and entropy maps, example PMFs), `plot_probability_map(...)`,
`plot_performance(...)`, `plot_realization(...)`. Like the MATLAB `f_plot` functions they were built for the study's data
sets; `examples/run_jura.py` shows them all.

### Checking the bin width

The bin width frames every entropy and DKL number. `pyher.binwidth` compares the delta-z PMF of a distance class with
PMFs of subsamples, for several bin widths (cross-entropy convergence, after Thiesen et al. 2019):

```python
from pyher.binwidth import actual_size_excess

ac = actual_size_excess(x, y, z, lag_dist=0.07, binwidths=[0.0075, 0.015, 0.03, 0.06, 0.12], n_first=10)
ac.excess            # [classes, bin widths]: expected excess DKL (bit) of each class PMF; a lower bound
ac.limiting_class    # the class with the fewest pairs among the first 10
ac.recommend(0.1)    # smallest bin width within 0.1 bit in all 10 classes
```

`cross_entropy_convergence` and `limiting_class_cross_entropy` give the subsample curves. Treat the result as a
diagnostic, not a rule; limits and results for the data of the two studies are in
[docs/PORTING_AND_VALIDATION.md](docs/PORTING_AND_VALIDATION.md) and `examples/binwidth_selection.py`.

## File structure

```
src/pyher/   the package
tests/       pytest; golden/ = MATLAB reference outputs, golden_gen/ = MATLAB scripts that make them
examples/    scripts, see below
docs/        port notes and validation
scripts/     check.py (ruff + pytest)
```

### Modules

| Module | MATLAB counterpart | Content |
| --- | --- | --- |
| `config.py` | `DEFINE` lines of the scripts | `HerConfig` |
| `infogram.py` | `f_her1_infogram` | HER1: `fit_infogram`; `infogram_entropy` (= `f_infogram_realizplot`) |
| `weights.py` | `f_her2_weight*` | HER2: `optimize_weights`, `search_alpha_beta` |
| `predict.py` | `f_her3_predict*`, `f_her4_HERs` | HER3 `Predictor`, HERs4 `simulate` |
| `stats.py` | `f_extract_pmf_statistics`, `f_perf*`, `f_ROC`, `f_G` | statistics and performance metrics |
| `pipeline.py` | the driver scripts | `run_her` chains the steps |
| `plots.py` | `f_plot_*` | figures |
| `binwidth.py` | `f_binverificationTDE` (idea only) | bin-width diagnostics |
| `smooth.py` | | regularisation and cross validation of the OR weights |
| `io.py`, `reference.py`, `core.py`, `pmfs.py` | | data loading, the study's reference weights, entropy and aggregation helpers |

### Examples

Run from the repository folder; figures go to `examples/output/` (not tracked).

| Script | What it does |
| --- | --- |
| `run_jura.py` | Jura lead data: HER1-HERs4, statistics, validation, figures. Flags `--bin` (single-bin objective; overwrites the threshold figures), `--no-shrinkage`, `--paper-weights` |
| `paper_figures.py`, `compare_with_matlab.py` | Paper-style figures (valid for z_thresh 1.6-1.8); MATLAB-vs-Python map comparison (needs MATLAB output files) |
| `neighbors_effect.py` | the effect of `n_neighbor` (9 / 30 / 150) |
| `smooth_or_weights.py` | smoothing and cross-validating the OR class weights |
| `reproduce_hess2020.py`, `hers_hess2020.py` | the 2020 study's synthetic data: benchmark and simulation checks |
| `binwidth_selection.py` | bin-width diagnostics for the Jura and HESS data |
| `verify_paper_her.py` | feeds the class weights and alpha/beta stored in a MATLAB results file of the 2020 study into `optimize_weights(fixed_weights=..., alpha_beta=...)` and compares the predicted map and the scores with the stored MATLAB output (needs that `.mat` file) |
| `her_nugget_vs_nonugget.py` | HER with and without nugget (`HerConfig(nugget=...)`) with the paper's weights: difference maps incl. the calibration nodes, error and predictive spread there, test / validation scores |
| `run_anisotropy.py` | geometric anisotropy of the Jura data: directional variograms, anisotropic vs isotropic fit, permutation test |

Some scripts have absolute paths of the author's machine near the top that you must change; they need `pip install -e ".[plot]"`.

### Tests

```bash
pip install -e ".[dev,plot]"
.venv/Scripts/python scripts/check.py          # ruff + all tests, about 3 min; --fast skips the slowest
```

Tests that need `../HER_datasets` are skipped if it is missing.

## Contact

Stephanie Thiesen | <https://github.com/sthiesen>
