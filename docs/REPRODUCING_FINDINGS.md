# Finding-by-finding reproduction instructions

Run commands at the repository root with the Python environment installed as
described in the root README. Every listed path is relative. The command driver
also works from an arbitrary current directory. The default workflow never
reads numerical targets from `reference/`.

## Tables 2, 3 and 4: full Monte Carlo workflow

### 1. Inspect the production design and parameters

```powershell
python tools/show_protocol.py
python simulation/run_simulation.py --config simulation/config.json --preflight
```

`simulation/config.json` fixes two grids, m=30,40 (n=900,1600), three response
models, three error laws, seven methods and 200 paired repetitions. Replication
IDs are 1000–1199. `simulation/data.py` generates the inputs; no saved summary
or external simulation file is read. The complete generator is retained for the
reported models only. Optional inspection export:

```powershell
python tools/generate_simulation_data.py
```

The export contains observed/latent curves, true conditional means, response,
scalar predictors, coordinates, model/test indices and all inner folds. It is
not needed before fitting and requires additional disk space.

### 2. Refit all seven methods

```powershell
python reproduce.py simulation --sim-workers 100
```

Reduce `--sim-workers` on smaller machines. Parallelism is across complete
replications, not nested folds/candidates. Each replication uses independent
predictor, measurement-error, response-error and split streams. Predictor draws
and partitions are paired across the compared response settings/methods. The
source generator records the exact stream construction.

For each generated dataset, 10 of 100 contiguous blocks are held out. The other
90 form five inner spatial splits, each with 72 fitting and 18 validation blocks.
All methods use the same saved indices. Fitting routines receive read-only NaNs
instead of test responses. Scaling, response standardisation and distance
scales are estimated on the applicable fitting rows. The proposed encoders use
M=15 and widths 64/32, with no input-to-head skip connection. Known target heads
participate in joint training, using **only fitting responses**; final held-out
predictions use the conditional ridge readout. Each head's weights have mean one
over fitting rows and the data loss is divided by the actual number of heads.

GWFRN/SGWFRN screen full (b, alpha, smoothing) combinations with continued
100/200/300-epoch budgets, 3/3/5 folds and 4/2/1 retained candidates. Conditional
validation MSE is evaluated every ten epochs. Validation responses select
parameters and a common epoch but do not enter gradients. Selected parameters
and the selected epoch are refitted from a new initialisation on the 90 blocks.
Baseline selection procedures are listed separately in PARAMETERS.md; they are
not silently forced onto the proposed parameter grid.

### 3. Reconstruct and verify prediction errors

The driver performs this automatically; it can be repeated without refitting:

```powershell
python tools/build_findings.py --part simulation
```

It requires all 25,200 completed method records and their held-out prediction
files. Each cell must contain exactly the prescribed 200 run IDs. For each
replication, it recomputes

`RMSE = sqrt(mean((yhat - y_test)^2))` and
`MAE = mean(abs(yhat - y_test))`.

The test counts are 90 and 160. Only after replication-level metrics are formed
does it take their arithmetic means and sample SDs (`ddof=1`). It does **not**
pool all residuals and take a single RMSE. A failed or missing method is not
replaced with another seed or omitted from a reported mean.

### 4. Locate each numbered table and the associated findings

| Finding | Model/config key | Output |
|---|---|---|
| Table 2 | Model I, `linear` | `generated/findings/table2.csv`, `.md`, `.tex` |
| Table 3 | Model II, `nonlinear_kappa0` | `generated/findings/table3.csv`, `.md`, `.tex` |
| Table 4 | Model III, `nonlinear_kappa1` | `generated/findings/table4.csv`, `.md`, `.tex` |

Each table contains n=900 then n=1600, the seven methods and Normal, Student's t
and bimodal RMSE/MAE columns. Bold and underline identify the smallest and second
smallest **unrounded mean**, respectively; SDs are never marked. The underlying
CSV, rather than formatting, determines ranks. `reference/tables2_4.csv` contains
the manuscript's rounded numerical targets only.

The statements that the proposed methods are competitive in the linear setting,
lead the nonlinear comparisons, and that the similarity-guided version performs
favourably under interactions can be checked directly against the corresponding
rows/columns. SDs are not used to claim statistical significance.

Intermediate evidence created by the run:

- `generated/simulation/predictions/*.npz`: held-out IDs, coordinates, y, yhat,
  and signed prediction error yhat−y;
- `metrics_per_replication.csv`: RMSE, MAE and seconds for each method/run;
- `summary_metrics.csv`, `summary_timing.csv`: setting-level means/SDs;
- `splits/`: the paired block and inner-fold indices;
- `artifacts/`: selected parameters, validation curves and stage audits;
- `manifest.json`, `coverage.csv`, `failures.csv`: environment, source hashes
  and completion checks. These are generated locally, not uploaded old logs.

## Table 5: independent annual log-yield analyses

### 1. Restore and validate the shared source inputs

```powershell
python tools/prepare_data.py
python realdata/run_annual_log.py --prepare-only
```

The first command verifies the location/regression file hashes, concatenates the
temperature chunks in manifest order, validates county-year keys and checks
`avgPRCP = annual_inches * 25.4 / days_in_year` against the bundled NOAA-derived
county-year source. The second pairs functional rows with regression rows before
filtering/sorting; joins locations by `CountyI`; retains 1999–2018; and writes
validated annual caches. See DATA.md for field definitions and upstream limits.

The expected sample is 6,152 observed county-years from 403 distinct counties;
annual counts vary from 255 to 345. No missing county-year is imputed. Yield must
be positive and finite. The analysed response is **natural log Yield**, not
centred Yield or raw-scale yield. `avgPRCP` is the only scalar predictor; Area
is not used. The two 365-day curves are already preprocessed source-study inputs.

### 2. Fit each year with outer ten-fold prediction

```powershell
python reproduce.py realdata --real-workers 4
```

Each year's permutation is seeded by that calendar year. Every observed county
is held out exactly once. Models are fitted separately by year; neither past nor
future responses are pooled. Within each outer fitting sample, five deterministic
coordinate-only spatial folds select tuning parameters. The same outer folds
are shared by all seven methods. Every method uses M=30, not the simulation's
M=15. Fit-only predictor transformations are applied unchanged to held-out rows.
Neural response standardisation is undone before assessing **log-yield** errors;
predictions are not exponentiated for these metrics.

The proposed methods' grids are b={0.25,0.5,1,1.5}, smoothing={0.001,0.01,0.1};
GWFRN fixes alpha=1 and SGWFRN uses {0.3,0.6,0.9,1}. Screening fits 3/4/5 folds for
20/60/200 epochs, restarting each stage. The final fit uses 200 epochs and only
fitting-location heads. Conditional output heads use relative ridge 0.1. This
empirical protocol is deliberately not replaced by the later simulation
continuation/target-head protocol. PARAMETERS.md enumerates baseline settings.

### 3. Verify the complete county-fold predictions and calculate Table 5

```powershell
python realdata/verify_results.py --require-all
python tools/build_findings.py --part realdata
```

The checks require all 20×10×7 method-fold records, matching county IDs,
held-out indices and one OOF prediction per observed county-year. The annual
RMSE and MAE are computed from **all counties' OOF predictions in that year**,
not by taking an unweighted average of ten fold metrics. Table 5 then takes the
arithmetic mean and sample SD across the **20 annual metrics**. Thus the
parenthetical SD is between years, not between the 200 folds.

Outputs: `generated/findings/annual_metrics.csv`, `table5.csv`, `table5.md` and
`table5.tex`. `reference/table5.csv` contains the reported full-precision summary
for comparison, but is not read by the reconstruction command.

## Figure 2: county observation coverage

```powershell
python tools/prepare_data.py
python figures/plot_mean_absolute_errors.py --coverage-only
```

No model fitting is needed for coverage. Count distinct observed years during
1999–2018 by `CountyI`, join `State.ANSI + County.ANSI` to Census GEOID, then draw
county polygons from the bundled 2018 Census boundaries. The 403 counts have a
5–20-year range. Source: `generated/data/county_coverage.csv`. Outputs:
`generated/figures/figure2_county_year_coverage.pdf`, `.svg` and `.png`.

## Figure 3 and the county/state error findings

First complete the Table 5 modelling workflow. Then:

```powershell
python tools/build_findings.py --part realdata
python figures/plot_mean_absolute_errors.py
```

For each county-year first compute `abs(log_Yield - log_yhat)`. Then average
these absolute values over **that county's observed years**, giving A_i. It is
not the absolute value of a signed average, and counties with missing years are
not filled with zeros. All six map panels use the same colour limits, with the
pooled 98th percentile as the upper cap and a colourbar extension indicating
saturation. Grey counties are outside the observed sample.

Panel rows: GWFRN/SGWFRN; GNNWR/SANNWR; GWR/SGWR. FNN remains in Table 5 but is not
a spatial map panel. Outputs: `generated/figures/figure3_county_mean_absolute_error.*`
and `figure3_audit.json`; numerical source: `generated/findings/county_mae.csv`.

Each panel mean and the reported 0.127/0.125 are arithmetic means of A_i across
403 counties. They are independently exported to `equal_county_mae.csv`. State
comparisons use the same A_i, averaging counties equally within each state;
the output is `state_mae.csv`. The Kansas/coverage discussion is an
interpretation, not a separately estimated causal effect. Coverage and boundary
position can be inspected from Figure 2 and the source geography; the scripts do
not turn that interpretation into a hypothesis test.

Table 5 weights **years** equally; Figure 3 weights **counties** equally after
county-specific temporal averaging. With incomplete temporal coverage these
MAE means are not algebraically identical. Both are intentionally exported.

## Structural material and other quantities

- Table 1: inspect each method's documented model structure in PARAMETERS.md
  and the cited software/source files. It contains no Monte Carlo estimates.
- Figure 1: conceptual shared encoder/location-head architecture, not a data
  plot. Encoder widths and the absence of a skip connection can be verified in
  `simulation/functional_core.py` and `realdata/neural_methods.py`.
- Method times: measured from method preprocessing/selection through final fit
  and prediction; DGP, queue time, scoring and report writing are excluded.
  Means/SDs are in `summary_timing.csv` for simulations. Real-data JSON records
  contain `fit_seconds`; no claim of hardware-independent speed is made.
- Data dimensions and annual/county coverage: `generated/data/input_audit.json`
  and `annual_sample_sizes.csv`, reconstructed directly from shared source rows.
- Numerical targets do not certify retraining replication. See the explicit
  execution and upstream-source limitations in REPRODUCIBILITY_STATUS.md.
