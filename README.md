Reproduction materials: GWFRN and SGWFRN
This repository contains the data, fitting code and finding-by-finding instructions for Geographically Weighted Representation Learning for Spatially Heterogeneous Regression with Functional Inputs. It is an anonymous review deposit. Author names, contact details, local computer paths, Git history, exploratory runs and intermediate experiment results are not included. Third-party software attribution and licences are retained.

Start here
Use 64-bit Python 3.12.10 on Windows, the recorded execution environment. Open PowerShell in the extracted repository folder:

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B reproduce.py prepare
.\.venv\Scripts\python.exe -B reproduce.py preflight

Alternatively, ./setup_windows.ps1 runs those four steps. Installation needs internet access; the subsequent data and modelling workflows use bundled inputs and do not download observations. Do not change package pins to obtain a preferred result. tools/show_protocol.py prints the software versions and complete parameter manifests.

Run the full experiments, then regenerate every numerical finding:

.\.venv\Scripts\python.exe -u -B reproduce.py simulation --sim-workers 100
.\.venv\Scripts\python.exe -u -B reproduce.py realdata --real-workers 4
.\.venv\Scripts\python.exe -B reproduce.py findings

These are full refits, not commands that plot stored predictions. Worker counts can be reduced using the command-line options without editing code or changing the statistical protocol. The simulation has 3,600 generated datasets and 25,200 method fits; the application has 1,400 method-year-fold jobs, many containing nested tuning fits. A full run is computationally substantial. The numerical libraries use one thread per worker. Do not start 100 processes on a small-memory computer. Run times depend on hardware and concurrent resource contention.

Commands can be issued from another working directory by supplying the path to reproduce.py; it resolves the repository root itself. No personal path or manual path replacement is required. Repeating a fitting command resumes compatible completed jobs. Do not mix outputs produced by different code/configurations.

Reproduce each reported finding
The detailed instructions, equations and output columns are in REPRODUCING_FINDINGS.md. The essential mapping is:

Finding	Source-to-output command after installation	Newly generated output
Table 2: linear model, n=900 and 1600, all three error laws	python reproduce.py simulation	generated/findings/table2.csv, .md, .tex
Table 3: nonlinear additive model, kappa=0	same production simulation command	generated/findings/table3.*
Table 4: nonlinear interaction model, kappa=1	same production simulation command	generated/findings/table4.*
Simulation RMSE/MAE means, SDs and rankings	same command; 200 repetitions per setting	per-repetition metrics, prediction NPZs, selected parameters, validation curves and Tables 2–4
Table 5: 20 annual log-yield prediction analyses	python reproduce.py realdata	generated/findings/table5.*, annual_metrics.csv
Figure 2: observed-year coverage	python tools/prepare_data.py, then python figures/plot_mean_absolute_errors.py --coverage-only	generated/data/county_coverage.csv, generated/figures/figure2_county_year_coverage.*
Figure 3: county mean absolute out-of-fold errors	python reproduce.py realdata	generated/findings/county_mae.csv, generated/figures/figure3_county_mean_absolute_error.*
403 counties; 255–345 annual observations; coverage of 5–20 years	python tools/prepare_data.py	generated/data/input_audit.json, annual_sample_sizes.csv
Equal-county MAEs (reported as 0.127 and 0.125), state comparisons	python reproduce.py realdata	generated/findings/equal_county_mae.csv, state_mae.csv
Computation times, not accuracy findings in the manuscript tables	simulation/real-data fitting commands	simulation summary_timing.csv; empirical fold records and annual metrics
Here python means the installed .venv interpreter, as in the full commands above. Table 1 is a qualitative specification comparison, not an estimated finding; see PARAMETERS.md. Figure 1 is the conceptual network architecture, not an empirical graphic. No saved image is used as an input for Figures 2 or 3.

Data workflow
Simulation: fixed seeds → noisy and latent functional predictors, scalar predictors and geographical grid → generated response → shared spatial split → fit-only preprocessing and tuning → refit → held-out predictions → RMSE/MAE per replication → arithmetic mean and sample SD over 200 repetitions. An optional python tools/generate_simulation_data.py exports all generated inputs and truth; the fitting program uses the identical generator in memory.

Application: restore the supplied temperature matrix → pair its rows with the regression source rows → join county locations by CountyI → validate the NOAA precipitation conversion and select 1999–2018 → form 20 independent annual analyses → log-transform Yield → project onto 30 orthonormalised B-splines → outer random ten-fold prediction with inner spatial tuning → return predictions to the log-yield scale → calculate annual and county-level findings → draw county polygons. No other years' responses are borrowed to fit a given year. Area is not a model input. All models exclude held-out responses from estimation and tuning.

The CSV matrix supplied by the source study is already centred/preprocessed temperature data, not raw NOAA station observations. Original station-download and curve-centring programs are not available here. This deposit supports the complete modelling workflow from the frozen, shared source-study inputs; it does not claim to recreate those inputs from raw station files. This limitation and source links are explicit in DATA.md.

Organisation
reproduce.py                 one entry point for preparation, fitting and findings
simulation/                  seven-method production Monte Carlo implementation
realdata/                    seven-method annual out-of-fold implementation
data/input/                  source-study CSVs and compressed temperature chunks
data/source/                 NOAA county-year precipitation conversion source
data/geography/              Census county boundaries for both maps
config/                      empirical parameter manifest
tools/                       input checks, DGP export and result reconstruction
figures/                     direct numerical county-map rendering
reference/                   reported numerical table targets; never model inputs
docs/                        reproduction instructions, parameters and provenance
generated/                   created locally by execution; not deposited

The temperature file is stored as consecutive independently gzip-compressed byte blocks, each below 25 MiB. prepare_data.py restores the exact 88,572,790-byte CSV and checks its SHA-256 before modelling. Do not read each part as a separate CSV. The restored duplicate and all run outputs are Git-ignored.

Interpretation and verification
Tables 2–4 report means and between-replication sample SDs. Table 5 reports equal-year means and between-year sample SDs of 20 annual OOF metrics, not SDs of 200 folds. Figure 3 first averages absolute county-year errors within each county and then weights counties equally. Its mean is not Table 5's equal-year MAE. The scripts export both aggregations and check county/fold coverage.

reference/ contains only reported numerical targets for comparison, not fitted models, input predictions or an alternative plotting shortcut. Full refits are required to verify the reported predictive results. Stochastic floating-point training need not be bitwise identical across environments. A prediction-level retraining match to the retained empirical results has not been established; see REPRODUCIBILITY_STATUS.md. A repository being executable is not, by itself, evidence that exact numerical replication has been completed.

All packages are called through scripts; there is no GIS/R/GUI point-and-click analysis. SOFTWARE_WORKFLOW.md gives the package calls, parameters, commands and recorded execution checks. Actual software screenshots remain to be added as explained there. SANNWR and some adapters are transparent reproductions, not falsely labelled official author software. See THIRD_PARTY_NOTICES.md.

