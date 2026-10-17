# Step-by-step software procedures

All off-the-shelf packages are invoked programmatically. No ArcGIS, QGIS,
interactive R application, GUI model dialog, or manually entered GUI parameter
is used. Source files and parameter manifests are the authoritative record.
This document gives the equivalent complete procedures for the scripted
software, rather than suggesting nonexistent point-and-click operations.

## Step 1: create the fixed environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B tools/show_protocol.py
```

The third command prints every pinned package and its installed version, plus
the two complete protocol manifests. An installation must finish successfully
before running models. A different interpreter/package environment is not
automatically treated as equivalent to the recorded one.

## Step 2: restore/prepare the source inputs

```powershell
.\.venv\Scripts\python.exe -B reproduce.py prepare
```

Inputs: the three source CSVs (fundat reconstructed from its parts), the NOAA
precipitation source and the manifest. Parameters: analysed years 1999–2018,
CountyI join, row-wise curve alignment, positive Yield, M=30, days 1–365, no
imputation. Outputs: data audit/coverage CSVs and annual NPZ caches. Inspect
`generated/data/input_audit.json` and the console PASS message.

## Step 3: check software imports and all input parameters

```powershell
.\.venv\Scripts\python.exe -B reproduce.py preflight
```

This checks imports of PySAL GWR, the author GNNWR source, FastSGWR's selector,
the installed neural network definitions and the empirical CLI. It does not
fit the 200-replication study. The current parameter values are in PARAMETERS.md,
`simulation/config.json`, `config/empirical_protocol.json`, and printed output.

## Step 4: run the simulation packages through their adapters

```powershell
.\.venv\Scripts\python.exe -u -B reproduce.py simulation --sim-workers 100
```

Each fit calls these modules, with all parameters already specified:

1. GWR: `simulation/native_linear.py` → `legacy_gwr.fit_gwr` →
   `mgwr.gwr.GWR(... fixed=True, kernel='gaussian', constant=True, n_jobs=1)`;
   select the five data-dependent quantile candidates using the saved folds;
   call fitted model `predict` on held-out coordinates/features.
2. SGWR: the same dispatcher → vendored `FastSGWR/serial_sgwr/alpha.py:ALPHA`;
   call `.fit()` using its native data-dependent bounds/AICc search; form its
   geographical/attribute mixture and external weighted least-squares prediction.
3. GNNWR: `simulation/native_neural.py:author_gnnwr` → unchanged author
   GNNWR and SWNN definitions; use its automatic architecture and defaults
   documented in PARAMETERS.md; select epochs on the first common validation
   fold; refit the outer modelling sample, then predict.
4. FNN: the disclosed `fit_fnn` port, with FuncNN's inspected R source retained
   under upstream; defaults explicitly listed in PARAMETERS.md.
5. SANNWR: the disclosed SAPDNN–SWNN reproduction, with its architecture,
   optimiser, batch, epoch cap and patience explicitly listed.
6. GWFRN/SGWFRN: `train_representation.fit_representation`, staged candidate
   screening/continuation and conditional ridge readout using config.json.

Successful completion gives 25,200 method records and no missing cohort cells.
The table builder verifies individual predictions rather than trusting only the
summary rows. Do not use incomplete outputs to replace reported tables.

## Step 5: run the empirical analysis

```powershell
.\.venv\Scripts\python.exe -u -B reproduce.py realdata --real-workers 4
```

Parameters: the fixed annual protocol in the empirical manifest; all seven
methods; years 1999–2018; ten outer folds; five spatial inner folds; M=30;
log response; avgPRCP only; conditional ridge 0.1. GWR/SGWR use their empirical
spatial-CV adapters. Neural methods use the separate empirical implementation
and the settings enumerated in PARAMETERS.md. MGWR and GWANN are not dispatched.

## Step 6: reconstruct and inspect findings

```powershell
.\.venv\Scripts\python.exe -B reproduce.py findings
```

The commands reconstruct numbered tables, county and state assessment metrics,
then draw both maps directly from numeric outputs/polygons. Inspect the CSV/MD
tables before importing their TEX equivalents. Maps are exported as PDF, SVG
and PNG, without raster-panel compositing.

## Screenshots / execution evidence

The requested screenshot evidence should show the six procedures above, with
the displayed commands, parameter printout and actual PASS/completion outputs.
No GUI parameter dialog exists for this workflow. `EXECUTION_CHECKS.txt` records
the input/preflight checks actually completed while preparing this deposit;
it is not a screenshot or a claim that the full experiment was rerun.

**Actual software screenshots have not been supplied in this deposit.** The
packaging environment did not expose an available browser/native screenshot
surface. No simulated terminal or fabricated GUI images are substituted. For
literal compliance with an editor's screenshot request, capture the actual
installation, protocol printout and completed fitting/reconstruction consoles
on the executing machine, then add those captures here. Crop or hide the
machine username/path without editing numerical results. Do not label a
configuration screenshot as evidence that an experiment finished.
