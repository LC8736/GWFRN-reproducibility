# Reproduction scope and outstanding checks

## What this deposit provides

An executable, source-input-to-finding workflow for the retained seven methods;
fixed seeds/configurations; explicit relative paths; provenance and software
notices; full fitting, prediction, aggregation and polygon-rendering programs;
and small reported numerical table targets used **only** for comparison.
No archived fitted predictions or pilot/test scripts substitute for a refit.

The simulation implementation is the target-head/conditional-readout protocol
associated with the retained 200-replication summaries. The empirical
implementation is the separate annual M=30 protocol associated with the retained
application, not a newly substituted training algorithm. Production arithmetic
and seed offsets are preserved when removing unused routines and renaming the
SANNWR display label. Outer responses are masked before estimator entry.

## What must not be claimed

- Preparing this deposit did not itself rerun 200 repetitions of all 18
  simulation settings or all 1,400 annual method-fold fits. Passing preparation,
  import and reconstruction checks is not verification of the full experiment.
- A previous single-fold empirical retraining check did not reproduce the
  archived predictions pointwise, and its cause was not resolved. Exact
  prediction-level empirical retraining replication is therefore **not
  certified**. Refit outputs must be compared with the numerical targets rather
  than assumed identical. No target predictions are silently substituted.
- The station-download/curve-centring segment preceding the supplied
  preprocessed temperature matrix is unavailable (see DATA.md). This release
  cannot certify reconstruction from unprocessed station data.

## Manuscript/protocol alignment requiring attention

The numerical Tables 2–4 are associated with a simulation implementation that
uses known target heads during joint estimation and conditional ridge readout.
The retained method exposition describes fitting-location heads (R=n) and
conditional new-location prediction without fully distinguishing that simulation
variant. In addition, portions of the simulation benchmark prose describe an
earlier SGWR spatial-grid selector and an earlier 64/32 FNN/Adam protocol. The
result-generating simulation software instead uses FastSGWR's native ALPHA/AICc
selector, the FuncNN-default port (64/64 sigmoid/linear), and the author GNNWR
optimiser; PARAMETERS.md records these precisely.

These are reporting differences, not a reason to alter software silently until
it matches stale prose. This deposit preserves the result-associated
implementation. Before claiming full compliance with a result-level
reproducibility requirement, the manuscript description and actual protocol
should be reconciled, and the empirical retraining discrepancy should be
resolved or its scope clearly disclosed. Neither missing evidence nor protocol
differences are hidden by removing author information.

## Checks included in the execution workflow

Input SHA-256 and row alignment; county/year uniqueness and NOAA conversion;
fold disjointness; held-out response masking; saved county/fold correspondence;
full cohort completeness; metric reconstruction from individual predictions;
sample-SD aggregation; common map scale and numerical source exports. Failures
raise errors rather than trigger alternative estimators or favourable-seed
replacement. The distribution's CHECKSUMS.sha256 inventories its actual files.
