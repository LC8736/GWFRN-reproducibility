# Software calls and complete fitting settings

This describes the implemented analyses, not a claim that all comparators are
unchanged official software. Production fitting needs no parameter entry in a
GUI. All inputs are loaded from relative repository paths. Numerical safeguards
and library defaults are part of the implementation, not additional searched
hyperparameters.

## Common inputs and numerical conventions

| Analysis | Functional grid | Cubic B-splines per curve | Model inputs |
|---|---|---:|---|
| Simulation | 100 equally spaced points on [0,1] | 15 | two noisy curves, three scalars, coordinates |
| Application | days 1–365 | 30 | two supplied temperature curves, avgPRCP, coordinates |

The open cubic spaces have M−4 equally spaced interior knots. A symmetric
inverse square root of the trapezoidal Gram matrix orthonormalises the basis.
All compared methods retain the same curve information within an analysis.
Projection coefficients and functional inner products coincide numerically in
that common orthonormal basis. Predictor centring/scaling is fitted on fitting
rows and reused unchanged for validation/test rows. The shared encoder has no
raw-feature skip connection.

Roughness uses integrated squared second derivatives, trace normalisation of
the roughness matrix and a coefficient-norm-relative penalty with fixed
denominator floor 1e−24. The conditional ridge factor is 0.1, with unpenalised
intercept and a hidden-Gram trace adjustment. There is no ridge-parameter search.
Distances and other numerical safeguards are directly visible in each analysis'
`data.py` and representation/neural modules.

## Simulation (Tables 2–4)

The authoritative proposed-method grid and stage schedule are in
`simulation/config.json`:

- b: 0.1, 0.2, 0.3, 0.5;
- alpha: 1 for GWFRN; 0.25, 0.5, 0.75 for SGWFRN;
- smoothing: 0.001, 0.01, 0.1;
- hidden widths 64/32, ReLU; Adam 0.005; mini-batch size 50;
- cumulative budgets 100/200/300 epochs; 3/3/5 spatial folds; retain 4/2/1;
- evaluate conditional validation predictions every 10 epochs; continue
  surviving network/Adam/random-generator states; final new initialisation at
  the minimum pooled-validation-MSE common epoch;
- fitting plus known target heads are trained jointly. Only fitting responses
  appear in residuals. Final predictions use frozen-encoder conditional heads;
- one numerical thread per worker; CPU; fixed seeds recorded per method/run.

| Method | Executed implementation | Selection / optimisation settings |
|---|---|---|
| GWR | `mgwr.gwr.GWR`, package 2.2.1, fixed Gaussian kernel, intercept, n_jobs=1; external-location `predict` | five-fold spatial MSE over positive modelling-distance quantiles 0.10,0.25,0.50,0.75,0.90; validation WLS has 1e−9 trace-scaled numerical safeguard; not a new statistical ridge grid |
| SGWR | vendored FastSGWR `ALPHA` selector plus held-out WLS adapter | native data-dependent bounds and golden-section/divide-and-conquer AICc search; adaptive bisquare geography, row-normalised Gaussian attribute kernel and native mixture; no proposed b/alpha grid; no added prediction ridge |
| FNN | disclosed PyTorch port of public FuncNN `fnn.fit` defaults | hidden 64/64 with sigmoid then linear, linear output; Xavier initialisation; Adam 0.001, eps=1e−7; batch 32; maximum 100 epochs; patience 15; native callback does not restore best weights; common spatial validation replaces random validation; full modelling-sample refit at stopping epoch |
| GNNWR | vendored author GNNWR class and SWNN definitions, NumPy OLS backend | package automatic architecture; Adagrad 0.1, dropout 0.2, weight decay 0.001, batch 32; maximum 3000 epochs; validation patience 100; final refit at selected epoch |
| SANNWR | paper-based SAPDNN–SWNN implementation | SAPDNN 2–3–1 and SWNN 50–20; Adam 0.02, batch 16; maximum 3000 epochs; validation patience 100; full modelling-sample refit |

`simulation/native_linear.py` and `native_neural.py` are the dispatchers. The
GNNWR wrapper executes the author classes without eager plotting imports and
uses NumPy unpenalised OLS instead of an equivalent sklearn backend. FNN and
SANNWR are not advertised as original author executables. SGWR's AICc is its
baseline software selector, not the proposed methods' selection criterion.

## Application (Table 5; Figures 2–3)

The fixed implementation is `realdata/`; its settings are printed from
`config/empirical_protocol.json`. That manifest documents the fixed analysis;
it is not substituted for the simulation configuration.

- 20 independent annual fits, 1999–2018; response natural log Yield;
- outer random ten-fold county partitions, seeded by year; each county is tested
  once in its observed year; no inter-year pooling;
- five inner spatial folds by recursive longest-axis coordinate bisection;
- widths 64/32, ReLU, Adam 0.005, batch 50, relative conditional ridge 0.1;
- proposed b={0.25,0.5,1,1.5}, smoothing={0.001,0.01,0.1}; GWFRN alpha=1;
  SGWFRN alpha={0.3,0.6,0.9,1};
- independent screening fits at 20/60/200 epochs using 3/4/5 folds, retaining
  4/2/1 candidates for GWFRN and 12/4/1 for SGWFRN; final fit 200 epochs;
- only fitting-location heads are jointly trained in this retained empirical
  protocol; new-location outputs are conditional ridge predictions;
- GWR and SGWR: five-fold spatial MSE; distance quantiles 0.10/0.25/0.50/0.75/0.90;
  SGWR mixture candidates 0.25/0.5/0.75/1; fixed Gaussian geographical kernel;
- FNN: ReLU 64/32; Adam 0.005; batch 50; maximum 3000, patience 100;
- GNNWR/SANNWR: paper/package coefficient-weighting networks; Adam 0.02;
  batch 50 (reduced only if needed to avoid a final singleton batch in batch
  normalisation); maximum 3000, patience 100;
- neural benchmarks select epochs on the first spatial validation fold, then
  refit on the full outer fitting sample. Response scaling is undone to log
  Yield before metrics, not to raw Yield.

Fixed method seed offsets are explicit in `realdata/run_annual_log.py`.
Dropping an unreported method or rearranging table columns does not renumber
these seeds. County sorting is numeric by CountyI before annual outer-fold
permutation. Longitude's west convention is converted to negative values;
local equirectangular coordinates are measured in kilometres.

## Qualitative Table 1 and conceptual Figure 1

GWR/SGWR retain a local linear predictor specification. GNNWR/SANNWR learn
weights on regression coefficients, rather than the shared roughness-regularised
functional encoder. FNN has a common global output. GWFRN/SGWFRN use functional
inner products, the shared encoder and local prediction heads; SGWFRN also uses
covariate proximity. These code structures underlie Table 1 and Figure 1; they
are not separate numerical estimates. Neural coefficient weighting may itself
induce nonlinear predictions, as the manuscript's table note explains.
