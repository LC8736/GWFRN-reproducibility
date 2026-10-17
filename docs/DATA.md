# Data dictionary and provenance

## What is shared

The frozen source-study inputs are shared here with permission to redistribute.
They are the starting data for the complete analysis workflow, not model
predictions. No synthetic replacements, imputed responses or selectively chosen
counties are substituted during packaging.

| File | Contents and use |
|---|---|
| `data/input/regdat_1.csv` | 6,340 county-year rows, 1999–2020; Year, State, County, CountyI, Yield, avgPRCP, Area |
| `data/input/loc_info.csv` | 403 county records; CountyI, coordinates and state/county ANSI identifiers for map joins |
| `data/input/fundat_parts/*.gz` | consecutive byte chunks restoring the 6,340×730 `fundat.csv` temperature matrix |
| `data/input/manifest.json` | chunk order, bytes and SHA-256 hashes; hashes of the other two source inputs |
| `data/source/noaa_county_annual_precip.csv` | county/year keys, NOAA county code, annual precipitation in inches and mean mm/day |
| `data/geography/cb_2018_us_county_5m.*` | complete Census shapefile components for the county maps |

`fundat.csv` has no identifiers. Its row r corresponds to row r of regdat_1.csv
**before** either file is sorted or filtered. V1–V365 are supplied centred daily
maximum temperatures; V366–V730 are centred daily minimum temperatures. Do not
sort the temperature file independently. Location joins use CountyI, not a row
number or an assumed consecutive county code. The input manifest ensures byte-
identical restoration of the temperature matrix.

The analysed subset is 1999–2018: 6,152 observed county-years, 403 unique counties
in Illinois, Indiana, Iowa, Kansas and Missouri. There are 255–345 counties per
year, and 5–20 observed years per county. Missing county-years remain absent.
Rows in 2019–2020 are preserved in the source inputs to retain the original row
alignment, but are not analysed. No data are borrowed across years.

## Variables and transformations

- **Yield:** corn yield, bushels per acre; the analysis response is natural log
  Yield. Nonpositive/nonfinite values fail validation instead of being silently
  removed or offset. `Yield.c`, if encountered in another source object, is not
  used by this release.
- **X1(t), X2(t):** source-study preprocessed maximum/minimum temperature curves
  on days 1–365. The shared matrix is already centred; original physical-scale
  station records cannot be recovered from it. This release does not apply an
  invented reverse transformation. Model-level projection and fitting-sample
  scaling are fully implemented in `realdata/data.py` and `neural_methods.py`.
- **avgPRCP:** mean daily precipitation in mm/day, computed from the included
  annual NOAA inches as `inches * 25.4 / days_in_year`. Leap years use 366 in this
  conversion. `prepare_data.py` checks every county-year value and key.
- **Area:** retained in the original source CSV for integrity, but not used as a
  scalar covariate in any reported fit.
- **Coordinates:** source latitude and positive-west longitude; the loader sets
  longitude to `-abs(longitude)` and creates local equirectangular kilometres.
- **Neural response standardisation:** fitted only on fitting rows, reversed
  before assessment to the **log-yield** scale. No exponentiation for Table 5
  or Figure 3. Predictor scaling also uses only fitting-sample statistics.

## Original sources and scope of reconstruction

The assembled county sample follows Park, Li and Li, *Spatially Varying Deep
Functional Neural Network*, DOI: https://doi.org/10.1093/jrsssc/qlag023.
Underlying agency sources are USDA NASS Quick Stats
(https://quickstats.nass.usda.gov/) and NOAA NCEI Climate Data Online
(https://www.ncei.noaa.gov/cdo-web/). County polygons are the US Census Bureau
2018 cartographic boundaries, `cb_2018_us_county_5m`:
https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip.

The supplied CSVs and precipitation source are preserved rather than re-querying
mutable agency databases. The original station-level retrieval, source-study
curve centring code, source download date and a complete raw-station provenance
log were not supplied. Therefore this deposit reproduces **source-study analysis
inputs → annual preparation → model fitting → predictions → findings**, not
**raw station archive → source-study centred curves**. That missing upstream
segment is explicitly unresolved; no script is invented to suggest otherwise.

Simulation inputs are generated entirely by the included code and have no such
external-data dependency. The exported truth/latent curves are not supplied to
fitted estimators. See REPRODUCING_FINDINGS.md for their complete generation and
assessment workflow.

## Access and rights

The depositor confirmed permission to share the assembled empirical data. That
confirmation is not presented as a new CC0/CC-BY licence over third-party data.
Underlying agency data and Census boundaries retain their applicable source
terms. Third-party source software is attributed and its licence files are
included. No invented accession, data DOI, download date or author identity is
placed in this anonymous review deposit.
