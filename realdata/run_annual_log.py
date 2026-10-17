"""Annual 1999--2018 log-yield, manuscript seven-method, ten-fold comparison.

The supplied CSVs are restored into data/input/. Two 365-day functional inputs are projected
onto a common 30-basis space for all methods. Only the response is log-transformed:
y = log(Yield). Each year's outer ten folds are independent and reproducible.
"""

from __future__ import annotations

import os

for name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "data" / "input"
CACHE = ROOT.parent / "generated" / "realdata_cache"
sys.path.insert(0, str(ROOT))

import data as basis_data  # noqa: E402

# Set the common functional basis before importing neural_methods, which
# imports ANALYSIS_BASIS_DIM as a module-level constant.
BASIS_DIM = 30
basis_data.ANALYSIS_BASIS_DIM = BASIS_DIM
basis_data.ANALYSIS_INTERIOR_KNOTS = tuple(
    np.linspace(basis_data.T_GRID[0], basis_data.T_GRID[-1], BASIS_DIM - 2)[1:-1]
)

import real_data  # noqa: E402
import local_methods  # noqa: E402
import neural_methods  # noqa: E402


METHODS = ("GWR", "SGWR", "FNN", "GNNWR", "SANNWR", "GWFRN", "SGWFRN")
PROTOCOL = "annual-logyield-M30-random10fold-v1"
N_OUTER = 10
N_INNER = 5
RESULTS = ROOT.parent / "generated" / "realdata"
YEARS = tuple(range(1999, 2019))
# Fixed per-method offsets; changing displayed method order must not change seeds.
SEEDS = {"GWR": 10_000_000, "SGWR": 10_000_002, "FNN": 10_000_003,
         "GNNWR": 10_000_004, "SANNWR": 10_000_005,
         "GWFRN": 10_000_007, "SGWFRN": 10_000_008}


def load_annual(year: int):
    annual = real_data.load_year(CACHE, year)
    if len(annual.y) < N_OUTER or not np.all(np.isfinite(annual.y)):
        raise AssertionError(f"Invalid annual source data for {year}")
    if np.any(annual.y <= 0):
        raise ValueError("All yield observations must be positive for log response")
    if annual.x.shape != (len(annual.y), 2, 365):
        raise AssertionError("The two 365-day functional inputs are missing")
    projected = basis_data.bspline_coefficients(annual.x)
    if projected.shape != (len(annual.y), 2 * BASIS_DIM):
        raise AssertionError("M=30 basis projection failed")
    basis = basis_data.analysis_bspline_basis()
    gram = basis.T @ (basis * basis_data.QUAD_WEIGHTS[:, None])
    if not np.allclose(gram, np.eye(BASIS_DIM), atol=1e-8):
        raise AssertionError("The 30 B-splines are not orthonormal")
    return replace(
        annual,
        y=np.log(annual.y),
        z=annual.z[:, :1].copy(),
        bspline_coef=projected,
    )


def fold_labels(n: int, year: int) -> np.ndarray:
    permutation = np.random.default_rng(year).permutation(n)
    labels = np.full(n, -1, dtype=np.int64)
    for fold, indices in enumerate(np.array_split(permutation, N_OUTER)):
        labels[indices] = fold
    if not np.all(labels >= 0) or np.bincount(labels).max() - np.bincount(labels).min() > 1:
        raise AssertionError("Invalid ten-fold partition")
    return labels


def make_fold(annual, labels: np.ndarray, fold: int) -> real_data.ModelInput:
    train = np.flatnonzero(labels != fold).astype(np.int64)
    test = np.flatnonzero(labels == fold).astype(np.int64)
    inner = real_data.spatial_folds(
        annual.coords[train], annual.county_id[train], N_INNER
    )
    cv_folds = tuple(
        (train[inner != j], train[inner == j]) for j in range(N_INNER)
    )
    return real_data.ModelInput(
        coords=annual.coords,
        x=annual.x,
        bspline_coef=annual.bspline_coef,
        z=annual.z,
        y=annual.y,
        model_idx=train,
        test_idx=test,
        cv_folds=cv_folds,
    )




def fit_one(annual, labels: np.ndarray, fold: int, method: str) -> dict:
    model_input = make_fold(annual, labels, fold)
    # Only the scoring/collection layer retains held-out responses. All model
    # entry points receive NaNs at test rows, including nested validation fits.
    training_y = model_input.y.copy()
    training_y[model_input.test_idx] = np.nan
    training_y.setflags(write=False)
    model_input = replace(model_input, y=training_y)
    started = time.perf_counter()
    if method == "GWR":
        result = local_methods.fit_gwr(model_input, pilot=False)
    elif method == "SGWR":
        result = local_methods.fit_sgwr(model_input, pilot=False)
    else:
        batch_size = 50
        if method in {"GNNWR", "SANNWR"}:
            # SWNN has batch normalisation. Avoid a terminal singleton batch
            # in both its tuning split and its outer-training refit.
            tuning_size = len(model_input.cv_folds[0][0])
            refit_size = len(model_input.model_idx)
            while tuning_size % batch_size == 1 or refit_size % batch_size == 1:
                batch_size -= 1
            if batch_size < 2:
                raise RuntimeError("No admissible SWNN mini-batch size")
        neural_methods.PROPOSED_REPRESENTATION_DIM = 32
        neural_methods.GWFRN_IMPLEMENTATION_VERSION = (
            "encoder_only_32d_m30_days1to365_normalised_curvature_headnormalised_v1"
        )
        config = neural_methods.NeuralConfig(
            epochs=3000, batch_size=batch_size, learning_rate=0.005,
            patience=100, device="cpu", threads=1, pilot=False,
        )
        function = {
            "FNN": neural_methods.fit_fnn,
            "GNNWR": neural_methods.fit_gnnwr,
            "SANNWR": neural_methods.fit_sannwr,
            "GWFRN": neural_methods.fit_gwfrn,
            "SGWFRN": neural_methods.fit_sgwfrn,
        }[method]
        result = function(model_input, config, SEEDS[method] + 100_003 * fold)
    yhat = np.asarray(result["test_prediction"], dtype=np.float64).reshape(-1)
    if yhat.shape != (len(model_input.test_idx),) or not np.all(np.isfinite(yhat)):
        raise RuntimeError(f"Invalid {method} prediction in fold {fold}")
    implementation = result.get("implementation", "")
    if method in {"GWFRN", "SGWFRN"}:
        implementation = (
            "30-basis functional-input encoder, 32-dimensional shared "
            "representation, location-indexed heads, joint staged spatial CV, "
            "functional-curvature penalty, conditional out-of-sample heads"
        )
    return {
        "protocol": PROTOCOL,
        "year": annual.year,
        "fold": fold,
        "method": method,
        "fold_signature": hashlib.sha256(labels.tobytes()).hexdigest(),
        "county_id": [str(x) for x in annual.county_id[model_input.test_idx]],
        "test_indices": model_input.test_idx.tolist(),
        "log_yhat": yhat.tolist(),
        "n_train": len(model_input.model_idx),
        "n_test": len(model_input.test_idx),
        "selected": result.get("selected", {}),
        "implementation": implementation,
        "basis_dimension_per_curve": BASIS_DIM,
        "representation_dimension": 32 if method in {"GWFRN", "SGWFRN"} else None,
        "mini_batch_size": batch_size if method in {"GNNWR", "SANNWR"} else (
            50 if method in {"FNN", "GWFRN", "SGWFRN"} else None
        ),
        "fit_seconds": float(result.get("seconds", time.perf_counter() - started)),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        suffix=".tmp", delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
        suffix=".tmp", delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, path)




def collect(annual, labels: np.ndarray, output: Path) -> dict[str, dict]:
    signature = hashlib.sha256(labels.tobytes()).hexdigest()
    complete: dict[str, np.ndarray] = {}
    runtimes: dict[str, list[float]] = {}
    for method in METHODS:
        predictions = np.full(len(annual.y), np.nan)
        seconds = []
        for fold in range(N_OUTER):
            path = output / "records" / f"fold{fold:02d}_{method}.json"
            if not path.is_file():
                break
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = np.flatnonzero(labels == fold)
            if (
                record.get("protocol") != PROTOCOL
                or record.get("year") != annual.year
                or record.get("fold_signature") != signature
                or record.get("test_indices") != expected.tolist()
                or record.get("county_id") != [str(x) for x in annual.county_id[expected]]
            ):
                raise AssertionError(f"Invalid saved fold record: {path}")
            predictions[expected] = record["log_yhat"]
            seconds.append(float(record["fit_seconds"]))
        if len(seconds) == N_OUTER:
            if not np.all(np.isfinite(predictions)):
                raise AssertionError(f"Incomplete predictions for {method}")
            complete[method] = predictions
            runtimes[method] = seconds
    if not complete:
        return {}
    rows = []
    for i in range(len(annual.y)):
        row = {
            "Year": annual.year,
            "CountyI": str(annual.county_id[i]),
            "State": str(annual.state[i]),
            "County": str(annual.county[i]),
            "latitude": float(annual.latitude[i]),
            "longitude": float(annual.longitude[i]),
            "source_row": int(annual.source_row[i]),
            "outer_fold": int(labels[i]),
            "Yield_original": float(np.exp(annual.y[i])),
            "log_Yield": float(annual.y[i]),
        }
        for method, yhat in complete.items():
            row[f"{method}_log_yhat"] = float(yhat[i])
            row[f"{method}_log_error_yhat_minus_y"] = float(yhat[i] - annual.y[i])
        rows.append(row)
    metrics = []
    for method, prediction in complete.items():
        errors = prediction - annual.y
        metrics.append({
            "Year": annual.year,
            "method": method,
            "n_counties": len(annual.y),
            "RMSE_logY": float(np.sqrt(np.mean(errors ** 2))),
            "MAE_logY": float(np.mean(np.abs(errors))),
            "mean_fit_seconds_per_fold": float(np.mean(runtimes[method])),
            "total_fit_seconds": float(np.sum(runtimes[method])),
        })
    _atomic_csv(output / "oof_predictions_logY.csv", rows)
    _atomic_csv(output / "oof_metrics_logY.csv", metrics)
    _atomic_json(output / "protocol.json", {
        "protocol": PROTOCOL,
        "year": annual.year,
        "n_counties": len(annual.y),
        "response": "natural log of positive Yield",
        "evaluation_scale": "natural log Yield",
        "signed_error": "predicted log Yield minus observed log Yield",
        "outer_folds": N_OUTER,
        "outer_seed": annual.year,
        "inner_spatial_folds": N_INNER,
        "functional_basis": "quadrature-orthonormalised cubic B-spline",
        "basis_dimension_per_curve_for_every_method": BASIS_DIM,
        "scalar_input": "avgPRCP only",
        "functional_inputs": ["X1(t)", "X2(t)"],
        "completed_methods": list(complete),
        "neural_max_epochs": 3000,
        "neural_patience_where_applicable": 100,
        "proposed_representation_dimension": 32,
        "raw_annual_cache": f"generated/realdata_cache/annual_{annual.year}.npz",
    })
    return {row["method"]: row for row in metrics}


def _record_is_valid(path: Path, annual, labels: np.ndarray, fold: int, method: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = np.flatnonzero(labels == fold)
        prediction = np.asarray(value.get("log_yhat", []), dtype=np.float64)
        return (
            value.get("protocol") == PROTOCOL
            and value.get("year") == annual.year
            and value.get("method") == method
            and value.get("fold") == fold
            and value.get("fold_signature") == hashlib.sha256(labels.tobytes()).hexdigest()
            and value.get("test_indices") == expected.tolist()
            and value.get("county_id") == [str(x) for x in annual.county_id[expected]]
            and value.get("n_train") == len(annual.y) - len(expected)
            and prediction.shape == (len(expected),)
            and bool(np.all(np.isfinite(prediction)))
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _fit_job(job: tuple) -> dict:
    """Top-level worker: compatible with Windows process spawning."""

    annual, labels, fold, method = job
    return fit_one(annual, labels, fold, method)


def _save_result(result: dict, annual, output: Path) -> None:
    method = result["method"]
    fold = result["fold"]
    path = output / "records" / f"fold{fold:02d}_{method}.json"
    _atomic_json(path, result)
    test = np.asarray(result["test_indices"], dtype=np.int64)
    error = np.asarray(result["log_yhat"], dtype=np.float64) - annual.y[test]
    print(
        f"done year={annual.year} fold={fold} method={method} "
        f"seconds={result['fit_seconds']:.1f} "
        f"log_RMSE={np.sqrt(np.mean(error ** 2)):.4f}",
        flush=True,
    )


def _save_failure(annual, output: Path, fold: int, method: str, exc: BaseException) -> None:
    path = output / "failures" / f"fold{fold:02d}_{method}.json"
    _atomic_json(path, {
        "protocol": PROTOCOL,
        "year": annual.year,
        "fold": fold,
        "method": method,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "retry_on_next_run": True,
    })
    print(f"FAILED year={annual.year} fold={fold} method={method}: {exc}", flush=True)


def summarise_all_years() -> None:
    annual_rows: list[dict] = []
    for year in YEARS:
        path = RESULTS / "by_year" / str(year) / "oof_metrics_logY.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                annual_rows.extend(dict(row) for row in csv.DictReader(stream))
    if not annual_rows:
        return
    _atomic_csv(RESULTS / "annual_metrics_logY.csv", annual_rows)
    summary = []
    for method in METHODS:
        rows = [row for row in annual_rows if row["method"] == method]
        if not rows:
            continue
        rmse = np.array([float(row["RMSE_logY"]) for row in rows])
        mae = np.array([float(row["MAE_logY"]) for row in rows])
        summary.append({
            "method": method,
            "n_complete_years": len(rows),
            "complete_1999_2018": len(rows) == len(YEARS),
            "mean_annual_RMSE_logY": float(rmse.mean()),
            "sd_annual_RMSE_logY": float(rmse.std(ddof=1)) if len(rows) > 1 else "",
            "mean_annual_MAE_logY": float(mae.mean()),
            "sd_annual_MAE_logY": float(mae.std(ddof=1)) if len(rows) > 1 else "",
        })
    _atomic_csv(RESULTS / "summary_by_method_logY.csv", summary)


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_OUTER)))
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent method-fold jobs in parallel; use conservatively on Windows")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.output.resolve()
    if not set(args.years).issubset(YEARS) or len(set(args.years)) != len(args.years):
        parser.error("--years must be distinct values from 1999 to 2018")
    if not set(args.folds).issubset(range(N_OUTER)) or len(set(args.folds)) != len(args.folds):
        parser.error("--folds must be distinct values from 0 to 9")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.prepare_only and args.collect_only:
        parser.error("--prepare-only and --collect-only are mutually exclusive")

    real_data.prepare_data(INPUT, CACHE)
    if args.prepare_only:
        print("Prepared annual caches in generated/realdata_cache")
        return

    failed = 0
    for year in args.years:
        annual = load_annual(year)
        labels = fold_labels(len(annual.y), year)
        output = RESULTS / "by_year" / str(year)
        jobs = []
        if not args.collect_only:
            for method in args.methods:
                for fold in args.folds:
                    path = output / "records" / f"fold{fold:02d}_{method}.json"
                    if _record_is_valid(path, annual, labels, fold, method):
                        print(f"reuse year={year} fold={fold} method={method}", flush=True)
                    else:
                        jobs.append((annual, labels, fold, method))
        print(f"year={year} n={len(annual.y)} pending_jobs={len(jobs)}", flush=True)
        if args.workers == 1:
            for job in jobs:
                _, _, fold, method = job
                print(f"fit year={year} fold={fold} method={method}", flush=True)
                try:
                    _save_result(_fit_job(job), annual, output)
                except Exception as exc:
                    failed += 1
                    _save_failure(annual, output, fold, method, exc)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_fit_job, job): (job[2], job[3]) for job in jobs}
                for future in as_completed(futures):
                    fold, method = futures[future]
                    try:
                        _save_result(future.result(), annual, output)
                    except Exception as exc:
                        failed += 1
                        _save_failure(annual, output, fold, method, exc)
        complete = collect(annual, labels, output)
        print(f"year={year} complete_methods={list(complete)}", flush=True)
        summarise_all_years()
    if failed:
        raise SystemExit(f"{failed} job(s) failed; see results/by_year/<year>/failures and rerun")


if __name__ == "__main__":
    main()
