"""Audit completed annual out-of-fold predictions and log-scale metrics."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from run_annual_log import (
    METHODS, N_OUTER, RESULTS, YEARS, _record_is_valid, fold_labels,
    load_annual,
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--require-all", action="store_true",
                        help="Fail unless all seven manuscript methods finished every requested year")
    args = parser.parse_args()
    if not set(args.years).issubset(YEARS):
        parser.error("Years must lie between 1999 and 2018")
    complete_total = 0
    for year in args.years:
        output = RESULTS / "by_year" / str(year)
        annual = load_annual(year)
        labels = fold_labels(len(annual.y), year)
        predictions_path = output / "oof_predictions_logY.csv"
        metrics_path = output / "oof_metrics_logY.csv"
        if not predictions_path.is_file() or not metrics_path.is_file():
            if args.require_all:
                raise AssertionError(f"No complete methods for {year}")
            print(f"{year}: no complete methods")
            continue
        rows = read_csv(predictions_path)
        metrics = read_csv(metrics_path)
        assert len(rows) == len(annual.y)
        assert [row["CountyI"] for row in rows] == [str(x) for x in annual.county_id]
        assert np.array_equal([int(row["outer_fold"]) for row in rows], labels)
        assert all(int(row["Year"]) == year for row in rows)
        observed = np.array([float(row["log_Yield"]) for row in rows])
        assert np.allclose(observed, annual.y, atol=1e-12, rtol=0)
        raw_yield = np.array([float(row["Yield_original"]) for row in rows])
        assert np.allclose(np.log(raw_yield), observed, atol=1e-12, rtol=0)
        completed = set()
        for metric in metrics:
            method = metric["method"]
            assert method in METHODS and method not in completed
            completed.add(method)
            for fold in range(N_OUTER):
                record = output / "records" / f"fold{fold:02d}_{method}.json"
                assert _record_is_valid(record, annual, labels, fold, method), record
            prediction = np.array([float(row[f"{method}_log_yhat"]) for row in rows])
            error = np.array([
                float(row[f"{method}_log_error_yhat_minus_y"]) for row in rows
            ])
            assert np.allclose(error, prediction - observed, atol=1e-12, rtol=0)
            assert np.isclose(np.sqrt(np.mean(error ** 2)),
                              float(metric["RMSE_logY"]), atol=1e-12)
            assert np.isclose(np.mean(np.abs(error)),
                              float(metric["MAE_logY"]), atol=1e-12)
            complete_total += 1
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        assert set(protocol["completed_methods"]) == completed
        if args.require_all:
            assert completed == set(METHODS), f"{year}: missing {set(METHODS) - completed}"
        print(f"PASS {year}: n={len(rows)}, complete={len(completed)}/{len(METHODS)}")
    print(f"Verified {complete_total} complete year-method combinations")


if __name__ == "__main__":
    main()
