"""Build numbered manuscript findings from newly generated model predictions.

No file in reference/ is used to fit, aggregate or draw a result. Completeness
checks prevent a partial run from being labelled as a reported experiment.
RMSE is calculated within each replication/year before averaging; the SD is
the sample SD of those replication/year metrics, not the SD of residuals.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/findings"
REALDATA = ROOT / "generated/realdata"
METHODS = ("GWR", "SGWR", "GNNWR", "SANNWR", "FNN", "GWFRN", "SGWFRN")
YEARS = range(1999, 2019)
ERRORS = ("normal", "student_t", "bimodal")
SCENARIOS = ("linear", "nonlinear_kappa0", "nonlinear_kappa1")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(errors):
    return math.sqrt(stats.fmean(e*e for e in errors)), stats.fmean(abs(e) for e in errors)


def close(left, right, label):
    if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"{label}: saved and recomputed quantities disagree.")


def mark(mean, sd, ordered, latex=False):
    # Mark only the mean, never its SD. Ranking uses unrounded values.
    unique = sorted(set(ordered))
    text = f"{mean:.4f}"
    if mean == unique[0]:
        text = (r"\textbf{" + text + "}") if latex else f"**{text}**"
    elif len(unique) > 1 and mean == unique[1]:
        text = (r"\underline{" + text + "}") if latex else f"<u>{text}</u>"
    return text + f" ({sd:.4f})"


def simulation():
    import numpy as np
    output = ROOT / "generated/simulation"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (config["replications"] != 200 or config["run_start"] != 1000
            or set(config["methods"]) != set(METHODS)
            or set(config["lattice_sizes"]) != {30, 40}
            or set(config["scenarios"]) != set(SCENARIOS)
            or set(config["errors"]) != set(ERRORS)):
        raise AssertionError("Tables 2--4 require the complete prespecified 200-replication cohort.")
    expected = {(m, s, e, method, r) for m in (30, 40) for s in SCENARIOS
                for e in ERRORS for method in METHODS for r in range(1000, 1200)}
    observed = set()
    for path in (output / "records").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(record[k] for k in ("m", "scenario", "error", "method", "run_id"))
        if key not in expected:
            continue
        if key in observed or record.get("status") != "completed" or record["signature"] != manifest["signature"]:
            raise AssertionError(f"Duplicate, incomplete or incompatible record: {path.name}")
        prediction = (output / record["prediction_file"]).resolve()
        if not prediction.is_relative_to(output.resolve()):
            raise ValueError("Prediction path escapes its run directory.")
        with np.load(prediction, allow_pickle=False) as data:
            errors = data["yhat"] - data["y_true"]
            if len(errors) != key[0]**2 // 10 or not np.isfinite(errors).all():
                raise AssertionError("Unexpected held-out count or nonfinite prediction.")
            close(np.sqrt(np.mean(errors**2)), record["rmse"], "simulation RMSE")
            close(np.mean(np.abs(errors)), record["mae"], "simulation MAE")
        observed.add(key)
    if observed != expected:
        raise AssertionError(f"Missing {len(expected-observed)} of {len(expected)} simulation records.")
    subprocess.run([sys.executable, "-B", str(ROOT / "simulation/report.py"), str(output)], check=True)
    rows = read(output / "summary_metrics.csv")
    # Table 2=linear; Table 3=additive; Table 4=interaction. Errors always appear
    # Normal, Student's t, bimodal; sample size is n, not lattice side length m.
    for number, scenario in enumerate(SCENARIOS, 2):
        found = {(int(r["m"]), r["error"], r["method"]): r for r in rows if r["scenario"] == scenario}
        numeric, md, tex = [], [], []
        header = " | ".join(f"{e} {metric.upper()}" for e in ERRORS for metric in ("rmse", "mae"))
        md += [f"# Table {number}", "", "| n | Method | " + header + " |", "|---:|---|" + "---:|"*6]
        tex += [r"\begin{tabular}{rlrrrrrr}", r"\toprule", "n & Method & " + header.replace(" | ", " & ") + r" \\", r"\midrule"]
        for m in (30, 40):
            for method in METHODS:
                cells, tcells = [], []
                for error in ERRORS:
                    row = found[m, error, method]
                    numeric.append(dict(table=number, n=m*m, scenario=scenario, error=error, method=method,
                                        **{k: row[k] for k in ("rmse_mean", "rmse_sd", "mae_mean", "mae_sd")}))
                    for metric in ("rmse", "mae"):
                        values = [float(found[m, error, method2][metric+"_mean"]) for method2 in METHODS]
                        a, b = float(row[metric+"_mean"]), float(row[metric+"_sd"])
                        cells.append(mark(a, b, values)); tcells.append(mark(a, b, values, latex=True))
                md.append(f"| {m*m} | {method} | " + " | ".join(cells) + " |")
                tex.append(f"{m*m} & {method} & " + " & ".join(tcells) + r" \\")
            tex.append(r"\midrule")
        tex[-1:] = [r"\bottomrule", r"\end{tabular}"]
        write(OUT / f"table{number}.csv", numeric)
        (OUT / f"table{number}.md").write_text("\n".join(md)+"\n", encoding="utf-8")
        (OUT / f"table{number}.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    print(f"PASS Tables 2--4: {len(observed)} prediction files checked; means/SD across 200 repetitions.")


def empirical():
    annual, all_rows, county_errors, coverage = [], [], {}, {}
    for year in YEARS:
        base = REALDATA / "by_year" / str(year)
        rows = read(base / "oof_predictions_logY.csv")
        ids = [r["CountyI"] for r in rows]
        if len(set(ids)) != len(ids) or set(int(r["outer_fold"]) for r in rows) != set(range(10)):
            raise AssertionError(f"Invalid or incomplete OOF keys/folds in {year}.")
        for r in rows:
            if int(r["Year"]) != year:
                raise AssertionError("Wrong year in prediction file.")
            close(math.log(float(r["Yield_original"])), r["log_Yield"], "log response")
            coverage.setdefault(r["CountyI"], set()).add(year)
        for method in METHODS:
            errors = []
            seen = set()
            for fold in range(10):
                rec = json.loads((base / "records" / f"fold{fold:02d}_{method}.json").read_text(encoding="utf-8"))
                if rec["year"] != year or rec["fold"] != fold or rec["method"] != method:
                    raise AssertionError("Wrong method/year/fold in record.")
                for county, prediction in zip(rec["county_id"], rec["log_yhat"]):
                    if county in seen:
                        raise AssertionError("County was predicted more than once in a year.")
                    seen.add(county)
                    match = next(r for r in rows if r["CountyI"] == county)
                    if int(match["outer_fold"]) != fold:
                        raise AssertionError("Fold label differs from prediction record.")
                    close(prediction, match[f"{method}_log_yhat"], "OOF prediction")
            if seen != set(ids):
                raise AssertionError("Every county must be tested exactly once per year.")
            for r in rows:
                error = float(r[f"{method}_log_yhat"]) - float(r["log_Yield"])
                close(error, r[f"{method}_log_error_yhat_minus_y"], "signed prediction error")
                errors.append(error)
                county_errors.setdefault((r["CountyI"], method), []).append(abs(error))
            rmse, mae = metrics(errors)
            annual.append(dict(year=year, method=method, n=len(rows), rmse=rmse, mae=mae))
        all_rows.extend(rows)
    if len(coverage) != 403 or len(all_rows) != 6152:
        raise AssertionError("Expected 403 counties and 6,152 county-years.")
    write(OUT / "annual_metrics.csv", annual)
    summary = []
    for method in METHODS:
        selected = [r for r in annual if r["method"] == method]
        summary.append(dict(method=method, n_years=20,
                            rmse_mean=stats.fmean(r["rmse"] for r in selected),
                            rmse_sd=stats.stdev(r["rmse"] for r in selected),
                            mae_mean=stats.fmean(r["mae"] for r in selected),
                            mae_sd=stats.stdev(r["mae"] for r in selected)))
    write(OUT / "table5.csv", summary)
    columns = ("GWR", "SGWR", "FNN", "GNNWR", "SANNWR", "GWFRN", "SGWFRN")
    keyed = {r["method"]: r for r in summary}
    md = ["# Table 5", "", "| Metric | " + " | ".join(columns) + " |", "|---|" + "---:|"*7]
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", "Metric & " + " & ".join(columns) + r" \\", r"\midrule"]
    for metric in ("rmse", "mae"):
        means = [keyed[m][metric+"_mean"] for m in columns]
        # Table 5 marks only the best mean (unlike Tables 2--4).
        cells, tcells = [], []
        for method in columns:
            value = keyed[method]
            a, b = value[metric+"_mean"], value[metric+"_sd"]
            text = f"{a:.4f}"
            cells.append((f"**{text}**" if a == min(means) else text) + f" ({b:.4f})")
            tcells.append(((r"\textbf{"+text+"}") if a == min(means) else text) + f" ({b:.4f})")
        md.append("| " + metric.upper() + " | " + " | ".join(cells) + " |")
        tex.append(metric.upper() + " & " + " & ".join(tcells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table5.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    (OUT / "table5.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    locations = {r["CountyI"]: r for r in read(ROOT / "data/input/loc_info.csv")}
    counties = []
    for county in sorted(coverage, key=int):
        loc = locations[county]
        row = dict(CountyI=county, state=loc["state"], county=loc["county"],
                   GEOID=loc["State.ANSI"].zfill(2)+loc["County.ANSI"].zfill(3),
                   n_years=len(coverage[county]))
        for method in METHODS:
            row[f"{method}_mean_absolute_error_logY"] = stats.fmean(county_errors[county, method])
        counties.append(row)
    write(OUT / "county_mae.csv", counties)
    area = [dict(method=m, equal_county_mean=stats.fmean(r[f"{m}_mean_absolute_error_logY"] for r in counties)) for m in METHODS]
    write(OUT / "equal_county_mae.csv", area)
    states = []
    for state in sorted(set(r["state"] for r in counties)):
        selected = [r for r in counties if r["state"] == state]
        for method in METHODS:
            states.append(dict(state=state, method=method, n_counties=len(selected),
                               equal_county_mae=stats.fmean(r[f"{method}_mean_absolute_error_logY"] for r in selected)))
    write(OUT / "state_mae.csv", states)
    print("PASS Table 5 and map source metrics: 20 years, 7 methods, 1,400 held-out folds, 6,152 county-years.")


def main():
    global OUT, REALDATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("simulation", "realdata", "all"), default="all")
    parser.add_argument("--realdata-results", type=Path, default=REALDATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    REALDATA, OUT = args.realdata_results.resolve(), args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part in ("simulation", "all"):
        simulation()
    if args.part in ("realdata", "all"):
        empirical()


if __name__ == "__main__":
    main()
