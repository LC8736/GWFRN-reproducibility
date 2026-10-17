"""Aggregate out-of-sample RMSE/MAE and method time; SD uses ddof=1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = ["m", "scenario", "error", "method"]

def _cell(mean, sd, digits=4, bold=False, second=False, latex=False):
    if not np.isfinite(mean):
        return "--"
    value = f"{mean:.{digits}f}"
    spread = f"({sd:.{digits}f})" if np.isfinite(sd) else "(SD unavailable)"
    if latex:
        if bold:
            value = r"\textbf{" + value + "}"
        elif second:
            value = r"\underline{" + value + "}"
        return r"\shortstack{" + value + r"\\" + spread + "}"
    marked = f"**{value}**" if bold else f"<u>{value}</u>" if second else value
    return marked + " " + spread


def _rank_flags(rows, error, metric, value):
    values = np.sort(rows.loc[rows.error == error, metric + "_mean"].dropna().unique())
    if not len(values):
        return False, False
    best = bool(abs(value - values[0]) <= 1e-12)
    later = values[values > values[0] + 1e-12]
    second = bool(not best and len(later) and abs(value - later[0]) <= 1e-12)
    return best, second


def combined_tables(output, summary, config, complete):
    """Three manuscript tables, each combining m=30 and m=40.

    Best and second-best marks use unrounded means within each m/error/metric
    cell. The SD is never marked. Timing is exported separately.
    """
    names = {"normal": "Normal", "student_t": "Student's $t$", "bimodal": "Bimodal"}
    titles = {
        "linear": "Model I: spatially varying linear effects",
        "nonlinear_kappa0": r"Model II with $\kappa=0$: nonlinear additive effects",
        "nonlinear_kappa1": r"Model II with $\kappa=1$: nonlinear interaction effects",
    }
    labels = {"linear": "tab:model1-m40", "nonlinear_kappa0": "tab:model2-kappa0-m40",
              "nonlinear_kappa1": "tab:model2-kappa1-m40"}
    errors = config["errors"]
    lines = ["# Combined main-experiment tables", "",
             "Only out-of-sample errors; cells give mean (sample SD across replications).",
             "Best mean is bold; second-best distinct mean is underlined within each lattice/error/metric.",
             f"Complete cohort: {complete}. Planned repeats per cell: {config['replications']}.", ""]
    if not complete or config["profile"] == "smoke":
        lines += ["WARNING: not a complete production result; do not use these tables in the manuscript.", ""]
    numbers = {"linear": 2, "nonlinear_kappa0": 3, "nonlinear_kappa1": 4}
    for scenario in config["scenarios"]:
        table_number = numbers[scenario]
        lines += [f"## Table {table_number}: {titles[scenario]}", "",
                  "| m | Method | " + " | ".join(names[e].replace("$", "") + " " + metric.upper()
                                                      for e in errors for metric in ("rmse", "mae")) + " |",
                  "|---:|---|" + "---:|" * (2 * len(errors))]
        tex = [r"\begin{table}[tbp]", r"\centering",
               r"\caption{Out-of-sample prediction errors for " + titles[scenario] + ", $m=" +
               ",".join(map(str, config["lattice_sizes"])) + "$.}",
               r"\label{" + labels[scenario] + "}", r"\small",
               r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
               r"\begin{tabular}{@{}cl" + "c" * (2 * len(errors)) + r"@{}}", r"\toprule",
               " & & " + " & ".join(r"\multicolumn{2}{c}{" + names[e] + "}" for e in errors) + r" \\",
               "".join(r"\cmidrule(lr){" + str(3 + 2*i) + "-" + str(4 + 2*i) + "}"
                       for i in range(len(errors))),
               "$m$ & Method & " + " & ".join(metric.upper() for e in errors for metric in ("rmse", "mae")) + r" \\",
               r"\midrule"]
        for block, m in enumerate(config["lattice_sizes"]):
            if block:
                tex.append(r"\midrule")
            rows = summary[(summary.m == m) & (summary.scenario == scenario)]
            for row_number, method in enumerate(config["methods"]):
                md_cells, tex_cells = [], []
                for error in errors:
                    found = rows[(rows.error == error) & (rows.method == method)]
                    for metric in ("rmse", "mae"):
                        if found.empty:
                            md_cells.append("--"); tex_cells.append("--"); continue
                        value = found.iloc[0]
                        mean, sd = value[metric + "_mean"], value[metric + "_sd"]
                        best, second = _rank_flags(rows, error, metric, mean)
                        md_cells.append(_cell(mean, sd, bold=best, second=second))
                        tex_cells.append(_cell(mean, sd, bold=best, second=second, latex=True))
                lines.append(f"| {m} | {method} | " + " | ".join(md_cells) + " |")
                lattice = (r"\multirow{" + str(2 * len(config["methods"])) + "}{*}{" + str(m) + "}"
                           if row_number == 0 else "")
                tex.append(lattice + " & " + method + " & " + " & ".join(tex_cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (output / "latex_tables" / f"table{table_number}_{scenario}.tex").write_text(
            "\n".join(tex) + "\n", encoding="utf-8")
        lines.append("")
    (output / "summary_combined_formatted.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarise(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config, signature = manifest["config"], manifest["signature"]
    runs = set(range(config["run_start"], config["run_start"] + config["replications"]))
    records, failures = [], []
    for path in sorted((output / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"Foreign code/configuration record: {path}")
        if record["run_id"] not in runs:
            continue
        if record["status"] == "completed":
            records.append(record)
        else:
            failures.append(record)
    columns = [*GROUPS, "run_id", "rmse", "mae", "seconds"]
    raw = pd.DataFrame([{k: record[k] for k in columns} for record in records], columns=columns)
    if raw.duplicated([*GROUPS, "run_id"]).any():
        raise ValueError("Duplicate method/scenario/replication records")
    raw.to_csv(output / "metrics_per_replication.csv", index=False)
    coverage = []
    for m in config["lattice_sizes"]:
        for scenario in config["scenarios"]:
            for error in config["errors"]:
                for method in config["methods"]:
                    subset = raw[(raw.m == m) & (raw.scenario == scenario) &
                                 (raw.error == error) & (raw.method == method)]
                    missing = sorted(runs - set(subset.run_id.tolist()))
                    coverage.append(dict(m=m, scenario=scenario, error=error, method=method,
                                         n_complete=len(subset), n_expected=len(runs),
                                         missing_run_ids=";".join(map(str, missing))))
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    if len(raw):
        summary = raw.groupby(GROUPS, sort=False).agg(
            n=("rmse", "count"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
            seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std")
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=[*GROUPS, "n", "rmse_mean", "rmse_sd", "mae_mean", "mae_sd",
                                        "seconds_mean", "seconds_sd"])
    summary[[*GROUPS, "n", "rmse_mean", "rmse_sd", "mae_mean", "mae_sd"]].to_csv(
        output / "summary_metrics.csv", index=False)
    summary[[*GROUPS, "n", "seconds_mean", "seconds_sd"]].to_csv(
        output / "summary_timing.csv", index=False)
    pd.DataFrame([dict(**{k: r[k] for k in GROUPS}, run_id=r["run_id"],
                       **r["selected"]) for r in records]).to_json(
        output / "selected_hyperparameters.json", orient="records", indent=2)
    pd.DataFrame([dict(**{k: r[k] for k in GROUPS}, run_id=r["run_id"],
                       selected=json.dumps(r["selected"], ensure_ascii=False)) for r in records]).to_csv(
        output / "selected_hyperparameters.csv", index=False)
    pd.DataFrame([{k: r.get(k) for k in [*GROUPS, "run_id", "status", "error_traceback"]}
                  for r in failures]).to_csv(output / "failures.csv", index=False)
    expected = len(runs) * len(coverage)
    complete = len(raw) == expected and all(c["n_complete"] == len(runs) for c in coverage)
    lines = ["# Main experiment: out-of-sample accuracy and computation time", "",
             f"Profile: {config['profile']}; planned replications per cell: {len(runs)}.",
             f"Completed records: {len(raw)}/{expected}; complete: {complete}.", "",
             "Cells contain the arithmetic mean followed by the sample SD (ddof=1) across paired replications.",
             "Only out-of-sample RMSE and MAE are reported. Single-replication SD is unavailable, not zero.",
             "Method seconds include preprocessing, tuning, fresh final fitting and prediction, not queue wait or DGP.",
             "Wall times were measured while other repetitions ran concurrently; they are not isolated hardware benchmarks.", ""]
    if not complete:
        lines += ["WARNING: partial results; inspect coverage.csv before any manuscript use.", ""]
    if config["profile"] == "smoke":
        lines += ["WARNING: smoke-test budgets, not research results.", ""]
    names = {"normal": "Normal", "student_t": "Student's t", "bimodal": "Bimodal"}
    tex_dir = output / "latex_tables"
    tex_dir.mkdir(exist_ok=True)
    for m in config["lattice_sizes"]:
        for scenario in config["scenarios"]:
            rows = summary[(summary.m == m) & (summary.scenario == scenario)]
            if rows.empty:
                continue
            errors = [error for error in config["errors"] if error in set(rows.error)]
            lines += [f"## m={m}: {scenario}", "", "### Out-of-sample prediction", "",
                      "| Method | " + " | ".join(f"{names[e]} {metric.upper()}" for e in errors for metric in ("rmse", "mae")) + " |",
                      "|---|" + "---:|" * (2 * len(errors))]
            tex = [r"\begin{table}[htbp]", r"\centering",
                   r"\caption{Out-of-sample prediction errors: " + scenario.replace("_", " ") + f", $m={m}$" + ".}",
                   r"\begin{tabular}{l" + "r" * (2 * len(errors)) + "}", r"\hline",
                   "Method & " + " & ".join(names[e] + " " + metric.upper() for e in errors for metric in ("rmse", "mae")) + r" \\", r"\hline"]
            for method in config["methods"]:
                md_cells, tex_cells = [], []
                for error in errors:
                    row = rows[(rows.error == error) & (rows.method == method)]
                    for metric in ("rmse", "mae"):
                        if row.empty:
                            md_cells.append("--"); tex_cells.append("--"); continue
                        value = row.iloc[0]
                        best = rows[rows.error == error][metric + "_mean"].min()
                        bold = abs(value[metric + "_mean"] - best) <= 1e-12
                        md_cells.append(_cell(value[metric + "_mean"], value[metric + "_sd"], bold=bold))
                        tex_cells.append(_cell(value[metric + "_mean"], value[metric + "_sd"], bold=bold, latex=True))
                lines.append("| " + method + " | " + " | ".join(md_cells) + " |")
                tex.append(method + " & " + " & ".join(tex_cells) + r" \\")
            tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
            (tex_dir / f"m{m}_{scenario}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
            lines += ["", "### Method computation time (seconds)", "",
                      "| Method | " + " | ".join(names[e] for e in errors) + " |",
                      "|---|" + "---:|" * len(errors)]
            for method in config["methods"]:
                cells = []
                for error in errors:
                    row = rows[(rows.error == error) & (rows.method == method)]
                    cells.append("--" if row.empty else _cell(row.iloc[0].seconds_mean, row.iloc[0].seconds_sd, digits=2))
                lines.append("| " + method + " | " + " | ".join(cells) + " |")
            lines.append("")
    (output / "summary_formatted.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    combined_tables(output, summary, config, complete)
    return dict(complete=complete, completed_records=len(raw), expected_records=expected,
                incomplete_cells=sum(c["n_complete"] != len(runs) for c in coverage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(summarise(parser.parse_args().output))
