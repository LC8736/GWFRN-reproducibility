"""Windows server main experiment; one complete replication per worker task."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
SEED_GROUP = {"GWR": 1, "SGWR": 3, "FNN": 4, "GNNWR": 5, "SANNWR": 6,
              "GWFRN": 7, "SGWFRN": 7}


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(plain(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        temp = Path(stream.name)
    os.replace(temp, path)


def source_hashes():
    sources = list(ROOT.glob("*.py"))
    sources += list((ROOT/"upstream/gnnwr/src/gnnwr").glob("*.py"))
    sources += list((ROOT/"upstream/FastSGWR/serial_sgwr").glob("*.py"))
    sources += list((ROOT/"upstream/FuncNN/R").glob("*.R"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}


def signature(config):
    # Worker count and cohort length may be changed for continuation. Fitting
    # configuration and all executable source files must remain identical.
    fitting = {k: v for k, v in config.items() if k not in ("workers", "replications", "run_start")}
    payload = dict(config=fitting, source_hashes=source_hashes())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def masked_data(data):
    import numpy as np
    if np.intersect1d(data.model_idx, data.test_idx).size:
        raise ValueError("Outer fitting and test samples overlap")
    if not np.array_equal(np.sort(np.r_[data.model_idx, data.test_idx]), np.arange(len(data.y))):
        raise ValueError("Outer partition does not cover all observations")
    for fit_idx, val_idx in data.cv_folds:
        if np.intersect1d(fit_idx, val_idx).size or np.intersect1d(val_idx, data.test_idx).size:
            raise ValueError("Invalid spatial CV partition")
        if not np.array_equal(np.sort(np.r_[fit_idx, val_idx]), data.model_idx):
            raise ValueError("An inner fold does not partition the modelling set")
    y = data.y.copy()
    y[data.test_idx] = np.nan
    y.setflags(write=False)
    return replace(data, y=y)


def fit_method(method, data, output):
    from settings import CONFIG
    from native_neural import fit_native
    from native_linear import fit_gwr, fit_sgwr
    from train_representation import fit_representation
    seed = int(data.seed) + 100_003 * SEED_GROUP[method]
    if method == "GWR":
        result = fit_gwr(data, output)
    elif method == "SGWR":
        result = fit_sgwr(data, output)
    elif method in ("GWFRN", "SGWFRN"):
        result = fit_representation(method, data, seed, output)
    else:
        result = fit_native(method, data, seed, output)
    return result, seed


def method_record(task, data, method):
    """Only this scoring wrapper holds unmasked outer-test responses."""
    import numpy as np
    from metrics import regression_metrics
    key = f"m{task['m']}_{task['scenario']}_{task['error']}_r{task['run_id']:04d}_{method}"
    output = Path(task["output"])
    record_path = output / "records" / f"{key}.json"
    prediction_path = output / "predictions" / f"{key}.npz"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("signature") != task["signature"]:
            raise ValueError(f"Refusing to reuse incompatible record {record_path}")
        if previous.get("status") == "completed":
            if not prediction_path.exists():
                raise FileNotFoundError(f"Completed record has no predictions: {key}")
            return dict(key=key, status="skipped")
    artifact = output / "artifacts" / key
    artifact.mkdir(parents=True, exist_ok=True)
    record = dict(status="running", started=now(), signature=task["signature"], key=key,
                  profile=task["profile"], method=method, run_id=task["run_id"],
                  scenario=task["scenario"], model=data.model,
                  kappa=None if data.model == "model1" else data.interaction_kappa,
                  error=data.error, m=task["m"], n=task["m"] ** 2, data_seed=int(data.seed),
                  n_modelling=len(data.model_idx), n_test=len(data.test_idx))
    write_json(record_path, record)
    try:
        training = masked_data(data)
        # Method time includes response-free method preprocessing, tuning,
        # fresh final fit and test prediction. It excludes DGP, queue wait,
        # scoring, prediction-file writing and cross-replication reporting.
        with (output / "logs" / f"{key}.log").open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                wall, cpu = time.perf_counter(), time.process_time()
                fitted, seed = fit_method(method, training, artifact)
                seconds, cpu_seconds = time.perf_counter() - wall, time.process_time() - cpu
        prediction = np.asarray(fitted["test_prediction"], dtype=np.float64).copy()
        if prediction.shape != (len(data.test_idx),) or not np.isfinite(prediction).all():
            raise FloatingPointError("Missing, nonfinite or incorrectly ordered test predictions")
        if not np.isnan(training.y[data.test_idx]).all() or training.y.flags.writeable:
            raise RuntimeError("Outer-response masking was modified during fitting")
        # These are the only reported accuracy metrics. No fitted-sample
        # RMSE/MAE is exported to the experiment summaries. Native benchmark
        # trainers may compute their own fitting/validation diagnostics.
        metrics = regression_metrics(data.y[data.test_idx], prediction)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=prediction_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, test_idx=data.test_idx, coords=data.coords[data.test_idx],
                                y_true=data.y[data.test_idx], yhat=prediction,
                                residual=prediction - data.y[data.test_idx])
        os.replace(temporary, prediction_path)
        record.update(status="completed", finished=now(), rmse=metrics["rmse"], mae=metrics["mae"],
                      seconds=seconds, parent_cpu_seconds=cpu_seconds, training_seed=seed,
                      selected=fitted["selected"], implementation=fitted["implementation"],
                      method_protocol=fitted.get("protocol"),
                      outer_test_y_masked_before_fit=True, validation_y_in_gradient=False,
                      scores_computed_after_predictions=True, prediction_file=str(prediction_path.relative_to(output)))
        write_json(record_path, record)
        return dict(key=key, status="completed", rmse=metrics["rmse"], mae=metrics["mae"])
    except Exception:
        record.update(status="failed", failed_at=now(), error_traceback=traceback.format_exc())
        write_json(record_path, record)
        return dict(key=key, status="failed", message=record["error_traceback"])
    finally:
        gc.collect()


def worker_replication(task):
    """Serially process all requested scenarios/methods of one replication.

    Parallelism is across independent repetitions, not folds/candidates or
    methods. Thus 100 repeats can run at once without nested process pools.
    """
    import numpy as np
    import torch
    from data import generate_data
    from settings import CONFIG, SCENARIOS
    if signature(CONFIG) != task["signature"]:
        raise RuntimeError("Code/configuration changed after this run was dispatched")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    summary = dict(run_id=task["run_id"], completed=0, skipped=0, failed=0)
    for m in task["lattice_sizes"]:
        for scenario in task["scenarios"]:
            model, kappa = SCENARIOS[scenario]
            for error in task["errors"]:
                job = dict(task, m=m, scenario=scenario, error=error)
                data = generate_data(task["run_id"], model, error, kappa=kappa, inner_folds=5, m=m)
                split_path = Path(task["output"]) / "splits" / f"m{m}_{scenario}_{error}_r{task['run_id']:04d}.npz"
                split_path.parent.mkdir(parents=True, exist_ok=True)
                if not split_path.exists():
                    np.savez_compressed(split_path, model_idx=data.model_idx, test_idx=data.test_idx,
                                        block_ids=data.block_ids, coords=data.coords,
                                        **{f"fit_{k}": fi for k, (fi, vi) in enumerate(data.cv_folds)},
                                        **{f"val_{k}": vi for k, (fi, vi) in enumerate(data.cv_folds)})
                for method in task["methods"]:
                    result = method_record(job, data, method)
                    summary[result["status"]] += 1
                    print(f"{now()} {result['status'].upper()} {result['key']}", flush=True)
    return summary


def preflight():
    import numpy
    import scipy
    import pandas
    import torch
    from network_import import load_networks_only
    load_networks_only()
    from gnnwr.networks import SWNN
    from native_neural import author_gnnwr
    from native_linear import official_alpha_class
    author_gnnwr()
    official_alpha_class()
    from settings import CONFIG
    native_runtime = {}
    if "GWR" in CONFIG["methods"]:
        try:
            from mgwr.gwr import GWR
        except (ImportError, KeyError) as exc:
            raise RuntimeError(
                "PySAL GWR could not load its installed dependency stack. "
                "Fix the Python environment before starting production; "
                "this release does not silently replace GWR with another estimator."
            ) from exc
        native_runtime["mgwr"] = importlib.metadata.version("mgwr")
        native_runtime["gwr_import_checked"] = GWR.__name__
        for name in ("libpysal", "spreg", "spglm", "scikit-learn"):
            native_runtime[name] = importlib.metadata.version(name)
    versions = {}
    for name in ("numpy", "scipy", "pandas", "torch", "gnnwr"):
        versions[name] = importlib.metadata.version(name)
    return dict(python=sys.version, platform=platform.platform(), versions=versions,
                detected_logical_cpus=os.cpu_count(), device="cpu", numerical_threads_per_worker=1,
                native_runtime=native_runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results_200")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--run-start", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ["GWFRN_CONFIG_PATH"] = str(args.config.resolve())
    from settings import CONFIG
    config = dict(CONFIG)
    for arg, name in ((args.workers, "workers"), (args.runs, "replications"), (args.run_start, "run_start")):
        if arg is not None:
            config[name] = arg
    from settings import validate_config
    validate_config(config)
    environment = preflight()
    print(json.dumps(environment, indent=2), flush=True)
    if args.preflight:
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("records", "predictions", "logs", "splits", "artifacts"):
        (output / name).mkdir(exist_ok=True)
    run_signature = signature(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != run_signature:
            raise ValueError("Output belongs to another code/configuration. Use a fresh output directory.")
    else:
        manifest = dict(created=now(), signature=run_signature, source_hashes=source_hashes())
    manifest.update(config=config, environment=environment, started=now(), status="running",
                    timing_definition="method wall seconds: preprocessing+tuning+refit+test prediction; excludes DGP/queue/metrics/report",
                    validation_use="checkpoint/candidate selection only; no validation or test response gradients")
    write_json(manifest_path, manifest)
    base_task = dict(output=str(output), signature=run_signature, profile=config["profile"],
                     lattice_sizes=config["lattice_sizes"], scenarios=config["scenarios"],
                     errors=config["errors"], methods=config["methods"])
    tasks = [dict(base_task, run_id=run_id) for run_id in range(config["run_start"],
                                                            config["run_start"] + config["replications"])]
    expected = len(tasks) * len(config["lattice_sizes"]) * len(config["scenarios"]) * len(config["errors"]) * len(config["methods"])
    print(f"{config['profile']}: {len(tasks)} paired replications, {expected} method records, "
          f"{config['workers']} processes; parallelism across replications only.", flush=True)
    if config["profile"] == "smoke":
        print("SMOKE CHECK ONLY: reduced budgets; do not use these results in the paper.", flush=True)
    print("100 workers require sufficient physical RAM; reduce --workers on smaller computers.", flush=True)
    started = time.perf_counter()
    totals = dict(completed=0, skipped=0, failed=0)
    try:
        context = mp.get_context("spawn")
        # multiprocessing.Pool avoids the Windows ProcessPoolExecutor limit
        # of 61 processes. Each repetition runs serially inside a worker;
        # folds, candidates and methods do not start nested process pools.
        with context.Pool(processes=config["workers"], maxtasksperchild=1) as pool:
            for summary in pool.imap_unordered(worker_replication, tasks, chunksize=1):
                for key in totals:
                    totals[key] += summary[key]
                print(f"Replication {summary['run_id']} finished: {summary}; totals={totals}", flush=True)
        from report import summarise
        audit = summarise(output)
        if source_hashes() != manifest["source_hashes"]:
            raise RuntimeError("Source files changed during the experiment; do not combine this run")
        manifest.update(status="completed" if audit["complete"] else "incomplete",
                        finished=now(), run_wall_seconds=time.perf_counter() - started,
                        totals=totals, coverage=audit)
        write_json(manifest_path, manifest)
        if not audit["complete"]:
            raise SystemExit("Incomplete run: inspect failure/coverage files, then rerun the same command to resume.")
    except BaseException:
        if manifest.get("status") == "running":
            manifest.update(status="interrupted", interrupted_at=now())
            write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
