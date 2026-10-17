"""Staged spatial CV with continuation and validation-selected training epochs.

Candidate fits are continued across screening stages inside each Monte Carlo worker.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import functional_core as base
import representation as core
from settings import CONFIG


def fit_representation(method, data, seed, output):
    """Fit a complete GWFRN or SGWFRN without access to any outer-test Y."""
    if method not in ("GWFRN", "SGWFRN"):
        raise ValueError("Unsupported representation method")
    if not np.isnan(data.y[data.test_idx]).all() or data.y.flags.writeable:
        raise ValueError("Outer-test responses must be masked and read-only")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = np.column_stack((base.dgp.functional_integrals(data.x), data.z))
    active = core.candidates(method)
    history, curves, pooled_curves, audit = {}, [], [], []
    # Only checkpoints from this invocation live here. Each process works in a
    # distinct replication/scenario/method directory; there is no shared state.
    with tempfile.TemporaryDirectory(prefix="cv_state_", dir=output) as temporary:
        state_dir = Path(temporary).resolve()
        if state_dir.parent != output:
            raise RuntimeError("Checkpoint directory is outside the assigned output")
        for stage_number, stage in enumerate(CONFIG["stages"], 1):
            for k in range(stage["folds"]):
                fi, vi = data.cv_folds[k]
                fold = core.make_fold(method, data, raw, fi, vi,
                                      data.y[fi], data.y[vi], active)
                for candidate in active:
                    cid = candidate["candidate_id"]
                    path = state_dir / f"candidate_{cid:02d}_fold_{k}.pt"
                    state = (core.load_state(path, fold, candidate, seed + k)
                             if path.exists() else core.State(fold, candidate, seed + k))
                    before, old_length = state.epoch, len(state.history)
                    state.advance(fold, stage["total_epochs"], core.evaluate_checkpoint)
                    core.save_state(state, path)
                    history[(cid, k)] = SimpleNamespace(history=state.history.copy())
                    curves.extend(dict(stage_first_recorded=stage_number, fold=k,
                                       **candidate, **row) for row in state.history[old_length:])
                    if before not in (0, CONFIG["stages"][stage_number - 2]["total_epochs"]):
                        raise RuntimeError("Unexpected continuation epoch")
                    del state
                del fold
            ranked = []
            for candidate in active:
                pooled = base.pooled_history(history, candidate["candidate_id"], stage["folds"])
                pooled_curves.extend(dict(stage=stage_number, **candidate, **row) for row in pooled)
                ranked.append(dict(**candidate, **base.best_checkpoint(
                    history, candidate["candidate_id"], stage["folds"])))
            ranked.sort(key=lambda row: (row["cv_mse"], row["epoch"], row["candidate_id"]))
            retained = {row["candidate_id"] for row in ranked[:stage["retain"]]}
            audit.extend(dict(stage=stage_number, cumulative_budget=stage["total_epochs"],
                              retained=row["candidate_id"] in retained, **row) for row in ranked)
            active = [candidate for candidate in active if candidate["candidate_id"] in retained]
            # Eliminated models cannot re-enter. Free their optimiser files
            # immediately instead of storing all stages of all 200 repeats.
            for path in state_dir.glob("candidate_*_fold_*.pt"):
                cid = int(path.name.split("_")[1])
                if cid not in retained:
                    path.unlink()
            history = {key: value for key, value in history.items() if key[0] in retained}
            selected = ranked[0]
            pd.DataFrame(curves).to_csv(output / "validation_curves.csv", index=False)
            pd.DataFrame(pooled_curves).to_csv(output / "pooled_cv_curves.csv", index=False)
            pd.DataFrame(audit).to_csv(output / "selection_stage_audit.csv", index=False)
        if CONFIG["save_cv_checkpoints"]:
            shutil.copytree(state_dir, output / "cv_checkpoints", dirs_exist_ok=True)

    # No true target response is supplied to this container. The encoder has
    # fitting and target heads, and is refitted from scratch at the common
    # CV-selected epoch. Optimisation is never stopped using outer-test error.
    fi, ti = data.model_idx, data.test_idx
    fold = core.make_fold(method, data, raw, fi, ti, data.y[fi],
                          np.zeros(len(ti)), [selected])
    state = core.State(fold, selected, seed + 1_000_003)
    state.advance(fold, int(selected["epoch"]))
    state.network.eval()
    with torch.no_grad():
        hfit = state.network.representation(fold.x_fit).cpu().numpy()
        htest = state.network.representation(fold.x_val).cpu().numpy()
    prediction = core.predict(hfit, fold.y_fit, htest,
                              fold.weights[(selected["b"], selected["alpha"])][1])
    if CONFIG["save_final_models"]:
        torch.save(dict(model_state=state.network.state_dict(), selected=selected,
                        hidden_dim=32, ridge=core.RIDGE, final_epochs=state.epoch,
                        input_mean=raw[fi].mean(0), input_scale=raw[fi].std(0),
                        y_mean=fold.y_mean, y_scale=fold.y_scale,
                        proximity_scales=fold.proximity_scales), output / "fitted_model.pt")
    return dict(test_prediction=prediction, selected=selected,
                implementation="64/32 ReLU shared encoder; fitting and target heads jointly trained; conditional ridge prediction",
                protocol=dict(hidden_dim=32, skip_connection=False, target_heads_jointly_trained=True,
                              n_final_heads=len(fi)+len(ti),
                              head_normalisation="fit-row column mean one; loss divided by actual head count",
                              relative_head_ridge=core.RIDGE, validation_interval=base.CHECK_EVERY,
                              epoch_selection="minimum common-epoch pooled five-fold validation MSE",
                              validation_y_in_gradient=False, test_y_available=False))
