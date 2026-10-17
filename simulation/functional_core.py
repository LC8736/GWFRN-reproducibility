"""Verified 64/32 ReLU training core with validation-checkpoint selection."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import torch
from torch import nn
from network_import import load_networks_only
load_networks_only()

import data as dgp
import neural_methods as original
from metrics import regression_metrics
from settings import CONFIG

B_VALUES = tuple(CONFIG["bandwidths"])
SMOOTHNESS = tuple(CONFIG["smoothness"])
RIDGE = CONFIG["relative_head_ridge"]
CHECK_EVERY = CONFIG["validation_interval"]

def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Encoder(nn.Module):
    def __init__(self, n_inputs, hidden_dim):
        super().__init__()
        self.first = original._linear(n_inputs, 64)
        self.second = original._linear(64, hidden_dim)

    def forward(self, x):
        return torch.relu(self.second(torch.relu(self.first(x))))


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim):
        super().__init__()
        self.encoder = Encoder(n_inputs, hidden_dim)
        self.representation_dim = hidden_dim
        self.head = original._linear(hidden_dim, n_outputs, relu=False)

    def representation(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.representation(x))


def conditional_predict(h_fit, fitting_y, h_target, weights, ridge=RIDGE, chunk=128):
    """Exact same scaled, intercept-unpenalised ridge as the production code.

    Batch Gram construction uses the symmetric triangle to reduce work. No
    target responses enter this interface. All solves use float64.
    """
    h_fit, h_target = np.asarray(h_fit, dtype=float), np.asarray(h_target, dtype=float)
    design = np.column_stack((np.ones(len(h_fit)), h_fit))
    target = np.column_stack((np.ones(len(h_target)), h_target))
    d = design.shape[1]
    left, right = np.tril_indices(d)
    products = np.ascontiguousarray(design[:, left] * design[:, right])
    weighted_response = np.ascontiguousarray(design * np.asarray(fitting_y)[:, None])
    result = np.empty(len(h_target))
    for start in range(0, len(h_target), chunk):
        end = min(start + chunk, len(h_target))
        w = np.ascontiguousarray(weights[:, start:end].T)
        triangle = w @ products
        gram = np.empty((end - start, d, d))
        gram[:, left, right] = triangle
        gram[:, right, left] = triangle
        scale = np.maximum(np.trace(gram[:, 1:, 1:], axis1=1, axis2=2) / (d - 1), 1.0)
        diag = np.arange(1, d)
        gram[:, diag, diag] += ridge * scale[:, None]
        rhs = w @ weighted_response
        coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        result[start:end] = np.sum(target[start:end] * coefficients, axis=1)
    return result


@dataclass
class Fold:
    """Fitting and validation responses have separate, explicit roles."""
    fit_idx: np.ndarray
    val_idx: np.ndarray
    x_fit: torch.Tensor
    x_val: torch.Tensor
    y_scaled: torch.Tensor
    y_fit: np.ndarray
    y_val: np.ndarray
    y_mean: float
    y_scale: float
    functional_scales: torch.Tensor
    roughness: torch.Tensor
    weights: dict


def make_fold(raw, coords, fit_idx, val_idx, fitting_y, validation_y):
    assert len(fitting_y) == len(fit_idx)
    assert len(validation_y) == len(val_idx)
    assert np.isfinite(fitting_y).all() and np.isfinite(validation_y).all()
    mean, scale = raw[fit_idx].mean(0), raw[fit_idx].std(0)
    scale[scale < 1e-12] = 1.0
    y_mean = float(np.mean(fitting_y))
    y_scale = max(float(np.std(fitting_y)), 1e-12)
    geo = dgp.pairwise_euclidean(coords[fit_idx], coords[fit_idx])
    geo_scale = float(np.median(geo[geo > 1e-12]))
    target_geo = dgp.pairwise_euclidean(coords[fit_idx], coords[val_idx]) / geo_scale
    geo /= geo_scale
    weights = {}
    for b in B_VALUES:
        raw_w = np.exp(-np.square(geo / b))
        raw_target = np.exp(-np.square(target_geo / b))
        # Targets supply locations/profiles, never input rows or responses
        # to the gradient. Each head is normalised using fitting rows only.
        joint_w = np.column_stack((raw_w, raw_target))
        weights[b] = (
            torch.tensor(original._normalise_joint_training_weights(joint_w), dtype=torch.float32),
            raw_target,
        )
    return Fold(
        fit_idx, val_idx,
        torch.tensor((raw[fit_idx] - mean) / scale, dtype=torch.float32),
        torch.tensor((raw[val_idx] - mean) / scale, dtype=torch.float32),
        torch.tensor((fitting_y - y_mean) / y_scale, dtype=torch.float32),
        np.asarray(fitting_y).copy(), np.asarray(validation_y).copy(), y_mean, y_scale,
        torch.tensor(scale[:2*dgp.ANALYSIS_BASIS_DIM], dtype=torch.float32),
        torch.tensor(dgp.analysis_bspline_roughness(), dtype=torch.float32), weights,
    )


class FitState:
    def __init__(self, fold, candidate, hidden_dim, seed):
        original._configure(original.NeuralConfig(threads=1), seed)
        first_weights = next(iter(fold.weights.values()))[0]
        self.network = Network(fold.x_fit.shape[1], first_weights.shape[1], hidden_dim)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=CONFIG["learning_rate"])
        self.rng = np.random.default_rng(seed + 15485863)
        self.epoch = 0
        self.history = []
        self.candidate = candidate

    def advance(self, fold, total_epochs, callback=None):
        """Retain last parameters, Adam state and batch RNG between stages."""
        w, _ = fold.weights[self.candidate["b"]]
        n_fit = len(fold.fit_idx)
        n_heads = w.shape[1]
        assert n_heads == n_fit + len(fold.val_idx)
        while self.epoch < total_epochs:
            self.network.train()
            order = self.rng.permutation(n_fit)
            for start in range(0, n_fit, CONFIG["batch_size"]):
                index = torch.tensor(order[start:start+CONFIG["batch_size"]], dtype=torch.long)
                self.optimiser.zero_grad(set_to_none=True)
                residual = self.network(fold.x_fit[index]) - fold.y_scaled[index, None]
                error = torch.sum(w[index] * residual.square()) / (2 * len(index) * n_heads)
                penalty = 0.5 * self.candidate["lambda_smooth"] * original._normalised_functional_curvature(
                    self.network, fold.functional_scales, fold.roughness)
                (error + penalty).backward()
                self.optimiser.step()
            self.epoch += 1
            if callback is not None and self.epoch % CHECK_EVERY == 0:
                self.history.append(callback(self, fold))


def pooled_history(states, candidate_id, n_folds):
    histories = [{r["epoch"]: r for r in states[(candidate_id, k)].history} for k in range(n_folds)]
    common = set.intersection(*(set(h) for h in histories))
    rows = []
    for epoch in sorted(common):
        n = sum(h[epoch]["n_val"] for h in histories)
        mse = sum(h[epoch]["val_sse"] for h in histories) / n
        rows.append(dict(epoch=epoch, cv_mse=mse, cv_rmse=float(np.sqrt(mse)),
                         folds=n_folds, n_validation=n))
    return rows


def best_checkpoint(states, candidate_id, n_folds):
    return min(pooled_history(states, candidate_id, n_folds), key=lambda r: (r["cv_mse"], r["epoch"]))
