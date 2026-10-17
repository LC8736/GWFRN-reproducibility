"""Shared h=32 GWFRN/SGWFRN core with configured conditional ridge.

Training is delegated to the shared FitState, including its
Adam optimiser, batch permutations, curvature penalty and checkpoint cadence.
Only the proximity weights and conditional-readout ridge differ here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import tempfile

import functional_core as base
from settings import CONFIG

np = base.np
torch = base.torch

RIDGE = CONFIG["relative_head_ridge"]
HIDDEN_DIM = 32
ALPHA_VALUES = tuple(CONFIG["similarity_alphas"])
STATE_VERSION = 2
_CANDIDATE_FIELDS = ("candidate_id", "b", "alpha", "lambda_smooth")


def _method_name(method):
    method = str(method).upper()
    if method not in {"GWFRN", "SGWFRN"}:
        raise ValueError("method must be 'GWFRN' or 'SGWFRN'")
    return method


def candidates(method):
    """Return the fixed grid in bandwidth, alpha, smoothness order."""
    alpha_values = (1.0,) if _method_name(method) == "GWFRN" else ALPHA_VALUES
    return [
        dict(candidate_id=i, b=b, alpha=alpha, lambda_smooth=smoothness)
        for i, (b, alpha, smoothness) in enumerate(
            (b, alpha, smoothness)
            for b in base.B_VALUES
            for alpha in alpha_values
            for smoothness in base.SMOOTHNESS
        )
    ]


def _candidate_identity(candidate):
    identity = {key: candidate[key] for key in _CANDIDATE_FIELDS}
    candidate_id = int(identity["candidate_id"])
    if candidate_id != identity["candidate_id"] or candidate_id < 0:
        raise ValueError("candidate_id must be a non-negative integer")
    identity["candidate_id"] = candidate_id
    for key in ("b", "alpha", "lambda_smooth"):
        identity[key] = float(identity[key])
        if not np.isfinite(identity[key]):
            raise ValueError(f"Candidate {key} must be finite")
    if identity["b"] <= 0 or identity["lambda_smooth"] < 0:
        raise ValueError("Bandwidth must be positive and smoothness non-negative")
    if not 0 < identity["alpha"] <= 1:
        raise ValueError("Candidate alpha must lie in (0, 1]")
    return identity


def make_fold(
    method, data, raw, fit_idx, val_idx, fitting_y, validation_y,
    candidate_list=None,
) -> base.Fold:
    """Build standardised inputs and fit-only-calibrated proximity weights.

    Public weight keys are always ``(bandwidth, alpha)``.  ``candidate_list``
    limits the retained weight cache to the distinct requested keys; omitting
    it builds the method's complete grid.  No response is read from ``data``.
    """
    method = _method_name(method)
    requested = candidates(method) if candidate_list is None else list(candidate_list)
    identities = [_candidate_identity(candidate) for candidate in requested]
    if method == "GWFRN" and any(c["alpha"] != 1.0 for c in identities):
        raise ValueError("GWFRN candidates must have alpha=1")
    keys = list(dict.fromkeys((c["b"], c["alpha"]) for c in identities))

    # This also preserves the original feature/response scaling and tensor
    # construction.  The temporary geographical cache is replaced below.
    fold = base.make_fold(raw, data.coords, fit_idx, val_idx, fitting_y, validation_y)
    if method == "GWFRN":
        if any(b not in fold.weights for b, _ in keys):
            raise ValueError("GWFRN bandwidth must belong to base.B_VALUES")
        fold.weights = {(b, alpha): fold.weights[b] for b, alpha in keys}
        geographic = base.dgp.pairwise_euclidean(data.coords[fit_idx], data.coords[fit_idx])
        geo_scale = float(np.median(geographic[geographic > 1e-12]))
        fold.proximity_scales = {
            "geographical": geo_scale,
            "covariate": None,
            "functional": [],
            "scalar": [],
        }
    else:
        scales = base.original._distance_scales(data, fit_idx)
        train_geo, train_covariate = base.original._scaled_proximity_components(
            data, fit_idx, fit_idx, scales
        )
        target_geo, target_covariate = base.original._scaled_proximity_components(
            data, fit_idx, val_idx, scales
        )
        weights = {}
        for b, alpha in keys:
            raw_train = base.original._weights_from_components(
                train_geo, train_covariate, b, alpha
            )
            raw_target = base.original._weights_from_components(
                target_geo, target_covariate, b, alpha
            )
            weights[(b, alpha)] = (
                torch.tensor(
                    base.original._normalise_joint_training_weights(
                        np.column_stack((raw_train, raw_target))),
                    dtype=torch.float32,
                ),
                raw_target,
            )
        fold.weights = weights
        fold.proximity_scales = asdict(scales)
    return fold


def predict(hfit, yfit, htarget, weights):
    """Float64 conditional ridge with no fallback jitter or clipping."""
    hfit = np.asarray(hfit, dtype=np.float64)
    yfit = np.asarray(yfit, dtype=np.float64)
    htarget = np.asarray(htarget, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if hfit.ndim != 2 or htarget.ndim != 2:
        raise ValueError("Fitting and target representations must be matrices")
    if not len(hfit) or not hfit.shape[1] or htarget.shape[1] != hfit.shape[1]:
        raise ValueError("Representation dimensions are empty or incompatible")
    if yfit.shape != (len(hfit),):
        raise ValueError("Fitting response must have one value per fitting row")
    if weights.shape != (len(hfit), len(htarget)):
        raise ValueError("Weights must have shape (n_fit, n_target)")
    if not all(np.isfinite(value).all() for value in (hfit, yfit, htarget, weights)):
        raise ValueError("Conditional prediction inputs must all be finite")
    if np.any(weights < 0):
        raise ValueError("Conditional weights must be non-negative")
    if np.any(np.all(weights == 0, axis=0)):
        raise ValueError("Conditional prediction has an all-zero weight column")
    result = base.conditional_predict(hfit, yfit, htarget, weights, ridge=RIDGE)
    if not np.isfinite(result).all():
        raise FloatingPointError("Conditional prediction produced a non-finite result")
    return result


def evaluate_checkpoint(state, fold):
    """Score held-out rows through the same conditional readout used at test time."""
    state.network.eval()
    with torch.no_grad():
        h_fit = state.network.representation(fold.x_fit).cpu().numpy()
        h_val = state.network.representation(fold.x_val).cpu().numpy()
    key = (state.candidate["b"], state.candidate["alpha"])
    if key in fold.weights:
        target_weights = fold.weights[key][1]
    elif getattr(fold, "_bound_weight_key", None) == key and key[0] in fold.weights:
        target_weights = fold.weights[key[0]][1]
    else:
        raise KeyError("Checkpoint fold does not carry this (b, alpha) weight pair")
    prediction = predict(h_fit, fold.y_fit, h_val, target_weights)
    residual = prediction - fold.y_val
    return dict(
        epoch=state.epoch,
        val_sse=float(residual @ residual),
        n_val=len(residual),
        val_mse=float(np.mean(residual**2)),
        val_mae=float(np.mean(np.abs(residual))),
    )


def _fold_identity(fold):
    """Identify the fold and fitting inputs without serialising its weights."""
    digest = hashlib.sha256()
    for value in (
        np.asarray(fold.fit_idx, dtype=np.int64),
        np.asarray(fold.val_idx, dtype=np.int64),
        fold.x_fit.detach().cpu().numpy(),
        fold.x_val.detach().cpu().numpy(),
        fold.y_scaled.detach().cpu().numpy(),
        np.asarray(fold.y_fit, dtype=np.float64),
        np.asarray(fold.y_val, dtype=np.float64),
    ):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return {"sha256": digest.hexdigest(), "n_fit": len(fold.fit_idx),
            "n_target": len(fold.val_idx), "n_inputs": fold.x_fit.shape[1]}


class State(base.FitState):
    """Original h=32 training loop with pair-keyed proximity weights."""

    def __init__(self, fold, candidate, seed):
        identity = _candidate_identity(candidate)
        if (identity["b"], identity["alpha"]) not in fold.weights:
            raise KeyError("Fold does not cache this candidate's proximity weights")
        self.seed = int(seed)
        self.ridge = RIDGE
        self.hidden_dim = HIDDEN_DIM
        self.fold_identity = _fold_identity(fold)
        super().__init__(fold, deepcopy(candidate), HIDDEN_DIM, self.seed)

    def advance(self, fold, total_epochs, callback=None):
        key = (self.candidate["b"], self.candidate["alpha"])
        training_fold = replace(fold, weights={self.candidate["b"]: fold.weights[key]})
        # Preserve the selected alpha explicitly: scalar-b lookup alone must
        # never select another SG candidate that shares this bandwidth.
        training_fold._bound_weight_key = key
        if hasattr(fold, "proximity_scales"):
            training_fold.proximity_scales = fold.proximity_scales
        return super().advance(training_fold, total_epochs, callback)


def save_state(state, path):
    """Atomically persist an epoch-boundary state for local continuation."""
    if state.ridge != RIDGE or state.hidden_dim != HIDDEN_DIM:
        raise ValueError("State does not belong to this ridge/dimension protocol")
    payload = {
        "version": STATE_VERSION,
        "candidate": _candidate_identity(state.candidate),
        "seed": state.seed,
        "ridge": state.ridge,
        "hidden_dim": state.hidden_dim,
        "fold_identity": state.fold_identity,
        "model_state": state.network.state_dict(),
        "optimizer_state": state.optimiser.state_dict(),
        "rng_state": deepcopy(state.rng.bit_generator.state),
        "torch_rng_state": torch.get_rng_state(),
        "epoch": state.epoch,
        "history": deepcopy(state.history),
        "network_training": state.network.training,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_state(path, fold, candidate, seed) -> State:
    """Restore a checkpoint created locally by :func:`save_state`.

    State files include NumPy RNG state and are trusted local artifacts, not
    untrusted downloaded model files.  Candidate metadata is compared using
    the four grid fields, permitting additional selection-score fields.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    expected = {
        "version": STATE_VERSION,
        "candidate": _candidate_identity(candidate),
        "seed": int(seed),
        "ridge": RIDGE,
        "hidden_dim": HIDDEN_DIM,
        "fold_identity": _fold_identity(fold),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Saved state {key} does not match this continuation")
    epoch = payload["epoch"]
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Saved epoch must be a non-negative integer")
    state = State(fold, candidate, seed)
    state.network.load_state_dict(payload["model_state"], strict=True)
    state.optimiser.load_state_dict(payload["optimizer_state"])
    state.rng.bit_generator.state = deepcopy(payload["rng_state"])
    state.epoch = epoch
    state.history = deepcopy(payload["history"])
    state.network.train(bool(payload["network_training"]))
    torch.set_rng_state(payload["torch_rng_state"])
    return state
