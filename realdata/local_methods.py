"""Published local-linear benchmarks used in the main simulation.

GWR and MGWR are fitted with PySAL ``mgwr``.  SGWR follows the public
``FastSGWR`` implementation: geographical and similarity kernels are mixed at
the weight level.  A small custom weighted-least-squares routine is used for
spatial-block validation and for SGWR prediction because FastSGWR 0.5 indexes
target points as though they were rows of the training data and therefore does
not implement genuine held-out-location prediction.

All local-linear benchmarks receive the same 30-dimensional orthonormalised
cubic B-spline projection
coefficients per observed curve, together with the scalar predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

from data import SimulationData, pairwise_euclidean


@dataclass
class LinearFeatures:
    values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def linear_features(data: SimulationData, fit_idx: np.ndarray) -> LinearFeatures:
    raw = np.column_stack((data.bspline_coef, data.z))
    mean = raw[fit_idx].mean(axis=0)
    scale = raw[fit_idx].std(axis=0)
    scale[scale < 1e-12] = 1.0
    return LinearFeatures((raw - mean) / scale, mean, scale)


def _design(features: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(features)), features))


def _weighted_least_squares_predictions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_target: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Local WLS predictions for an n_fit by n_target weight matrix."""

    if weights.shape != (len(x_fit), len(x_target)):
        raise ValueError("weights have incompatible dimensions")
    result = np.empty(len(x_target), dtype=np.float64)
    p = x_fit.shape[1]
    eye = np.eye(p)
    for start in range(0, len(x_target), chunk_size):
        stop = min(start + chunk_size, len(x_target))
        w = weights[:, start:stop]
        gram = np.einsum("nr,np,nq->rpq", w, x_fit, x_fit, optimize=True)
        rhs = np.einsum("nr,np,n->rp", w, x_fit, y_fit, optimize=True)
        ridge = np.maximum(np.trace(gram, axis1=1, axis2=2) / p, 1.0) * 1e-9
        gram += ridge[:, None, None] * eye
        try:
            beta = np.linalg.solve(gram, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            beta = np.stack([np.linalg.pinv(g) @ r for g, r in zip(gram, rhs)])
        result[start:stop] = np.sum(x_target[start:stop] * beta, axis=1)
    return result


def _positive_scale(distance: np.ndarray) -> float:
    positive = distance[distance > 1e-12]
    if positive.size == 0:
        raise ValueError("distance matrix is degenerate")
    return float(np.median(positive))


def _bandwidth_grid(coords: np.ndarray) -> tuple[np.ndarray, float]:
    distance = pairwise_euclidean(coords, coords)
    scale = _positive_scale(distance)
    positive = (distance / scale)[distance > 1e-12]
    return np.unique(np.quantile(positive, (0.10, 0.25, 0.50, 0.75, 0.90))), scale


def _gwr_weights(
    coords_fit: np.ndarray,
    coords_target: np.ndarray,
    bandwidth_scaled: float,
    geographical_scale: float,
) -> np.ndarray:
    distance = pairwise_euclidean(coords_fit, coords_target)
    return np.exp(-0.5 * np.square(distance / (bandwidth_scaled * geographical_scale)))


def _sgwr_weights(
    coords_fit: np.ndarray,
    attrs_fit: np.ndarray,
    coords_target: np.ndarray,
    attrs_target: np.ndarray,
    bandwidth_scaled: float,
    geographical_scale: float,
    alpha: float,
) -> np.ndarray:
    geographical = _gwr_weights(
        coords_fit, coords_target, bandwidth_scaled, geographical_scale
    )
    # FastSGWR defines similarity by the mean absolute attribute difference
    # and applies an unscaled Gaussian kernel to that quantity.
    similarity_distance = np.mean(
        np.abs(attrs_fit[:, None, :] - attrs_target[None, :, :]), axis=2
    )
    similarity = np.exp(-np.square(similarity_distance))
    return alpha * geographical + (1.0 - alpha) * similarity


def _spatial_cv_gwr(
    data: SimulationData,
    *,
    similarity: bool,
    pilot: bool,
) -> dict:
    model_idx = data.model_idx
    full_features = linear_features(data, model_idx).values
    candidates_b, geo_scale = _bandwidth_grid(data.coords[model_idx])
    if pilot:
        candidates_b = candidates_b[[0, 2, -1]]
    candidates_alpha = (0.25, 0.50, 0.75, 1.0) if similarity else (1.0,)
    scores: list[dict] = []
    for b in candidates_b:
        for alpha in candidates_alpha:
            sse = 0.0
            count = 0
            for train_idx, validation_idx in data.cv_folds:
                fold_features = linear_features(data, train_idx).values
                x_fit = _design(fold_features[train_idx])
                x_validation = _design(fold_features[validation_idx])
                if similarity:
                    weights = _sgwr_weights(
                        data.coords[train_idx],
                        fold_features[train_idx],
                        data.coords[validation_idx],
                        fold_features[validation_idx],
                        float(b),
                        geo_scale,
                        float(alpha),
                    )
                else:
                    weights = _gwr_weights(
                        data.coords[train_idx],
                        data.coords[validation_idx],
                        float(b),
                        geo_scale,
                    )
                prediction = _weighted_least_squares_predictions(
                    x_fit, data.y[train_idx], x_validation, weights
                )
                residual = data.y[validation_idx] - prediction
                sse += float(residual @ residual)
                count += len(validation_idx)
            scores.append({"b": float(b), "alpha": float(alpha), "mse": sse / count})
    best = min(scores, key=lambda item: (item["mse"], item["b"], -item["alpha"]))
    return {**best, "geographical_scale": geo_scale, "candidates": scores}


def fit_gwr(data: SimulationData, *, pilot: bool = False) -> dict:
    """Spatial-block selection followed by an official PySAL GWR fit."""

    started = time.perf_counter()
    selected = _spatial_cv_gwr(data, similarity=False, pilot=pilot)
    fit_idx, test_idx = data.model_idx, data.test_idx
    features = linear_features(data, fit_idx).values
    raw_bandwidth = selected["b"] * selected["geographical_scale"]
    model = GWR(
        data.coords[fit_idx],
        data.y[fit_idx, None],
        features[fit_idx],
        raw_bandwidth,
        fixed=True,
        kernel="gaussian",
        constant=True,
        n_jobs=1,
    )
    result = model.fit()
    prediction_test = model.predict(
        data.coords[test_idx],
        features[test_idx],
        exog_scale=result.scale,
        exog_resid=result.resid_response,
    ).predictions.ravel()
    return {
        "train_prediction": result.predy.ravel(),
        "test_prediction": prediction_test,
        "selected": selected,
        "seconds": time.perf_counter() - started,
        "implementation": "mgwr.gwr.GWR 2.2.1",
    }


def fit_sgwr(data: SimulationData, *, pilot: bool = False) -> dict:
    """Fit the published FastSGWR geographical/similarity weight mixture."""

    started = time.perf_counter()
    selected = _spatial_cv_gwr(data, similarity=True, pilot=pilot)
    fit_idx, test_idx = data.model_idx, data.test_idx
    features = linear_features(data, fit_idx).values
    x_fit = _design(features[fit_idx])
    x_test = _design(features[test_idx])
    w_fit = _sgwr_weights(
        data.coords[fit_idx], features[fit_idx], data.coords[fit_idx], features[fit_idx],
        selected["b"], selected["geographical_scale"], selected["alpha"]
    )
    w_test = _sgwr_weights(
        data.coords[fit_idx], features[fit_idx], data.coords[test_idx], features[test_idx],
        selected["b"], selected["geographical_scale"], selected["alpha"]
    )
    return {
        "train_prediction": _weighted_least_squares_predictions(
            x_fit, data.y[fit_idx], x_fit, w_fit
        ),
        "test_prediction": _weighted_least_squares_predictions(
            x_fit, data.y[fit_idx], x_test, w_test
        ),
        "selected": selected,
        "seconds": time.perf_counter() - started,
        "implementation": "FastSGWR 0.5 equations with corrected held-out prediction",
    }




