"""Evaluation metrics and Monte Carlo summaries."""

from __future__ import annotations

import numpy as np


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape or y.ndim != 1:
        raise ValueError("y and prediction must be one-dimensional arrays of equal length")
    residual = y - prediction
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }
