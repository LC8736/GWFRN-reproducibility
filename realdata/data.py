"""Basis and distance routines shared with the main-simulation methods.

The real-data functional inputs have 365 daily observations per curve.
No simulation data are generated during the empirical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ModelName = Literal["model1", "model3"]
ErrorName = Literal["normal", "student_t", "bimodal"]

M = 40
N = M * M
N_T = 365
T_GRID = np.arange(1.0, N_T + 1.0, dtype=np.float64)
DELTA_T = 1.0
QUAD_WEIGHTS = np.full(N_T, DELTA_T, dtype=np.float64)
QUAD_WEIGHTS[[0, -1]] *= 0.5
MODEL_NAMES: tuple[ModelName, ...] = (
    "model1", "model3"
)
ERROR_NAMES: tuple[ErrorName, ...] = ("normal", "student_t", "bimodal")
ANALYSIS_BASIS_DIM = 30
ANALYSIS_INTERIOR_KNOTS = tuple(
    np.linspace(T_GRID[0], T_GRID[-1], ANALYSIS_BASIS_DIM - 2)[1:-1]
)


@dataclass(frozen=True)
class SimulationData:
    coords: FloatArray
    x: FloatArray
    latent_x: FloatArray
    bspline_coef: FloatArray
    z: FloatArray
    y: FloatArray
    mu: FloatArray
    eta: FloatArray
    zeta: FloatArray
    interaction_kappa: float
    block_ids: IntArray
    model_idx: IntArray
    test_idx: IntArray
    cv_folds: tuple[tuple[IntArray, IntArray], ...]
    seed: int
    model: str
    error: str


def _open_cubic_bspline_basis(interior_knots: tuple[float, ...]) -> FloatArray:
    """Evaluate the complete open cubic B-spline basis on ``T_GRID``."""

    degree = 3
    knots = np.r_[
        np.full(degree + 1, T_GRID[0]),
        interior_knots,
        np.full(degree + 1, T_GRID[-1]),
    ]
    n_basis = len(knots) - degree - 1
    basis = np.empty((N_T, n_basis), dtype=np.float64)
    for k in range(n_basis):
        coefficient = np.zeros(n_basis)
        coefficient[k] = 1.0
        basis[:, k] = BSpline(knots, coefficient, degree, extrapolate=False)(T_GRID)
    return basis





def analysis_bspline_basis() -> FloatArray:
    """Quadrature-orthonormalised 30-dimensional cubic B-spline basis.

    The underlying space on days 1--365 has 26 equally spaced interior
    knots. A symmetric inverse square root of their Gram
    matrix provides an orthonormal coordinate system under the same
    trapezoidal inner product used throughout the simulation.
    """

    raw = _open_cubic_bspline_basis(ANALYSIS_INTERIOR_KNOTS)
    gram = raw.T @ (raw * QUAD_WEIGHTS[:, None])
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    inverse_root = (eigenvector / np.sqrt(eigenvalue)) @ eigenvector.T
    return raw @ inverse_root


def analysis_bspline_roughness() -> FloatArray:
    """Return the trace-normalised second-derivative roughness matrix.

    If ``beta(t) = sum_m c_m phi_m(t)``, this matrix satisfies
    ``c.T @ Omega @ c = integral beta''(t)^2 dt / (trace(Omega0) / M)``, where
    ``Omega0`` is the unnormalised roughness matrix. This fixed normalisation
    makes the curvature-penalty grid interpretable without altering which coefficient
    functions are preferred for a given positive penalty multiplier.
    """

    degree = 3
    knots = np.r_[
        np.full(degree + 1, T_GRID[0]),
        ANALYSIS_INTERIOR_KNOTS,
        np.full(degree + 1, T_GRID[-1]),
    ]
    raw = _open_cubic_bspline_basis(ANALYSIS_INTERIOR_KNOTS)
    gram = raw.T @ (raw * QUAD_WEIGHTS[:, None])
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    inverse_root = (eigenvector / np.sqrt(eigenvalue)) @ eigenvector.T
    raw_second = np.empty((N_T, ANALYSIS_BASIS_DIM), dtype=np.float64)
    for k in range(ANALYSIS_BASIS_DIM):
        coefficient = np.zeros(ANALYSIS_BASIS_DIM)
        coefficient[k] = 1.0
        spline = BSpline(knots, coefficient, degree, extrapolate=False)
        raw_second[:, k] = spline.derivative(2)(T_GRID)
    second = raw_second @ inverse_root
    roughness = second.T @ (second * QUAD_WEIGHTS[:, None])
    roughness = 0.5 * (roughness + roughness.T)
    return roughness / (np.trace(roughness) / ANALYSIS_BASIS_DIM)


def bspline_coefficients(x: FloatArray) -> FloatArray:
    """Project observed curves onto the common 30-dimensional B-spline basis.

    Coefficients minimise the trapezoidally weighted squared reconstruction
    error. They are the fixed vector inputs used by GWR, MGWR, SGWR, GNNWR,
    SANNWR, and GWANN. The response is never used in this projection.
    """

    basis = analysis_bspline_basis()
    weighted_basis = QUAD_WEIGHTS[:, None] * basis
    coefficient = np.einsum("npt,tm->npm", x, weighted_basis, optimize=True)
    return coefficient.reshape(len(x), -1)


def functional_integrals(x: FloatArray) -> FloatArray:
    """Fixed integrals used by the trainable functional first layer.

    The coefficient functions are represented by 30 cubic B-splines. Learning
    a linear combination of these integrals is
    algebraically identical to learning the spline coefficients of each
    functional neuron.
    """

    basis = analysis_bspline_basis()
    integral = np.einsum("npt,t,tk->npk", x, QUAD_WEIGHTS, basis)
    return integral.reshape(len(x), -1)








def pairwise_euclidean(left: FloatArray, right: FloatArray) -> FloatArray:
    squared = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :]
    squared -= 2.0 * left @ right.T
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def distance_components(
    data: SimulationData,
    left_idx: IntArray,
    right_idx: IntArray,
    *,
    covariate_subset: tuple[str, ...] = ("x1", "x2", "z"),
    functional_scales: FloatArray | None = None,
    scalar_scales: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return geographical and scale-balanced covariate distances.

    Each functional predictor contributes the mean-domain L2 distance between
    its projections onto the common orthonormalised M=30 B-spline space,
    divided by a training-sample scale. Each scalar predictor contributes its
    absolute difference divided by a training-sample scale. The functional and
    scalar blocks are aggregated by root mean squares and then weighted
    equally. Scale values must be estimated from the fitting observations via
    :func:`covariate_distance_scales` and reused for held-out locations.
    """

    geographic = pairwise_euclidean(data.coords[left_idx], data.coords[right_idx])
    functional_distances: list[FloatArray] = []
    # L2 distances are computed between the curves projected onto the same
    # 30-dimensional B-spline space used by all fitted models. If a(t)=B(t)c,
    # then ||a_i-a_r||_L2 equals the Euclidean distance between c_i L and c_r L,
    # where G=int B(t)B(t)^T dt=L L^T is the B-spline Gram matrix.
    basis = analysis_bspline_basis()
    gram = basis.T @ (basis * QUAD_WEIGHTS[:, None])
    chol = np.linalg.cholesky(gram)
    coefficient = data.bspline_coef.reshape(len(data.x), data.x.shape[1], -1)
    transformed = coefficient @ chol
    selected_functional: list[int] = []
    if "x1" in covariate_subset:
        selected_functional.append(0)
    if "x2" in covariate_subset:
        selected_functional.append(1)
    if selected_functional:
        if functional_scales is None:
            raise ValueError("functional_scales must be estimated from fitting data")
        if len(functional_scales) != len(selected_functional):
            raise ValueError("functional_scales has incompatible length")
        domain_length = max(float(T_GRID[-1] - T_GRID[0]), 1e-12)
        for position, j in enumerate(selected_functional):
            distance = pairwise_euclidean(
                transformed[left_idx, j], transformed[right_idx, j]
            ) / np.sqrt(domain_length)
            functional_distances.append(distance / functional_scales[position])

    scalar_distances: list[FloatArray] = []
    if "z" in covariate_subset:
        if scalar_scales is None:
            raise ValueError("scalar_scales must be estimated from fitting data")
        if len(scalar_scales) != data.z.shape[1]:
            raise ValueError("scalar_scales has incompatible length")
        for k in range(data.z.shape[1]):
            scalar_distances.append(
                np.abs(
                    data.z[left_idx, k, None] - data.z[right_idx, k][None, :]
                ) / scalar_scales[k]
            )

    blocks: list[FloatArray] = []
    if functional_distances:
        blocks.append(
            np.sqrt(np.mean(np.square(np.stack(functional_distances)), axis=0))
        )
    if scalar_distances:
        blocks.append(np.sqrt(np.mean(np.square(np.stack(scalar_distances)), axis=0)))
    if not blocks:
        raise ValueError("covariate_subset must contain at least one component")
    covariate = np.mean(np.stack(blocks), axis=0)
    same = left_idx[:, None] == right_idx[None, :]
    geographic[same] = 0.0
    covariate[same] = 0.0
    return geographic, covariate


def _positive_median(values: FloatArray) -> float:
    positive = values[values > 1e-12]
    if positive.size == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-12)


def covariate_distance_scales(
    data: SimulationData,
    fit_idx: IntArray,
    *,
    covariate_subset: tuple[str, ...] = ("x1", "x2", "z"),
) -> tuple[FloatArray, FloatArray]:
    """Estimate projected-curve and scalar distance scales from fitting data only."""

    basis = analysis_bspline_basis()
    gram = basis.T @ (basis * QUAD_WEIGHTS[:, None])
    chol = np.linalg.cholesky(gram)
    coefficient = data.bspline_coef.reshape(len(data.x), data.x.shape[1], -1)
    transformed = coefficient @ chol
    domain_length = max(float(T_GRID[-1] - T_GRID[0]), 1e-12)
    functional: list[float] = []
    for label, j in (("x1", 0), ("x2", 1)):
        if label in covariate_subset:
            distance = pairwise_euclidean(
                transformed[fit_idx, j], transformed[fit_idx, j]
            ) / np.sqrt(domain_length)
            functional.append(_positive_median(distance))
    scalar: list[float] = []
    if "z" in covariate_subset:
        for k in range(data.z.shape[1]):
            difference = np.abs(
                data.z[fit_idx, k, None] - data.z[fit_idx, k][None, :]
            )
            scalar.append(_positive_median(difference))
    return np.asarray(functional, dtype=np.float64), np.asarray(scalar, dtype=np.float64)
