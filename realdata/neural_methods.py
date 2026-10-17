"""Neural benchmarks and the two proposed representation networks.

The FNN first layer is implemented through fixed numerical inner products with
cubic B-splines and trainable basis coefficients, as in the public FuncNN
implementation.  GNNWR follows the public ``gnnwr`` package: a spatial-weight
neural network transforms distances to training reference locations into
multipliers of global OLS coefficients.  SANNWR reproduces the SANNWR paper's
SAPDNN--SWNN architecture by adding a shared 2--3--1 spatial/attribute
proximity network before the 50--20 coefficient-weighting network. GWFRN and
SGWFRN use a common 30-dimensional spline space and functional regularisation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from network_import import load_networks_only
load_networks_only()
from gnnwr.networks import SWNN as OfficialSWNN

from data import (
    ANALYSIS_BASIS_DIM,
    SimulationData,
    analysis_bspline_roughness,
    covariate_distance_scales,
    distance_components,
    functional_integrals,
    pairwise_euclidean,
)


GWFRN_IMPLEMENTATION_VERSION = (
    "encoder_only_32d_m30_days1to365_normalised_curvature_headnormalised_v1"
)
PROPOSED_REPRESENTATION_DIM = 32
SMOOTHNESS_VALUES = (1e-3, 1e-2, 1e-1)
BANDWIDTH_VALUES = (0.25, 0.50, 1.00, 1.50)
ALPHA_VALUES = (0.30, 0.60, 0.90, 1.00)
CURVATURE_DENOMINATOR_FLOOR = 1e-24


@dataclass(frozen=True)
class NeuralConfig:
    epochs: int = 3000
    batch_size: int = 50
    learning_rate: float = 0.005
    patience: int = 100
    device: str = "cpu"
    threads: int = 1
    pilot: bool = False


@dataclass
class StandardisedData:
    x: np.ndarray
    y: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float


@dataclass(frozen=True)
class ProximityScales:
    geographical: float
    covariate: float
    functional: np.ndarray
    scalar: np.ndarray


def _standardise(raw_x: np.ndarray, y: np.ndarray, fit_idx: np.ndarray) -> StandardisedData:
    x_mean = raw_x[fit_idx].mean(axis=0)
    x_scale = raw_x[fit_idx].std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = float(y[fit_idx].mean())
    y_scale = max(float(y[fit_idx].std()), 1e-12)
    return StandardisedData(
        x=(raw_x - x_mean) / x_scale,
        y=(y - y_mean) / y_scale,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )


def _configure(config: NeuralConfig, seed: int) -> None:
    torch.set_num_threads(config.threads)
    torch.manual_seed(seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _reset_parameters(model: nn.Module) -> None:
    """Reset a constructed module after the deterministic seed is fixed."""

    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.PReLU)):
            module.reset_parameters()


def _linear(in_features: int, out_features: int, *, relu: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu" if relu else "linear")
    nn.init.zeros_(layer.bias)
    return layer


class FunctionalEncoder(nn.Module):
    def __init__(self, n_inputs: int, representation_dim: int = 32):
        super().__init__()
        self.first = _linear(n_inputs, 64)
        self.second = _linear(64, representation_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.second(torch.relu(self.first(x))))


class FunctionalNetwork(nn.Module):
    """Shared functional encoder followed by one or more prediction heads.

    GWFRN heads receive a 32-dimensional shared representation. The global
    FNN benchmark also uses a 32-dimensional second hidden layer.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int = 1, *, representation_dim: int = 32
    ):
        super().__init__()
        self.encoder = FunctionalEncoder(n_inputs, representation_dim)
        self.head = _linear(representation_dim, n_outputs, relu=False)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.representation(x))






class CoefficientWeightNetwork(nn.Module):
    """GNNWR's distance-to-coefficient-weight network."""

    def __init__(self, n_references: int, n_coefficients: int):
        super().__init__()
        # Use the public gnnwr package's SWNN module directly. The layer widths
        # and dropout rate are those reported in the comparison protocol.
        self.network = OfficialSWNN(
            dense_layer=[50, 20],
            insize=n_references,
            outsize=n_coefficients,
            drop_out=0.70,
            activate_func=nn.PReLU(init=0.4),
            batch_norm=True,
        )

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return self.network(distance)


class SpatialAttributeCoefficientNetwork(nn.Module):
    """SANNWR/SANNWR: SAPDNN (2--3--1) followed by SWNN (R--50--20--p)."""

    def __init__(self, n_references: int, n_coefficients: int):
        super().__init__()
        self.sapdnn = nn.Sequential(
            # One shared PReLU slope is required because the pairwise tensor
            # has shape (batch, reference, channel).
            _linear(2, 3), nn.PReLU(1, init=0.4), _linear(3, 1, relu=False)
        )
        self.swnn = CoefficientWeightNetwork(n_references, n_coefficients)

    def forward(self, pairwise_distance: torch.Tensor) -> torch.Tensor:
        unified = self.sapdnn(pairwise_distance).squeeze(-1)
        return self.swnn(unified)


def _train_supervised(
    model: nn.Module,
    train_inputs,
    train_y: np.ndarray,
    validation_inputs,
    validation_y: np.ndarray,
    config: NeuralConfig,
    seed: int,
    prediction_fn,
    *,
    learning_rate: float | None = None,
) -> tuple[nn.Module, int, float]:
    """Mini-batch Adam with validation-based early stopping."""

    _configure(config, seed)
    device = torch.device(config.device)
    model = model.to(device)
    _reset_parameters(model)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate if learning_rate is None else learning_rate
    )
    tensors_train = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in train_inputs)
    tensors_validation = tuple(
        torch.as_tensor(x, dtype=torch.float32, device=device) for x in validation_inputs
    )
    y_train_t = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    y_validation_t = torch.as_tensor(validation_y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed + 7919)
    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    wait = 0
    for epoch in range(config.epochs):
        model.train()
        for start in range(0, len(train_y), config.batch_size):
            if start == 0:
                order = rng.permutation(len(train_y))
            index = torch.as_tensor(
                order[start : start + config.batch_size], dtype=torch.long, device=device
            )
            optimiser.zero_grad(set_to_none=True)
            batch_inputs = tuple(value[index] for value in tensors_train)
            prediction = prediction_fn(model, batch_inputs)
            loss = torch.mean((prediction - y_train_t[index]) ** 2)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = prediction_fn(model, tensors_validation)
            validation_loss = float(torch.mean((validation_prediction - y_validation_t) ** 2))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("neural optimisation produced no finite validation loss")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def _refit_fixed_epochs(
    model: nn.Module,
    inputs,
    y: np.ndarray,
    config: NeuralConfig,
    seed: int,
    prediction_fn,
    epochs: int,
    *,
    learning_rate: float | None = None,
) -> nn.Module:
    _configure(config, seed)
    device = torch.device(config.device)
    model = model.to(device)
    _reset_parameters(model)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate if learning_rate is None else learning_rate
    )
    tensors = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in inputs)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed + 104729)
    for _ in range(max(1, epochs)):
        order = rng.permutation(len(y))
        model.train()
        for start in range(0, len(y), config.batch_size):
            index = torch.as_tensor(
                order[start : start + config.batch_size], dtype=torch.long, device=device
            )
            optimiser.zero_grad(set_to_none=True)
            prediction = prediction_fn(model, tuple(value[index] for value in tensors))
            loss = torch.mean((prediction - y_t[index]) ** 2)
            loss.backward()
            optimiser.step()
    model.eval()
    return model


def fit_fnn(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    """Fit the functional neural network benchmark with one global head."""

    started = time.perf_counter()
    raw = np.column_stack((functional_integrals(data.x), data.z))
    train_idx, validation_idx = data.cv_folds[0]
    scaled = _standardise(raw, data.y, train_idx)
    predictor = lambda model, values: model(values[0]).squeeze(1)
    tuned, best_epoch, validation_loss = _train_supervised(
        FunctionalNetwork(raw.shape[1]),
        (scaled.x[train_idx],),
        scaled.y[train_idx],
        (scaled.x[validation_idx],),
        scaled.y[validation_idx],
        config,
        seed,
        predictor,
    )
    del tuned
    final_scaled = _standardise(raw, data.y, data.model_idx)
    final = _refit_fixed_epochs(
        FunctionalNetwork(raw.shape[1]),
        (final_scaled.x[data.model_idx],),
        final_scaled.y[data.model_idx],
        config,
        seed + 1,
        predictor,
        best_epoch,
    )
    device = torch.device(config.device)
    with torch.no_grad():
        train_prediction = final(
            torch.as_tensor(final_scaled.x[data.model_idx], dtype=torch.float32, device=device)
        ).squeeze(1).cpu().numpy()
        test_prediction = final(
            torch.as_tensor(final_scaled.x[data.test_idx], dtype=torch.float32, device=device)
        ).squeeze(1).cpu().numpy()
    return {
        "train_prediction": final_scaled.y_mean + final_scaled.y_scale * train_prediction,
        "test_prediction": final_scaled.y_mean + final_scaled.y_scale * test_prediction,
        "selected": {"epochs": best_epoch, "validation_mse_scaled": validation_loss},
        "seconds": time.perf_counter() - started,
        "implementation": "PyTorch port of the FuncNN functional-inner-product layer",
    }


def _ols_coefficient(x_design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x_design, y, rcond=None)[0]


def _distance_to_reference(
    coords: np.ndarray, reference_coords: np.ndarray, scale: float
) -> np.ndarray:
    return pairwise_euclidean(coords, reference_coords) / scale


def _fit_coefficient_network(
    data: SimulationData,
    config: NeuralConfig,
    seed: int,
    *,
    similarity_guided: bool,
) -> dict:
    started = time.perf_counter()
    raw = np.column_stack((data.bspline_coef, data.z))
    train_idx, validation_idx = data.cv_folds[0]
    scaled = _standardise(raw, data.y, train_idx)
    x_design = np.column_stack((np.ones(len(data.y)), scaled.x))
    beta = _ols_coefficient(x_design[train_idx], scaled.y[train_idx])
    geo_train = pairwise_euclidean(data.coords[train_idx], data.coords[train_idx])
    geo_scale = max(float(np.max(geo_train)), 1e-12)
    d_train = _distance_to_reference(data.coords[train_idx], data.coords[train_idx], geo_scale)
    d_validation = _distance_to_reference(
        data.coords[validation_idx], data.coords[train_idx], geo_scale
    )
    if similarity_guided:
        a_train = np.mean(
            np.abs(scaled.x[train_idx, None, :] - scaled.x[None, train_idx, :]), axis=2
        )
        a_validation = np.mean(
            np.abs(scaled.x[validation_idx, None, :] - scaled.x[None, train_idx, :]), axis=2
        )
        train_inputs = (np.stack((d_train, a_train), axis=2), x_design[train_idx])
        validation_inputs = (
            np.stack((d_validation, a_validation), axis=2), x_design[validation_idx]
        )
        model_factory = lambda n_ref: SpatialAttributeCoefficientNetwork(n_ref, x_design.shape[1])
        predictor = lambda model, values: torch.sum(
            values[1] * model(values[0]) * torch.as_tensor(beta, dtype=torch.float32, device=values[1].device),
            dim=1,
        )
    else:
        train_inputs = (d_train, x_design[train_idx])
        validation_inputs = (d_validation, x_design[validation_idx])
        model_factory = lambda n_ref: CoefficientWeightNetwork(n_ref, x_design.shape[1])
        predictor = lambda model, values: torch.sum(
            values[1] * model(values[0]) * torch.as_tensor(beta, dtype=torch.float32, device=values[1].device),
            dim=1,
        )
    tuned, best_epoch, validation_loss = _train_supervised(
        model_factory(len(train_idx)),
        train_inputs,
        scaled.y[train_idx],
        validation_inputs,
        scaled.y[validation_idx],
        config,
        seed,
        predictor,
        learning_rate=0.02,
    )
    del tuned

    fit_idx, test_idx = data.model_idx, data.test_idx
    final_scaled = _standardise(raw, data.y, fit_idx)
    final_design = np.column_stack((np.ones(len(data.y)), final_scaled.x))
    final_beta = _ols_coefficient(final_design[fit_idx], final_scaled.y[fit_idx])
    final_geo = pairwise_euclidean(data.coords[fit_idx], data.coords[fit_idx])
    final_geo_scale = max(float(np.max(final_geo)), 1e-12)
    final_train_distance = final_geo / final_geo_scale
    final_test_distance = _distance_to_reference(
        data.coords[test_idx], data.coords[fit_idx], final_geo_scale
    )
    if similarity_guided:
        final_train_attribute = np.mean(
            np.abs(final_scaled.x[fit_idx, None, :] - final_scaled.x[None, fit_idx, :]), axis=2
        )
        final_test_attribute = np.mean(
            np.abs(final_scaled.x[test_idx, None, :] - final_scaled.x[None, fit_idx, :]), axis=2
        )
        final_inputs = (
            np.stack((final_train_distance, final_train_attribute), axis=2),
            final_design[fit_idx],
        )
        test_inputs = (
            np.stack((final_test_distance, final_test_attribute), axis=2),
            final_design[test_idx],
        )
    else:
        final_inputs = (final_train_distance, final_design[fit_idx])
        test_inputs = (final_test_distance, final_design[test_idx])

    # Rebind the frozen OLS coefficient used by the prediction closure.
    beta = final_beta
    final = _refit_fixed_epochs(
        model_factory(len(fit_idx)),
        final_inputs,
        final_scaled.y[fit_idx],
        config,
        seed + 1,
        predictor,
        best_epoch,
        learning_rate=0.02,
    )
    device = torch.device(config.device)
    tensor_train = tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in final_inputs)
    tensor_test = tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in test_inputs)
    with torch.no_grad():
        prediction_train = predictor(final, tensor_train).cpu().numpy()
        prediction_test = predictor(final, tensor_test).cpu().numpy()
    label = "SANNWR/SANNWR paper architecture" if similarity_guided else "GNNWR paper/package architecture"
    return {
        "train_prediction": final_scaled.y_mean + final_scaled.y_scale * prediction_train,
        "test_prediction": final_scaled.y_mean + final_scaled.y_scale * prediction_test,
        "selected": {"epochs": best_epoch, "validation_mse_scaled": validation_loss},
        "seconds": time.perf_counter() - started,
        "implementation": label,
    }


def fit_gnnwr(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    return _fit_coefficient_network(data, config, seed, similarity_guided=False)


def fit_sannwr(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    return _fit_coefficient_network(data, config, seed, similarity_guided=True)


def _proximity_weights(
    data: SimulationData,
    fit_idx: np.ndarray,
    target_idx: np.ndarray,
    b: float,
    alpha: float,
    scales: ProximityScales,
) -> np.ndarray:
    geographic, covariate = _scaled_proximity_components(
        data, fit_idx, target_idx, scales
    )
    return _weights_from_components(geographic, covariate, b, alpha)


def _scaled_proximity_components(
    data: SimulationData,
    fit_idx: np.ndarray,
    target_idx: np.ndarray,
    scales: ProximityScales,
) -> tuple[np.ndarray, np.ndarray]:
    geographic, covariate = distance_components(
        data,
        fit_idx,
        target_idx,
        functional_scales=scales.functional,
        scalar_scales=scales.scalar,
    )
    return geographic / scales.geographical, covariate / scales.covariate


def _weights_from_components(
    geographic: np.ndarray,
    covariate: np.ndarray,
    b: float,
    alpha: float,
) -> np.ndarray:
    mixed = alpha * geographic + (1.0 - alpha) * covariate
    return np.exp(-np.square(mixed / b))


def _distance_scales(data: SimulationData, fit_idx: np.ndarray) -> ProximityScales:
    functional_scale, scalar_scale = covariate_distance_scales(data, fit_idx)
    geographic, covariate = distance_components(
        data,
        fit_idx,
        fit_idx,
        functional_scales=functional_scale,
        scalar_scales=scalar_scale,
    )
    positive_g = geographic[geographic > 1e-12]
    positive_c = covariate[covariate > 1e-12]
    return ProximityScales(
        geographical=float(np.median(positive_g)),
        covariate=float(np.median(positive_c)),
        functional=functional_scale,
        scalar=scalar_scale,
    )


def _normalised_functional_curvature(
    model: FunctionalNetwork,
    functional_input_scales: torch.Tensor,
    roughness: torch.Tensor,
    *,
    n_functional_predictors: int = 2,
) -> torch.Tensor:
    """Average scale-invariant curvature of first-layer weight functions.

    The network receives standardized functional integrals. If ``w_ljm`` is a
    first-layer weight and ``s_jm`` is the corresponding training-fold input
    scale, the effective coefficient of ``phi_m`` is ``w_ljm / s_jm``.
    Each function contributes the Rayleigh quotient ``c' Omega c / c'c``.
    This makes the regularizer invariant both to input standardisation and to
    nonzero rescaling of a first-layer functional weight. Computation is
    performed in float64 and the denominator is floored at ``1e-24`` only to
    define a finite value at a numerically zero function. Exact scale
    invariance is retained whenever the effective coefficient norm exceeds
    ``1e-12``, far below its practical range under the stated initialisation.
    """

    expected = n_functional_predictors * ANALYSIS_BASIS_DIM
    if functional_input_scales.numel() != expected:
        raise ValueError("functional_input_scales has incompatible length")
    coefficients = model.encoder.first.weight[:, :expected].reshape(
        model.encoder.first.out_features,
        n_functional_predictors,
        ANALYSIS_BASIS_DIM,
    )
    effective = coefficients / functional_input_scales.reshape(
        1, n_functional_predictors, ANALYSIS_BASIS_DIM
    )
    effective64 = effective.to(torch.float64)
    roughness64 = roughness.to(torch.float64)
    numerator = torch.einsum(
        "hjm,mn,hjn->hj", effective64, roughness64, effective64
    )
    denominator = torch.sum(effective64.square(), dim=2).clamp_min(
        CURVATURE_DENOMINATOR_FLOOR
    )
    return torch.mean(numerator / denominator).to(coefficients.dtype)


def _normalise_joint_training_weights(weights: np.ndarray) -> np.ndarray:
    """Give every location head mean-one weights in the joint network loss.

    For ``n`` fitting observations, every returned column sums to ``n``. This
    removes the bandwidth-dependent scale of the weighted-error component and
    makes a common curvature-penalty grid comparable across ``b`` and
    ``alpha``. The transformation is restricted to joint neural training;
    conditional weighted least squares continues to use raw kernel weights.
    """

    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
        raise ValueError("joint-training weights must be a non-empty matrix")
    column_mean = np.mean(weights, axis=0)
    if np.any(~np.isfinite(column_mean)) or np.any(column_mean <= 0.0):
        raise ValueError("every location head must have a positive finite weight mean")
    return weights / column_mean[None, :]


def _train_weighted_network(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: NeuralConfig,
    seed: int,
    epochs: int,
    *,
    smoothness: float,
    functional_input_scales: np.ndarray,
) -> FunctionalNetwork:
    """Fit the shared encoder and all location-indexed heads jointly.

    Every candidate receives exactly ``epochs`` passes through its fitting
    observations.  Spatial validation is performed only after fitting, via a
    conditional head at each held-out location; validation responses therefore
    neither stop training nor enter the encoder or a training-location head.
    Before optimisation, the kernel weights of each training-location head are
    divided by their mean over fitting observations, so every head has total
    weight equal to the fitting sample size.

    The mini-batch criterion is the geographically weighted squared error plus
    ``smoothness / (2 * 64 * p)`` times the sum of the trace-normalised
    curvature Rayleigh quotients of the first-layer functional weight
    functions. Effective coefficients are recovered from the standardised
    network weights by division by ``functional_input_scales``. The response
    supplied here is standardised using fitting-fold statistics. No scalar,
    later hidden-layer, or location-head parameter enters this penalty.
    """

    _configure(config, seed)
    device = torch.device(config.device)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed + 15485863)
    model = FunctionalNetwork(
        x.shape[1], weights.shape[1],
        representation_dim=PROPOSED_REPRESENTATION_DIM,
    ).to(device)
    # ``_configure`` fixes the random seed before construction.  The linear
    # layers therefore use the He initialisation implemented in ``_linear``.
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    normalised_weights = _normalise_joint_training_weights(weights)
    w_t = torch.as_tensor(normalised_weights, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(
        functional_input_scales, dtype=torch.float32, device=device
    )
    roughness_t = torch.as_tensor(
        analysis_bspline_roughness(), dtype=torch.float32, device=device
    )
    for _ in range(epochs):
        order = rng.permutation(len(y))
        model.train()
        for start in range(0, len(y), config.batch_size):
            index = torch.as_tensor(
                order[start : start + config.batch_size], dtype=torch.long, device=device
            )
            optimiser.zero_grad(set_to_none=True)
            residual = model(x_t[index]) - y_t[index, None]
            # Equation (mini-batch GW loss) averages the weighted error over
            # both observations and active location-indexed heads.
            weighted_error = torch.sum(w_t[index] * residual.square()) / (
                2.0 * len(index) * weights.shape[1]
            )
            penalty = 0.5 * smoothness * _normalised_functional_curvature(
                model, scale_t, roughness_t
            )
            loss = weighted_error + penalty
            loss.backward()
            optimiser.step()
    model.eval()
    return model


def _encode(model: FunctionalNetwork, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        return model.representation(
            torch.as_tensor(x, dtype=torch.float32, device=device)
        ).cpu().numpy()


def _local_head_coefficients(
    h_fit: np.ndarray,
    y_fit: np.ndarray,
    weights: np.ndarray,
    *,
    penalty_factor: float = 1e-1,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(h_fit)), h_fit))
    coefficients = np.empty((weights.shape[1], design.shape[1]), dtype=np.float64)
    for r in range(weights.shape[1]):
        root = np.sqrt(weights[:, r])
        weighted_design = design * root[:, None]
        gram = weighted_design.T @ weighted_design
        scale = max(float(np.trace(gram[1:, 1:])) / (design.shape[1] - 1), 1.0)
        penalty_diagonal = np.r_[0.0, np.full(h_fit.shape[1], penalty_factor * scale)]
        penalty = np.diag(penalty_diagonal)
        coefficients[r] = np.linalg.solve(
            gram + penalty, weighted_design.T @ (y_fit * root)
        )
    return coefficients


def _conditional_head_prediction(
    h_fit: np.ndarray,
    y_fit: np.ndarray,
    h_target: np.ndarray,
    weights: np.ndarray,
    *,
    penalty_factor: float = 1e-1,
) -> np.ndarray:
    target_design = np.column_stack((np.ones(len(h_target)), h_target))
    coefficients = _local_head_coefficients(
        h_fit,
        y_fit,
        weights,
        penalty_factor=penalty_factor,
    )
    return np.sum(target_design * coefficients, axis=1)


def _multifidelity_stages(
    config: NeuralConfig,
    available_folds: int,
    *,
    similarity_guided: bool = True,
) -> tuple[dict, ...]:
    """Prespecified staged multi-fidelity schedule for joint spatial CV."""

    if config.pilot:
        specification = ((3, 10), (6, 30), (9, 80))
    else:
        specification = ((3, 20), (4, 60), (5, 200))
    retention = (12, 4, 1) if similarity_guided else (4, 2, 1)
    return tuple(
        {
            "folds": min(int(folds), available_folds),
            "epochs": max(1, min(int(epochs), config.epochs)),
            "retain": int(retain),
        }
        for (folds, epochs), retain in zip(specification, retention)
    )


def _joint_candidates(similarity_guided: bool) -> list[dict]:
    alpha_values = ALPHA_VALUES if similarity_guided else (1.0,)
    return [
        {
            "candidate_id": int(candidate_id),
            "b": float(b),
            "alpha": float(alpha),
            "lambda_smooth": float(smoothness),
        }
        for candidate_id, (b, alpha, smoothness) in enumerate(
            (b, alpha, smoothness)
            for b in BANDWIDTH_VALUES
            for alpha in alpha_values
            for smoothness in SMOOTHNESS_VALUES
        )
    ]


def _candidate_order(score: dict) -> tuple:
    return (
        score["mse"],
        score["b"],
        -score["alpha"],
        -score["lambda_smooth"],
        score["candidate_id"],
    )


def _evaluate_candidate_stage(
    data: SimulationData,
    config: NeuralConfig,
    seed: int,
    raw_features: np.ndarray,
    candidates: list[dict],
    stage: dict,
    conditional_penalty_factor: float,
    stage_number: int,
) -> list[dict]:
    accumulation = {
        item["candidate_id"]: {"sse": 0.0, "count": 0} for item in candidates
    }
    folds = data.cv_folds[: stage["folds"]]
    for fold_number, (train_idx, validation_idx) in enumerate(folds):
        scaled = _standardise(raw_features, data.y, train_idx)
        scales = _distance_scales(data, train_idx)
        train_geo, train_covariate = _scaled_proximity_components(
            data, train_idx, train_idx, scales
        )
        validation_geo, validation_covariate = _scaled_proximity_components(
            data, train_idx, validation_idx, scales
        )
        weight_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for candidate in candidates:
            weight_key = (candidate["b"], candidate["alpha"])
            if weight_key not in weight_cache:
                weight_cache[weight_key] = (
                    _weights_from_components(
                        train_geo,
                        train_covariate,
                        candidate["b"],
                        candidate["alpha"],
                    ),
                    _weights_from_components(
                        validation_geo,
                        validation_covariate,
                        candidate["b"],
                        candidate["alpha"],
                    ),
                )
            weights_train, weights_validation = weight_cache[weight_key]
            network = _train_weighted_network(
                scaled.x[train_idx],
                scaled.y[train_idx],
                weights_train,
                config,
                seed + fold_number,
                stage["epochs"],
                smoothness=candidate["lambda_smooth"],
                functional_input_scales=scaled.x_scale[
                    : 2 * ANALYSIS_BASIS_DIM
                ],
            )
            h_fit = _encode(network, scaled.x[train_idx], config.device)
            h_validation = _encode(
                network, scaled.x[validation_idx], config.device
            )
            prediction = _conditional_head_prediction(
                h_fit,
                data.y[train_idx],
                h_validation,
                weights_validation,
                penalty_factor=conditional_penalty_factor,
            )
            residual = data.y[validation_idx] - prediction
            entry = accumulation[candidate["candidate_id"]]
            entry["sse"] += float(residual @ residual)
            entry["count"] += len(validation_idx)
            del network
    return [
        {
            **candidate,
            "mse": accumulation[candidate["candidate_id"]]["sse"]
            / accumulation[candidate["candidate_id"]]["count"],
            "stage": int(stage_number),
            "folds": int(stage["folds"]),
            "epochs": int(stage["epochs"]),
        }
        for candidate in candidates
    ]


def _select_proximity(
    data: SimulationData,
    config: NeuralConfig,
    seed: int,
    *,
    similarity_guided: bool,
    raw_features: np.ndarray,
    conditional_penalty_factor: float,
) -> dict:
    """Jointly select ``(b, alpha, lambda)`` by staged spatial CV.

    Candidates are always ranked as complete triples; no coordinate is fixed
    and tuned conditionally on another. Successive stages increase both the
    number of contiguous spatial folds and the optimisation budget, retaining
    only a prespecified number of candidates after each stage.
    """

    active = _joint_candidates(similarity_guided)
    initial_count = len(active)
    stages = _multifidelity_stages(
        config, len(data.cv_folds), similarity_guided=similarity_guided
    )
    history: list[dict] = []
    final_scores: list[dict] = []
    for stage_number, stage in enumerate(stages, start=1):
        scores = _evaluate_candidate_stage(
            data,
            config,
            seed,
            raw_features,
            active,
            stage,
            conditional_penalty_factor,
            stage_number,
        )
        ranked = sorted(scores, key=_candidate_order)
        retained = ranked[: min(len(ranked), stage["retain"])]
        retained_ids = {item["candidate_id"] for item in retained}
        recorded_scores = [
            {**item, "retained": item["candidate_id"] in retained_ids}
            for item in ranked
        ]
        history.append(
            {
                "stage": int(stage_number),
                "folds": int(stage["folds"]),
                "epochs": int(stage["epochs"]),
                "evaluated": int(len(ranked)),
                "retained": int(len(retained)),
                "retained_candidate_ids": sorted(retained_ids),
                "scores": recorded_scores,
            }
        )
        active = [
            {
                key: value
                for key, value in candidate.items()
                if key in {"candidate_id", "b", "alpha", "lambda_smooth"}
            }
            for candidate in ranked
            if candidate["candidate_id"] in retained_ids
        ]
        final_scores = ranked
    best = min(final_scores, key=_candidate_order)
    return {
        **best,
        "initial_candidate_count": int(initial_count),
        "stage_history": history,
        "selection_method": "joint staged multi-fidelity spatial block CV",
    }


def _fit_gwfrn_family(
    data: SimulationData,
    config: NeuralConfig,
    seed: int,
    *,
    similarity_guided: bool,
    raw_features: np.ndarray,
    conditional_penalty_factor: float,
    implementation: str,
) -> dict:
    started = time.perf_counter()
    selected = _select_proximity(
        data,
        config,
        seed,
        similarity_guided=similarity_guided,
        raw_features=raw_features,
        conditional_penalty_factor=conditional_penalty_factor,
    )
    scaled = _standardise(raw_features, data.y, data.model_idx)
    scales = _distance_scales(data, data.model_idx)
    weights_train = _proximity_weights(
        data, data.model_idx, data.model_idx,
        selected["b"], selected["alpha"], scales
    )
    # Refit at the highest fidelity used to select the joint candidate. This
    # keeps the optimisation budget identical between validation and final fit.
    final_epochs = int(selected["epochs"])
    network = _train_weighted_network(
        scaled.x[data.model_idx], scaled.y[data.model_idx], weights_train,
        config,
        seed + 1_000_003,
        final_epochs,
        smoothness=selected["lambda_smooth"],
        functional_input_scales=scaled.x_scale[: 2 * ANALYSIS_BASIS_DIM],
    )
    device = torch.device(config.device)
    with torch.no_grad():
        fitted_matrix = network(
            torch.as_tensor(scaled.x[data.model_idx], dtype=torch.float32, device=device)
        ).cpu().numpy()
    # Output head r corresponds to the rth regression location in model_idx.
    fitted_standardised = np.diag(fitted_matrix)
    train_prediction = scaled.y_mean + scaled.y_scale * fitted_standardised
    h_fit = _encode(network, scaled.x[data.model_idx], config.device)
    h_test = _encode(network, scaled.x[data.test_idx], config.device)
    weights_test = _proximity_weights(
        data, data.model_idx, data.test_idx,
        selected["b"], selected["alpha"], scales
    )
    test_prediction = _conditional_head_prediction(
        h_fit,
        data.y[data.model_idx],
        h_test,
        weights_test,
        penalty_factor=conditional_penalty_factor,
    )
    with torch.no_grad():
        functional_scale_t = torch.as_tensor(
            scaled.x_scale[: 2 * ANALYSIS_BASIS_DIM],
            dtype=torch.float32,
            device=device,
        )
        roughness_t = torch.as_tensor(
            analysis_bspline_roughness(), dtype=torch.float32, device=device
        )
        average_curvature = float(
            _normalised_functional_curvature(
                network, functional_scale_t, roughness_t
            ).cpu()
        )
        standardised_functional_weight = network.encoder.first.weight[
            :, : 2 * ANALYSIS_BASIS_DIM
        ]
        effective_functional_weight = (
            standardised_functional_weight
            / functional_scale_t.reshape(1, -1)
        )
        weight_diagnostics = {
            "average_normalised_functional_curvature": average_curvature,
            "curvature_denominator_floor": CURVATURE_DENOMINATOR_FLOOR,
            "minimum_effective_function_squared_norm": float(
                effective_functional_weight.reshape(
                    network.encoder.first.out_features, 2, ANALYSIS_BASIS_DIM
                ).square().sum(dim=2).min().cpu()
            ),
            "standardised_functional_weight_frobenius": float(
                torch.linalg.vector_norm(standardised_functional_weight).cpu()
            ),
            "effective_functional_weight_frobenius": float(
                torch.linalg.vector_norm(effective_functional_weight).cpu()
            ),
            "second_layer_weight_frobenius": float(
                torch.linalg.vector_norm(network.encoder.second.weight).cpu()
            ),
        }
    return {
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "selected": {
            **selected,
            "final_epochs": final_epochs,
            "joint_training_weight_normalisation": (
                "each head divided by its fitting-sample mean; column sum equals n_fit"
            ),
            "weight_diagnostics": weight_diagnostics,
        },
        "seconds": time.perf_counter() - started,
        "implementation": implementation,
    }


def fit_gwfrn(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    raw = np.column_stack((functional_integrals(data.x), data.z))
    return _fit_gwfrn_family(
        data,
        config,
        seed,
        similarity_guided=False,
        raw_features=raw,
        conditional_penalty_factor=1e-1,
        implementation=(
            "M=30 curvature-regularised 32-dimensional shared functional representation with "
            "location-indexed geographical heads, joint staged spatial CV, and "
            "ridge-stabilised conditional heads"
        ),
    )


def fit_sgwfrn(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    raw = np.column_stack((functional_integrals(data.x), data.z))
    return _fit_gwfrn_family(
        data,
        config,
        seed,
        similarity_guided=True,
        raw_features=raw,
        conditional_penalty_factor=1e-1,
        implementation=(
            "M=30 curvature-regularised 32-dimensional shared functional representation with "
            "scale-balanced similarity-guided location heads, joint staged "
            "spatial CV, and ridge-stabilised conditional heads"
        ),
    )








