"""Neural benchmarks and the two proposed representation networks.

The FNN first layer is implemented through fixed numerical inner products with
cubic B-splines and trainable basis coefficients, as in the public FuncNN
implementation.  GNNWR follows the public ``gnnwr`` package: a spatial-weight
neural network transforms distances to training reference locations into
multipliers of global OLS coefficients.  SANNWR reproduces the SANNWR paper's
SAPDNN--SWNN architecture by adding a shared 2--3--1 spatial/attribute
proximity network before the 50--20 coefficient-weighting network. Functional
regularisation and proximity helpers are also used by GWFRN and SGWFRN.
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
from settings import CONFIG
from array_utils import mean_absolute_attribute_distance

from data import (
    ANALYSIS_BASIS_DIM,
    SimulationData,
    analysis_bspline_roughness,
    covariate_distance_scales,
    distance_components,
    functional_integrals,
    pairwise_euclidean,
)


SMOOTHNESS_VALUES = tuple(CONFIG["smoothness"])
BANDWIDTH_VALUES = tuple(CONFIG["bandwidths"])
ALPHA_VALUES = tuple(CONFIG["similarity_alphas"])
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
    def __init__(self, n_inputs: int):
        super().__init__()
        self.first = _linear(n_inputs, 64)
        self.second = _linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.second(torch.relu(self.first(x))))


class FunctionalNetwork(nn.Module):
    """Shared functional encoder followed by one or more prediction heads.

    Every head receives the 32-dimensional output of
    :class:`FunctionalEncoder`.  This is the GWFRN architecture defined in the
    manuscript: locations share the encoder, and their differences enter
    exclusively through the location-indexed output heads.
    """

    representation_dim = 32

    def __init__(self, n_inputs: int, n_outputs: int = 1):
        super().__init__()
        self.encoder = FunctionalEncoder(n_inputs)
        self.head = _linear(self.representation_dim, n_outputs, relu=False)

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
        a_train = mean_absolute_attribute_distance(scaled.x[train_idx], scaled.x[train_idx])
        a_validation = mean_absolute_attribute_distance(scaled.x[validation_idx], scaled.x[train_idx])
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
        final_train_attribute = mean_absolute_attribute_distance(final_scaled.x[fit_idx], final_scaled.x[fit_idx])
        final_test_attribute = mean_absolute_attribute_distance(final_scaled.x[test_idx], final_scaled.x[fit_idx])
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




def fit_sannwr(data: SimulationData, config: NeuralConfig, seed: int) -> dict:
    return _fit_coefficient_network(data, config, seed, similarity_guided=True)




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








