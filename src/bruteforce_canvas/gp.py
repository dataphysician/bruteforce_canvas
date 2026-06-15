from __future__ import annotations

import hashlib
import importlib
import math
from typing import Any

from pydantic import Field, model_validator

from bruteforce_canvas.shared import StrictModel

try:
    _gpytorch: Any | None = importlib.import_module("gpytorch")
    _torch: Any | None = importlib.import_module("torch")
except ImportError:
    _gpytorch = None
    _torch = None


FEATURE_DIMENSIONS = 16
HASH_PROJECTIONS_PER_PAIR = 4
MIN_LIKELIHOOD_NOISE = 1e-4


class ComboGPState(StrictModel):
    train_x: list[list[float]] = Field(default_factory=list)
    train_y: list[float] = Field(default_factory=list)
    likelihood_noise: float = Field(default=1e-2, gt=0.0)

    @model_validator(mode="after")
    def validate_training_data(self) -> ComboGPState:
        if not self.train_x and not self.train_y:
            return self
        _validate_feature_rows(self.train_x, self.train_y)
        return self


def gpytorch_available() -> bool:
    return _gpytorch is not None and _torch is not None


def _require_gpytorch() -> tuple[Any, Any]:
    if _gpytorch is None or _torch is None:
        raise ImportError("gpytorch and torch are required for GP prediction; install bruteforce-canvas[ml]")
    return _gpytorch, _torch


def encode_coordinate(coordinate: dict[str, str], *, dimensions: int = FEATURE_DIMENSIONS) -> list[float]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    features = [0.0] * dimensions
    if not coordinate:
        return features
    for axis, value in sorted(coordinate.items()):
        token = f"{axis}={value}"
        for projection in range(HASH_PROJECTIONS_PER_PAIR):
            digest = hashlib.sha256(f"{token}:{projection}".encode("utf-8")).digest()
            bucket = digest[0] % dimensions
            sign = 1.0 if digest[1] & 1 else -1.0
            magnitude = 0.5 + int.from_bytes(digest[2:4], "big") / 131070.0
            features[bucket] += sign * magnitude
    normalizer = max(1.0, float(len(coordinate) * HASH_PROJECTIONS_PER_PAIR))
    return [value / normalizer for value in features]


def encode_combo_signature(signature: str, *, dimensions: int = FEATURE_DIMENSIONS) -> list[float]:
    coordinate: dict[str, str] = {}
    for index, part in enumerate(piece for piece in signature.split("|") if piece):
        if "=" in part:
            axis, value = part.split("=", 1)
            axis = axis or f"combo_part_{index}"
        else:
            axis = f"combo_part_{index}"
            value = part
        key = axis if axis not in coordinate else f"{axis}#{index}"
        coordinate[key] = value
    if not coordinate and signature:
        coordinate["combo_signature"] = signature
    return encode_coordinate(coordinate, dimensions=dimensions)


_ExactGPBase: Any = _gpytorch.models.ExactGP if _gpytorch is not None else object


class ComboGPModel(_ExactGPBase):
    def __init__(self, train_x: Any, train_y: Any, likelihood: Any) -> None:
        gpytorch, _ = _require_gpytorch()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: Any) -> Any:
        gpytorch, _ = _require_gpytorch()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gp_uncertainty_decay(observations: int) -> float:
    if observations < 0:
        raise ValueError("observations must be non-negative")
    return 1.0 / math.sqrt(1.0 + float(observations))


def gp_posterior_update(X_train: Any, y_train: Any, X_test: Any) -> tuple[float, float]:
    train_rows = _as_feature_rows(X_train)
    train_targets = _as_target_values(y_train)
    test_rows = _as_feature_rows(X_test)
    _validate_feature_rows(train_rows, train_targets)
    _validate_test_rows(train_rows, test_rows)

    if not gpytorch_available():
        return _fallback_posterior(train_targets)

    gpytorch, torch = _require_gpytorch()
    torch.manual_seed(0)
    train_x = torch.as_tensor(train_rows, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_targets, dtype=torch.float32)
    test_x = torch.as_tensor(test_rows, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = max(MIN_LIKELIHOOD_NOISE, float(MIN_LIKELIHOOD_NOISE))
    model: Any = ComboGPModel(train_x, train_y_tensor, likelihood)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
        prediction = model(test_x)
        mean = float(prediction.mean.reshape(-1)[0].item())
        variance = max(0.0, float(prediction.variance.reshape(-1)[0].item()))
    return mean, variance


def gp_predict(state: ComboGPState, coordinate: dict[str, str]) -> tuple[float, float]:
    if not state.train_x:
        raise ValueError("gp_predict requires at least one training observation")
    test_x = [encode_coordinate(coordinate, dimensions=len(state.train_x[0]))]
    return gp_posterior_update(state.train_x, state.train_y, test_x)


def _as_feature_rows(values: Any) -> list[list[float]]:
    raw = _to_python(values)
    if not isinstance(raw, list):
        raise TypeError("feature data must be a 2D tensor or list of rows")
    if not raw:
        raise ValueError("feature data must contain at least one row")
    if all(isinstance(value, int | float) for value in raw):
        return [[float(value) for value in raw]]
    rows: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list):
            raise TypeError("feature rows must be lists of numeric values")
        rows.append([float(value) for value in row])
    return rows


def _as_target_values(values: Any) -> list[float]:
    raw = _to_python(values)
    if isinstance(raw, int | float):
        return [float(raw)]
    if not isinstance(raw, list):
        raise TypeError("target data must be a 1D tensor or list of numeric values")
    return [float(value) for value in raw]


def _to_python(values: Any) -> Any:
    if hasattr(values, "detach") and callable(values.detach):
        detached: Any = values.detach()
        return detached.cpu().tolist()
    if hasattr(values, "tolist") and callable(values.tolist):
        return values.tolist()
    return values


def _validate_feature_rows(rows: list[list[float]], targets: list[float]) -> None:
    if len(rows) != len(targets):
        raise ValueError("X_train and y_train must contain the same number of observations")
    if not rows:
        raise ValueError("X_train must contain at least one observation")
    dimensions = len(rows[0])
    if dimensions == 0:
        raise ValueError("training feature vectors must not be empty")
    if any(len(row) != dimensions for row in rows):
        raise ValueError("all training feature vectors must have the same length")


def _validate_test_rows(train_rows: list[list[float]], test_rows: list[list[float]]) -> None:
    dimensions = len(train_rows[0])
    if any(len(row) != dimensions for row in test_rows):
        raise ValueError("X_test feature vectors must match X_train dimensions")


def _fallback_posterior(targets: list[float]) -> tuple[float, float]:
    mean = sum(targets) / float(len(targets))
    return mean, gp_uncertainty_decay(len(targets))


__all__ = [
    "ComboGPModel",
    "ComboGPState",
    "encode_combo_signature",
    "encode_coordinate",
    "gp_posterior_update",
    "gp_predict",
    "gp_uncertainty_decay",
    "gpytorch_available",
]
