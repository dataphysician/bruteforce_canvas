from __future__ import annotations

import hashlib
import importlib
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
TRAINING_ITERATIONS = 25
MIN_LIKELIHOOD_NOISE = 1e-4


class ComboGPState(StrictModel):
    train_x: list[list[float]] = Field(default_factory=list)
    train_y: list[float] = Field(default_factory=list)
    likelihood_noise: float = Field(default=1e-2, gt=0.0)

    @model_validator(mode="after")
    def validate_training_data(self) -> ComboGPState:
        if len(self.train_x) != len(self.train_y):
            raise ValueError("train_x and train_y must contain the same number of observations")
        if not self.train_x:
            return self
        dimensions = len(self.train_x[0])
        if dimensions == 0:
            raise ValueError("training feature vectors must not be empty")
        if any(len(row) != dimensions for row in self.train_x):
            raise ValueError("all training feature vectors must have the same length")
        return self


def _require_gpytorch() -> tuple[Any, Any]:
    if _gpytorch is None or _torch is None:
        raise ImportError("gpytorch and torch are required for combo GP prediction; install bruteforce-canvas[ml]")
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


def gp_posterior_update(state: ComboGPState, coordinate: dict[str, str], outcome: float) -> ComboGPState:
    dimensions = len(state.train_x[0]) if state.train_x else FEATURE_DIMENSIONS
    encoded = encode_coordinate(coordinate, dimensions=dimensions)
    return ComboGPState(
        train_x=[*state.train_x, encoded],
        train_y=[*state.train_y, float(outcome)],
        likelihood_noise=state.likelihood_noise,
    )


def gp_predict(state: ComboGPState, coordinate: dict[str, str]) -> tuple[float, float]:
    gpytorch, torch = _require_gpytorch()
    if not state.train_x:
        raise ValueError("gp_predict requires at least one training observation")

    torch.manual_seed(0)
    train_x = torch.tensor(state.train_x, dtype=torch.float32)
    train_y = torch.tensor(state.train_y, dtype=torch.float32)
    test_x = torch.tensor(
        [encode_coordinate(coordinate, dimensions=len(state.train_x[0]))],
        dtype=torch.float32,
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = max(state.likelihood_noise, MIN_LIKELIHOOD_NOISE)
    model: Any = ComboGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with gpytorch.settings.lazily_evaluate_kernels(False):
        for _ in range(TRAINING_ITERATIONS):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -marginal_log_likelihood(output, train_y)
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
        prediction = likelihood(model(test_x))
        mean = float(prediction.mean.squeeze().item())
        variance = max(0.0, float(prediction.variance.squeeze().item()))
    return mean, variance


__all__ = [
    "ComboGPModel",
    "ComboGPState",
    "encode_coordinate",
    "gp_posterior_update",
    "gp_predict",
]
