from __future__ import annotations

import math
from typing import Any

import pytest

from bruteforce_canvas.combo_gp import (
    ComboGPModel,
    ComboGPState,
    encode_coordinate,
    gp_posterior_update,
    gp_predict,
)


def test_combo_gp_state_can_be_created_and_serialized() -> None:
    state = ComboGPState(
        train_x=[[0.1, 0.2, 0.3]],
        train_y=[0.7],
        likelihood_noise=0.05,
    )

    dumped = state.model_dump()
    assert dumped == {
        "train_x": [[0.1, 0.2, 0.3]],
        "train_y": [0.7],
        "likelihood_noise": 0.05,
    }
    assert '"likelihood_noise":0.05' in state.model_dump_json()


def test_gp_posterior_update_appends_training_points() -> None:
    state = ComboGPState()
    coordinate = {"material": "ceramic", "lighting": "blue_hour"}

    updated = gp_posterior_update(state, coordinate, 0.8)
    repeated = gp_posterior_update(state, {"lighting": "blue_hour", "material": "ceramic"}, 0.8)

    assert state.train_x == []
    assert len(updated.train_x) == 1
    assert len(updated.train_x[0]) == 16
    assert updated.train_y == [0.8]
    assert updated.train_x == repeated.train_x


def test_encode_coordinate_is_deterministic_and_sensitive_to_values() -> None:
    first = encode_coordinate({"axis_b": "value_2", "axis_a": "value_1"})
    second = encode_coordinate({"axis_a": "value_1", "axis_b": "value_2"})
    changed = encode_coordinate({"axis_a": "value_9", "axis_b": "value_2"})

    assert first == second
    assert first != changed
    assert all(isinstance(value, float) for value in first)


def test_combo_gp_model_uses_required_gpytorch_components() -> None:
    gpytorch = pytest.importorskip("gpytorch")
    torch = pytest.importorskip("torch")

    train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    train_y = torch.tensor([0.0, 1.0], dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model: Any = ComboGPModel(train_x, train_y, likelihood)

    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.RBFKernel)


def test_gp_predict_returns_mean_and_variance_after_observations() -> None:
    pytest.importorskip("gpytorch")
    pytest.importorskip("torch")
    state = ComboGPState(likelihood_noise=0.05)
    state = gp_posterior_update(state, {"lighting": "blue_hour", "material": "ceramic"}, 0.2)
    state = gp_posterior_update(state, {"lighting": "studio", "material": "leather"}, 0.9)

    mean, variance = gp_predict(state, {"lighting": "blue_hour", "material": "ceramic"})

    assert math.isfinite(mean)
    assert math.isfinite(variance)
    assert variance >= 0.0
