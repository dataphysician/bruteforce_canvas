from __future__ import annotations

import math
from typing import Any

import pytest

from bruteforce_canvas.gp import (
    ComboGPModel,
    ComboGPState,
    encode_combo_signature,
    encode_coordinate,
    gp_posterior_update,
    gp_predict,
    gp_uncertainty_decay,
    gpytorch_available,
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


def test_encode_coordinate_is_deterministic_and_sensitive_to_values() -> None:
    first = encode_coordinate({"axis_b": "value_2", "axis_a": "value_1"})
    second = encode_coordinate({"axis_a": "value_1", "axis_b": "value_2"})
    changed = encode_coordinate({"axis_a": "value_9", "axis_b": "value_2"})

    assert first == second
    assert first != changed
    assert all(isinstance(value, float) for value in first)


def test_encode_combo_signature_is_deterministic_and_order_insensitive() -> None:
    first = encode_combo_signature("material=CERAMIC|lighting=BLUE_HOUR")
    second = encode_combo_signature("lighting=BLUE_HOUR|material=CERAMIC")
    changed = encode_combo_signature("lighting=STUDIO|material=CERAMIC")

    assert len(first) == 16
    assert first == second
    assert first != changed


def test_gp_uncertainty_decay_matches_spec_formula() -> None:
    assert gp_uncertainty_decay(0) == 1.0
    assert gp_uncertainty_decay(1) == pytest.approx(1 / math.sqrt(2))
    assert gp_uncertainty_decay(3) == 0.5


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


def test_combo_gp_model_import_guard_without_ml_extra() -> None:
    if gpytorch_available():
        pytest.skip("guard path only applies when gpytorch is absent")

    with pytest.raises(ImportError):
        ComboGPModel(None, None, None)


def test_gp_posterior_update_returns_mean_and_variance_shape() -> None:
    train_x = [[0.0, 0.0], [1.0, 1.0]]
    train_y = [0.0, 1.0]
    test_x = [[0.5, 0.5]]

    mean, variance = gp_posterior_update(train_x, train_y, test_x)

    assert isinstance(mean, float)
    assert isinstance(variance, float)
    assert math.isfinite(mean)
    assert math.isfinite(variance)
    assert variance >= 0.0


def test_gp_posterior_update_is_deterministic() -> None:
    train_x = [[0.0, 0.0], [1.0, 1.0], [0.25, 0.75]]
    train_y = [-0.5, 0.5, 0.1]
    test_x = [[0.25, 0.75]]

    first = gp_posterior_update(train_x, train_y, test_x)
    second = gp_posterior_update(train_x, train_y, test_x)

    assert first == pytest.approx(second)


def test_gp_posterior_update_validates_training_shapes() -> None:
    with pytest.raises(ValueError, match="same number"):
        gp_posterior_update([[0.0], [1.0]], [0.0], [[0.0]])
    with pytest.raises(ValueError, match="match X_train dimensions"):
        gp_posterior_update([[0.0, 1.0]], [0.0], [[0.0]])


def test_gp_predict_returns_mean_and_variance_after_observations() -> None:
    state = ComboGPState(likelihood_noise=0.05)
    first = encode_coordinate({"lighting": "blue_hour", "material": "ceramic"})
    second = encode_coordinate({"lighting": "studio", "material": "leather"})
    state = state.model_copy(update={"train_x": [first, second], "train_y": [0.2, 0.9]})

    mean, variance = gp_predict(state, {"lighting": "blue_hour", "material": "ceramic"})

    assert math.isfinite(mean)
    assert math.isfinite(variance)
    assert variance >= 0.0
