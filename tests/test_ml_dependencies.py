from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("PIL")


def test_ml_dependencies_importable() -> None:
    """Smoke test confirming the optional [ml] dependencies are installed."""
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import safetensors  # noqa: F401
    from PIL import Image  # noqa: F401


def test_gpytorch_importable() -> None:
    pytest.importorskip("gpytorch")
    import gpytorch  # noqa: F401
