from __future__ import annotations

from typing import Any, Callable

GENERATOR_REGISTRY: dict[str, Callable[[Any], Any]] = {}
BUILDER_INCLUDES: set[str] = {"stub", "bonsai"}


def register(name: str, factory: Callable[..., Any]) -> None:
    GENERATOR_REGISTRY[name] = factory


def register_generator(kind_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        GENERATOR_REGISTRY[kind_name] = factory
        return factory

    return decorator
