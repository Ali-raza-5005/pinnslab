"""Component registration — extension by registration, not inheritance.

DESIGN.md §4: a new method is one new file with a ``@register_*`` decorator and
zero edits to existing files.

Naming note: this module is deliberately NOT ``pinnslab/registry/``. DESIGN.md §3
reserves that package for *run provenance* (Run object, config hashing, results
schema). The two senses of "registry" are unrelated; keeping them in separate
modules avoids the collision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> factory mapping populated by decorator at import time."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(
                    f"{self.kind} {key!r} is already registered by "
                    f"{self._items[key]!r}. Registration keys must be unique; "
                    f"pick a different name in your paper repo."
                )
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {key!r}; registered: {sorted(self._items)}"
            ) from None

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {sorted(self._items)})"


MODELS: Registry[Callable[..., Any]] = Registry("model")
OPTIMIZERS: Registry[Callable[..., Any]] = Registry("optimizer")
PROBLEMS: Registry[Callable[..., Any]] = Registry("problem")
RESIDUALS: Registry[Callable[..., Any]] = Registry("residual")
SAMPLERS: Registry[Callable[..., Any]] = Registry("sampler")
WEIGHTINGS: Registry[Callable[..., Any]] = Registry("weighting")

register_model = MODELS.register
register_optimizer = OPTIMIZERS.register
register_problem = PROBLEMS.register
register_residual = RESIDUALS.register
register_sampler = SAMPLERS.register
register_weighting = WEIGHTINGS.register

__all__ = [
    "MODELS",
    "OPTIMIZERS",
    "PROBLEMS",
    "RESIDUALS",
    "SAMPLERS",
    "WEIGHTINGS",
    "Registry",
    "register_model",
    "register_optimizer",
    "register_problem",
    "register_residual",
    "register_sampler",
    "register_weighting",
]
