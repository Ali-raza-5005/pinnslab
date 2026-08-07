"""Reduction of per-point residuals to a scalar loss.

DESIGN.md §4, decision 1: residual functions return per-point tensors of shape
``(N,)``, never scalars, and *all* reduction happens here. If residuals pre-reduce
then per-point weighting — causal, self-adaptive, RBA — becomes impossible
without editing every residual, which is precisely the trap this seam exists to
avoid.

The step-1 reducer is deliberately the dumbest one that could work. Real
weighting schemes are per-paper files registered with ``@register_weighting``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from pinnslab.components import WEIGHTINGS, register_weighting

if TYPE_CHECKING:
    from pinnslab.training.trainer import TrainState


@register_weighting("mean")
class MeanWeighting:
    """``sum_k coeff_k * mean(residual_k ** 2)``.

    Missing coefficients default to 1.0, so a config need only name the terms it
    wants to reweight.
    """

    def __init__(self, coefficients: dict[str, float] | None = None) -> None:
        self.coefficients = dict(coefficients or {})

    def __call__(
        self, residuals: dict[str, torch.Tensor], state: TrainState
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for name, value in residuals.items():
            term = self.coefficients.get(name, 1.0) * (value**2).mean()
            total = term if total is None else total + term
        if total is None:
            raise ValueError("residual_fn returned no terms; nothing to minimise")
        return total

    def __repr__(self) -> str:
        return f"MeanWeighting({self.coefficients!r})"


def build_weighting(name: str, **kwargs: Any) -> Any:
    return WEIGHTINGS.get(name)(**kwargs)


__all__ = ["MeanWeighting", "build_weighting"]
