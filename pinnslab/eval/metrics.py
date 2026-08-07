"""Metrics and evaluation grids.

The headline number for a forward PINN is relative L2 against a reference
solution, so it is defined once, here, and never re-derived in a paper repo —
two papers reporting "rel-L2" computed slightly differently is a silent
comparability failure that no reviewer would catch and no test would either.
"""

from __future__ import annotations

import torch

from pinnslab.geometry import Domain


def relative_l2(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """``||predicted - reference||_2 / ||reference||_2``.

    The denominator is the norm of the *reference*, not of the error, so the
    number is comparable across problems of different magnitude. Flattened
    before comparison, so ``(N,)`` and ``(N, 1)`` are interchangeable — the
    shape mismatch they would otherwise cause is a broadcast into an ``(N, N)``
    error matrix, which produces a plausible-looking number rather than an
    exception.
    """
    predicted = predicted.reshape(-1)
    reference = reference.reshape(-1)
    if predicted.shape != reference.shape:
        raise ValueError(
            f"predicted has {predicted.numel()} values and reference has "
            f"{reference.numel()}; they must describe the same points"
        )

    denominator = torch.linalg.vector_norm(reference)
    if denominator == 0:
        raise ValueError(
            "reference solution is identically zero, so relative error is "
            "undefined; use an absolute norm for this problem"
        )
    return float(torch.linalg.vector_norm(predicted - reference) / denominator)


def l2_error(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Absolute RMS error, for problems whose reference can be zero."""
    predicted = predicted.reshape(-1)
    reference = reference.reshape(-1)
    return float(torch.sqrt(torch.mean((predicted - reference) ** 2)))


def max_error(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst-case pointwise error — where a smooth-looking PINN actually fails.

    Reported alongside rel-L2 because the two disagree exactly where it
    matters: a solution that is excellent everywhere except across a steep
    front has a fine L2 and a terrible max.
    """
    return float(torch.max(torch.abs(predicted.reshape(-1) - reference.reshape(-1))))


def uniform_grid(
    domain: Domain,
    resolution: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A tensor-product grid over the domain's bounding box, shaped ``(prod, d)``.

    Evaluation grids are deliberately *not* drawn from the sampler: a metric
    computed on random points would move between runs, and comparing two
    methods would then be comparing two point clouds as much as two solutions.

    Only valid for box domains — on a non-convex or CSG geometry this returns
    points outside the domain, where a reference solution is undefined.
    """
    if len(resolution) != domain.dim:
        raise ValueError(
            f"resolution has {len(resolution)} entries but the domain has "
            f"{domain.dim} coordinates"
        )
    if any(n < 2 for n in resolution):
        raise ValueError(f"every axis needs at least 2 points, got {resolution}")

    lower, upper = domain.bounds(dtype=dtype, device=device)
    axes = [
        torch.linspace(float(lo), float(hi), n, dtype=lower.dtype, device=device)
        for lo, hi, n in zip(lower, upper, resolution, strict=True)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, domain.dim
    )


__all__ = ["l2_error", "max_error", "relative_l2", "uniform_grid"]
