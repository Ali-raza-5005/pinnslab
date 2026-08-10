"""Differential operators for PDE residuals.

One API, two backends is the eventual plan (DESIGN.md §5): classic
``torch.autograd.grad(create_graph=True)`` and ``torch.func.jacrev``/``hessian``
under ``vmap``. Only the autograd backend exists so far, deliberately — the
whole point of the second one is that it might be faster, and "might be faster"
is a measurement, not a design. The A/B needs a real benchmark to run on, which
is exactly what DESIGN.md §9 step 2 is building. The functions here take no
backend argument yet; adding one later is a keyword with a default, not a
rewrite of every residual.

The shape convention is fixed and worth stating once, because getting it wrong
produces a residual that trains happily to a meaningless number:

* network inputs ``x`` are ``(N, d)`` and **must** carry ``requires_grad=True``
  before the forward pass — a derivative cannot be recovered afterwards;
* network outputs ``y`` are ``(N, m)``;
* :func:`gradient` returns ``(N, d)``, one row per point;
* residuals hand back ``(N,)`` (CLAUDE.md rule 5), so they squeeze at the end.
"""

from __future__ import annotations

import torch


def requires_grad(points: torch.Tensor) -> torch.Tensor:
    """Return ``points`` ready to be differentiated with respect to.

    A leaf tensor is returned as-is once flagged; a non-leaf (already part of a
    graph, e.g. points produced by a sampler that itself differentiates) is
    detached first, because the residual wants derivatives with respect to the
    coordinates, not with respect to whatever produced them.
    """
    if points.requires_grad and points.is_leaf:
        return points
    return points.detach().requires_grad_(True)


def gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    *,
    component: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """``d outputs[..., component] / d inputs``, shaped like ``inputs``.

    Accepts ``(N, m)`` outputs against ``(N, d)`` inputs, and equally
    ``(P, N, m)`` against ``(P, N, d)`` — the search layer's batched population
    (DESIGN.md §6). Index derivatives with ``[..., i : i + 1]`` so a residual
    written once works in both.

    ``create_graph=True`` is the default because a PINN residual is itself
    differentiated — once to build a second derivative, and again by the
    optimizer through the whole residual. Pass ``False`` only for a quantity
    that is being reported rather than trained on.

    The ``grad_outputs=ones`` trick sums over the batch before differentiating.
    That is exact rather than an approximation: each row of ``outputs`` depends
    only on the matching row of ``inputs``, so the cross terms it would drop are
    identically zero, and one backward pass recovers all ``N`` gradients.
    """
    if outputs.ndim not in (2, 3):
        raise ValueError(
            f"outputs must be (N, m) or (P, N, m), got {tuple(outputs.shape)}; a "
            "residual that already reduced its outputs cannot be differentiated "
            "per point"
        )
    if not inputs.requires_grad:
        raise ValueError(
            "inputs do not require grad, so no derivative exists. Wrap the "
            "collocation points in diffops.requires_grad() *before* the forward "
            "pass — flagging them afterwards is too late, the graph is gone."
        )

    # Ellipsis, not a leading colon, so one residual serves both a single run
    # ``(N, m)`` and a whole search population ``(P, N, m)``. The maths is
    # unchanged: element ``(p, n)`` depends only on candidate p's parameters and
    # point ``(p, n)``, so the cross terms the grad_outputs=ones trick drops are
    # still identically zero (DESIGN.md §6, and measured in
    # tests/unit/test_search_population.py).
    selected = outputs[..., component : component + 1]
    (grad,) = torch.autograd.grad(
        selected,
        inputs,
        grad_outputs=torch.ones_like(selected),
        create_graph=create_graph,
        retain_graph=True,
    )
    return grad


def partial(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    wrt: int,
    *,
    component: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """A single first derivative ``d outputs[:, component] / d inputs[:, wrt]``.

    Shaped ``(N, 1)`` so it can be fed straight back into :func:`gradient` to
    build a second derivative.
    """
    grad = gradient(
        outputs, inputs, component=component, create_graph=create_graph
    )
    return grad[..., wrt : wrt + 1]


def second_partial(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    wrt: int,
    *,
    component: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """``d^2 outputs[:, component] / d inputs[:, wrt]^2``, shaped ``(N, 1)``.

    Note for callers computing several derivatives: this recomputes the first
    gradient internally. When a residual needs both ``u_x`` and ``u_xx`` — most
    of them do — call :func:`gradient` once and differentiate its column, which
    is one backward pass cheaper per step:

    .. code-block:: python

        du = gradient(u, x)                    # (N, d), all first derivatives
        u_xx = gradient(du[:, 0:1], x)[:, 0]   # reuses the graph
    """
    first = partial(outputs, inputs, wrt, component=component, create_graph=True)
    return gradient(first, inputs, component=0, create_graph=create_graph)[
        ..., wrt : wrt + 1
    ]


def laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    *,
    dims: tuple[int, ...] | None = None,
    component: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """``sum_i d^2 outputs / d inputs[:, i]^2`` over ``dims``, shaped ``(N, 1)``.

    ``dims`` defaults to every input coordinate. On a space-time domain that is
    almost never what you want — pass the spatial dimensions explicitly, since
    time is not part of a Laplacian.
    """
    first = gradient(outputs, inputs, component=component, create_graph=True)
    axes = tuple(range(inputs.shape[-1])) if dims is None else dims
    total = None
    for axis in axes:
        second = gradient(
            first[..., axis : axis + 1], inputs, component=0, create_graph=create_graph
        )[..., axis : axis + 1]
        total = second if total is None else total + second
    if total is None:
        raise ValueError("laplacian needs at least one dimension")
    return total


__all__ = [
    "gradient",
    "laplacian",
    "partial",
    "requires_grad",
    "second_partial",
]
