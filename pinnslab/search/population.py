"""Evaluating a whole population in one graph (DESIGN.md §6).

Why this is not ``vmap``, which §6 prescribed
---------------------------------------------
§6 specifies ``torch.func.vmap`` over ``stack_module_state`` +
``functional_call``. Measured 2026-08-08: **that does not compose with a PINN
residual.** A residual differentiates the network with respect to its *inputs*,
which means flagging the collocation points with ``requires_grad_()`` — and
``vmap`` refuses outright::

    RuntimeError: You are attempting to call Tensor.requires_grad_() (or
    perhaps using torch.autograd.functional.* APIs) inside of a function
    being transformed by a function transform

Making it work would mean rewriting every residual against
``torch.func.jacrev``/``hessian`` — a second way to spell every PDE, and
exactly the "monkey-patching for every paper" DESIGN.md §1 rejects.

The fix keeps the goal and drops the mechanism: **put the population on a
leading batch dimension and build one graph.** An :class:`Ensemble` evaluates P
independent MLPs as batched matmuls, so inputs are ``(P, N, d)``, outputs are
``(P, N, m)``, and plain ``torch.autograd.grad`` works unchanged — including
second derivatives. Residuals do not know the population exists.

Three facts make this correct rather than merely convenient, all measured:

1. **Independence.** ``grad_outputs=ones`` sums before differentiating, and
   output element ``(p, n)`` depends only on candidate ``p``'s parameters and
   point ``(p, n)`` — every cross term is identically zero. Batched and
   separate evaluation agree to **0.0e+00** on a Burgers residual.
2. **One Adam is P Adams.** Adam is elementwise and its state is per-element,
   so a single optimizer over stacked ``(P, ...)`` parameters *is* P
   independent Adams, provided the loss handed to ``backward`` is the **sum**
   over candidates. Measured drift after 25 steps: 1.1e-16.
3. **Speed.** Measured on this CPU with a real Burgers residual (width 20,
   depth 3, N=512): 1.7x at P=4, 2.8x at P=8, **3.4x at P=16**, then falling
   back to ~2.2x by P=50. §6's "20-50x on a T4" is a *GPU* claim about kernel
   launch overhead dominating for tiny nets and is **untested** — no GPU here.
   Do not quote it in a paper until it is measured on the hardware in question;
   compute parity is a reviewer defence and an unverified speedup is a hole.

What breaks independence
------------------------
Anything that reduces across the population. Global gradient-norm clipping
couples every candidate (clip per candidate or not at all), as would a shared
loss scale, an aggregate early stop, or batch normalisation over the batched
dim. :func:`train_population` refuses the clipping case rather than silently
producing a coupled search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

#: ``(ensemble, points) -> (P, N)`` per-point residuals, one row per candidate.
#:
#: The residual runs its **own** forward pass, exactly as the single-run
#: ``ResidualFn`` does (it takes a ``TrainState`` and reads ``state.nets``).
#: That is not an accident of style: a PINN residual differentiates the network
#: with respect to its inputs, so it needs the forward pass inside its own
#: graph. Handing it a precomputed output tensor would work only until a term
#: needed a second network or a different point group.
PopulationResidual = Callable[["Ensemble", torch.Tensor], torch.Tensor]


class Ensemble(nn.Module):
    """P independent MLPs of identical shape, evaluated as batched matmuls.

    Identical shape is the binding constraint, and it is the same one §6 noted
    for vmap: architecture search over varying width or depth does not batch
    directly, so group candidates by shape and batch within a group. Sampling,
    loss-weighting and activation search all batch cleanly.
    """

    def __init__(
        self,
        members: Sequence[nn.Module],
        *,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        layers = [_linears(m) for m in members]
        shapes = {
            tuple((lin.in_features, lin.out_features) for lin in ls)
            for ls in layers
        }
        if len(shapes) != 1:
            raise ValueError(
                f"every member must have the same shape to batch; got {len(shapes)} "
                "distinct architectures. Group candidates by shape and build one "
                "Ensemble per group (DESIGN.md §6)."
            )

        self.size = len(members)
        self.depth = len(layers[0])
        self.activation = activation
        # The members' own parameter names, so member_state_dict round-trips
        # through load_state_dict. Deriving them from the layer index instead
        # would be wrong for any container that interleaves modules — an
        # nn.Sequential puts its Linears at 0, 2, 4, not 0, 1, 2.
        self.member_names: tuple[str, ...] = tuple(
            name for name, module in members[0].named_modules()
            if isinstance(module, nn.Linear)
        )
        # Transposed once here so the hot loop is baddbmm rather than a
        # transpose per layer per step.
        self.weights = nn.ParameterList(
            nn.Parameter(torch.stack([ls[d].weight.detach().T for ls in layers]))
            for d in range(self.depth)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.stack([ls[d].bias.detach() for ls in layers]))
            for d in range(self.depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(P, N, d_in) -> (P, N, d_out)``."""
        if x.ndim != 3 or x.shape[0] != self.size:
            raise ValueError(
                f"expected points shaped ({self.size}, N, d), got {tuple(x.shape)}"
            )
        for index in range(self.depth):
            x = torch.baddbmm(self.biases[index].unsqueeze(1), x, self.weights[index])
            if index < self.depth - 1:
                x = self.activation(x)
        return x

    def member_state_dict(self, index: int) -> dict[str, torch.Tensor]:
        """Candidate ``index``'s parameters, in the layout a plain MLP wants.

        So a winning candidate can be pulled out of the ensemble and retrained,
        checkpointed or plotted through the ordinary single-run path.
        """
        state = {}
        for depth, name in enumerate(self.member_names):
            state[f"{name}.weight"] = self.weights[depth][index].detach().T.clone()
            state[f"{name}.bias"] = self.biases[depth][index].detach().clone()
        return state


def _linears(module: nn.Module) -> list[nn.Linear]:
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise ValueError(f"{type(module).__name__} has no Linear layers to batch")
    return layers


@dataclass
class PopulationResult:
    """Per-candidate outcome of a batched training run."""

    losses: torch.Tensor  # (P,), final per-candidate loss
    steps: int

    def tolist(self) -> list[float]:
        return [float(v) for v in self.losses]


def train_population(
    ensemble: Ensemble,
    points: torch.Tensor,
    residual: PopulationResidual,
    *,
    steps: int,
    lr: float = 1e-3,
    resample: Callable[[int], torch.Tensor] | None = None,
    max_grad_norm: float | None = None,
) -> PopulationResult:
    """Train every candidate simultaneously for ``steps`` Adam steps.

    ``points`` is ``(P, N, d)`` — each candidate carries its **own** collocation
    cloud, which is the whole point when the search axis is sampling.
    ``residual`` receives ``(ensemble, points)`` and returns ``(P, N)``.

    ``max_grad_norm`` is refused, loudly. Global-norm clipping computes one norm
    over every parameter in the call, which across a stacked population is one
    norm over all P candidates: a single diverging candidate would then shrink
    everyone else's step, and the population would stop being independent
    trainings. That is a silent coupling that produces a plausible search, so it
    is an error rather than a warning.
    """
    if max_grad_norm is not None:
        raise ValueError(
            "global gradient-norm clipping couples the population: one norm "
            "over all P candidates means a single diverging candidate damps "
            "every other one's step. Clip inside the residual per candidate, "
            "or leave it off."
        )
    if points.ndim != 3 or points.shape[0] != ensemble.size:
        raise ValueError(
            f"points must be ({ensemble.size}, N, d), got {tuple(points.shape)}"
        )

    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)

    for step in range(steps):
        if resample is not None:
            points = resample(step)
        losses = _losses(ensemble, points, residual)
        optimizer.zero_grad(set_to_none=True)
        # sum, never mean: the derivative of a sum keeps each candidate's
        # gradient exactly what it would be alone, while a mean would scale
        # every gradient by 1/P and make the effective learning rate depend on
        # the population size.
        losses.sum().backward()
        optimizer.step()

    # One more forward pass, so the reported fitness belongs to the parameters
    # actually returned. Taking the last in-loop loss would report the value at
    # theta(steps-1) while the ensemble holds theta(steps) — for a *trainer*
    # that is an off-by-one in a trace point, but here the number IS the search
    # signal, so a candidate would be selected on a score its own weights never
    # produced. Costs one residual evaluation per rung against `steps` of them.
    with torch.enable_grad():  # the residual differentiates wrt its inputs
        final = _losses(ensemble, points, residual).detach()
    return PopulationResult(losses=final, steps=steps)


def _losses(
    ensemble: Ensemble, points: torch.Tensor, residual: PopulationResidual
) -> torch.Tensor:
    """Mean squared residual per candidate, shaped ``(P,)``."""
    x = points.detach().requires_grad_(True)
    residuals = residual(ensemble, x)
    if residuals.shape[:1] != (ensemble.size,):
        raise ValueError(
            f"residual must return (P, N) with P={ensemble.size}, got "
            f"{tuple(residuals.shape)} (CLAUDE.md rule 5: per-point, and "
            "here per-candidate too)"
        )
    return (residuals**2).mean(dim=tuple(range(1, residuals.ndim)))


__all__ = [
    "Ensemble",
    "PopulationResidual",
    "PopulationResult",
    "train_population",
]
