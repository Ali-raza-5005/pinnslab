"""Config -> Trainer. The only supported way to start a real run.

The ``Trainer`` takes networks and a residual function as plain callables, which
is the escape hatch DESIGN.md §4 requires and what the infrastructure tests use.
This module is the other path: it builds those callables from a validated,
hashed :class:`RunConfig` and refuses to invent anything the config did not
declare, which is what keeps CLAUDE.md rule 4 ("no hyperparameter is ever a
Python literal in a script") true in practice rather than in principle.

A Kaggle notebook is then the ~20 lines of DESIGN.md §7: load a config, call
:func:`build_trainer`, call ``.fit()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import pinnslab.benchmarks  # noqa: F401  (registers the built-in problems)
import pinnslab.losses  # noqa: F401  (registers the built-in weightings)
from pinnslab.benchmarks.problem import Problem, ResidualTerm, build_problem
from pinnslab.components import RESIDUALS, WEIGHTINGS
from pinnslab.eval.metrics import max_error, relative_l2, uniform_grid
from pinnslab.models.mlp import build_net
from pinnslab.registry.config import RunConfig
from pinnslab.registry.run import Run
from pinnslab.training.trainer import (
    EvalFn,
    HookFn,
    ResidualFn,
    Trainer,
    TrainState,
    WeightingFn,
)
from pinnslab.utils.device import RuntimeContext
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

#: Where collocation points live between the resample hook and the residual
#: function. Keyed by point-group name.
POINTS = "points"


@dataclass
class Assembly:
    """Everything a config asked for, built but not yet running."""

    problem: Problem
    nets: dict[str, nn.Module]
    extra_params: dict[str, torch.Tensor]
    residual_fn: ResidualFn
    weighting: WeightingFn
    on_resample: HookFn
    eval_fn: EvalFn | None


def assemble(cfg: RunConfig, ctx: RuntimeContext) -> Assembly:
    """Turn a validated config into the callables the trainer needs."""
    if not cfg.residuals:
        raise ValueError(
            "this config declares no residuals, so there is nothing to train. "
            "Either declare problem/nets/residuals/sampling, or drive Trainer "
            "directly with your own callables."
        )
    if cfg.problem is None:
        raise ValueError(
            "this config declares residuals but no problem; the residual terms "
            "and the reference solution both come from the benchmark"
        )

    problem = build_problem(cfg.problem)
    nets = _build_nets(cfg, ctx)
    extra_params = _build_extra_params(cfg, ctx)
    terms = {
        name: RESIDUALS.get(spec.kind)(spec, problem)
        for name, spec in cfg.residuals.items()
    }

    return Assembly(
        problem=problem,
        nets=nets,
        extra_params=extra_params,
        residual_fn=_make_residual_fn(cfg, terms),
        weighting=_build_weighting(cfg),
        on_resample=_make_resampler(cfg, problem, ctx),
        eval_fn=_make_eval_fn(problem, ctx),
    )


def build_trainer(
    cfg: RunConfig, ctx: RuntimeContext, run: Run, **trainer_kwargs
) -> Trainer:
    """``RunConfig`` -> a ``Trainer`` with its first point cloud already drawn."""
    parts = assemble(cfg, ctx)
    trainer = Trainer(
        cfg=cfg,
        ctx=ctx,
        run=run,
        nets=parts.nets,
        extra_params=parts.extra_params,
        residual_fn=parts.residual_fn,
        weighting=parts.weighting,
        on_resample=parts.on_resample,
        eval_fn=parts.eval_fn,
        **trainer_kwargs,
    )

    # The trainer only fires ``on_resample`` when a stage sets
    # ``resample_every``; with fixed points it would never fire at all and the
    # first residual evaluation would find an empty scratch. Drawing here also
    # means the initial cloud comes from the generator's *initial* state, which
    # a resumed run reproduces exactly (``_restore`` overwrites the stream
    # afterwards, so subsequent resamples continue from the checkpoint).
    parts.on_resample(trainer.state)
    return trainer


# -- pieces -------------------------------------------------------------------


def _build_nets(cfg: RunConfig, ctx: RuntimeContext) -> dict[str, nn.Module]:
    nets = {}
    for name, spec in cfg.nets.items():
        net = build_net(spec).to(device=ctx.device, dtype=ctx.dtype)
        log.info(
            "net %r: %s, %d parameters",
            name,
            spec.arch,
            sum(p.numel() for p in net.parameters()),
        )
        nets[name] = net
    return nets


def _build_extra_params(
    cfg: RunConfig, ctx: RuntimeContext
) -> dict[str, torch.Tensor]:
    """Inverse-problem unknowns (DESIGN.md §4 conformance item 2)."""
    return {
        name: torch.full(
            tuple(spec.shape),
            spec.init,
            device=ctx.device,
            dtype=ctx.dtype,
            requires_grad=spec.trainable,
        )
        for name, spec in cfg.extra_params.items()
    }


def _build_weighting(cfg: RunConfig) -> WeightingFn:
    """Every weighting takes ``coefficients``; the rest is scheme-specific."""
    spec = cfg.weighting
    return WEIGHTINGS.get(spec.kind)(
        coefficients=dict(spec.coefficients), **spec.options
    )


def _make_residual_fn(cfg: RunConfig, terms: dict[str, ResidualTerm]):
    """Evaluate every declared term on its own point group.

    Reads points from ``state.scratch`` rather than drawing them, because
    L-BFGS's line search can invoke the closure more than once per step and a
    residual that resampled itself would move the objective underneath it.
    """
    groups = {name: spec.points for name, spec in cfg.residuals.items()}

    def residual_fn(state: TrainState) -> dict[str, torch.Tensor]:
        points = state.scratch.get(POINTS)
        if not points:
            raise RuntimeError(
                "no collocation points in state.scratch; build the trainer with "
                "build_trainer(), which draws the first cloud before training"
            )
        return {name: terms[name](state, points[groups[name]]) for name in terms}

    return residual_fn


def _make_resampler(cfg: RunConfig, problem: Problem, ctx: RuntimeContext):
    """Draw every declared point group from the trainer's own RNG stream."""
    specs = cfg.sampling.points

    def on_resample(state: TrainState) -> None:
        state.scratch[POINTS] = {
            name: problem.domain.sample(
                spec.region,
                spec.n,
                generator=state.generator,
                strategy=spec.strategy,
                dtype=ctx.dtype,
                device=ctx.device,
            )
            for name, spec in specs.items()
        }

    return on_resample


def _make_eval_fn(problem: Problem, ctx: RuntimeContext) -> EvalFn | None:
    """rel-L2 and max error against the benchmark's reference solution.

    On a fixed grid, computed once: DESIGN.md §8's comparability requires the
    metric to be a property of the solution, not of a point cloud that moves
    between runs. Returns ``None`` when the benchmark has no reference, so a
    problem without ground truth trains and reports residuals only, instead of
    failing at the first trace point.
    """
    if problem.reference is None or not problem.eval_resolution:
        return None

    grid = uniform_grid(
        problem.domain, problem.eval_resolution, dtype=ctx.dtype, device=ctx.device
    )
    truth = problem.reference_at(grid)
    field = problem.solution_net

    def eval_fn(state: TrainState) -> dict[str, float]:
        # no_grad is safe here and nowhere near the residuals: this is a plain
        # forward evaluation with no derivatives in it.
        with torch.no_grad():
            predicted = state.nets[field](grid)
        return {
            "rel_l2": relative_l2(predicted, truth),
            "max_error": max_error(predicted, truth),
        }

    return eval_fn


__all__ = ["POINTS", "Assembly", "assemble", "build_trainer"]
