"""The training loop (DESIGN.md §1, §4).

We own this loop because the research novelty lives inside it. It is deliberately
thin: it knows about *stages*, *optimizers with a direction*, *checkpointing*,
*timing* and *provenance*, and nothing at all about PDEs, geometry, architectures
or sampling. Those enter through three callables the caller supplies.

The load-bearing contract (DESIGN.md §4):

* ``residual_fn(state) -> dict[str, Tensor]``, each of shape ``(N,)`` — per-point,
  never pre-reduced.
* ``weighting(residuals, state) -> Tensor`` — a scalar. All reduction lives here.
* optimizers are a **list** with a parameter selector and a direction, so
  min-max and self-adaptive schemes are expressible without touching this file.
* ``residual_fn`` must be a deterministic function of ``state`` within a single
  step: read collocation points from ``state.points`` (populated by
  ``on_resample``) rather than drawing fresh samples inline. L-BFGS's closure
  can be invoked more than once per ``.step()`` (line search), and a stochastic
  closure breaks the line search's assumptions.

Conformance (DESIGN.md §4): multiple networks, inverse-problem parameters,
ascent optimizers, per-point weights, hard constraints via the caller's output
transform, staged training, and coupled systems are all expressible through those
three callables plus ``RunConfig.stages``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

import pinnslab.training.optimizers as _optimizers  # noqa: F401  (registers builtins)
from pinnslab.registry.config import OptimizerSpec, RunConfig, StageSpec
from pinnslab.registry.run import Run
from pinnslab.registry.schema import ResultRow, RunStatus
from pinnslab.training.checkpoint import CheckpointManager, CheckpointPayload
from pinnslab.training.optimizers import build_optimizer
from pinnslab.utils.device import RuntimeContext
from pinnslab.utils.logging import get_logger
from pinnslab.utils.seeding import (
    capture_rng_state,
    derive_seed,
    make_generator,
    restore_rng_state,
)

log = get_logger(__name__)

EXTRA_PREFIX = "extra"


@dataclass
class TrainState:
    """What a residual function, weighting or hook is allowed to see."""

    cfg: RunConfig
    nets: dict[str, nn.Module]
    extra_params: dict[str, torch.Tensor]
    generator: torch.Generator
    device: torch.device
    dtype: torch.dtype
    step: int = 0
    stage_index: int = 0
    stage_name: str = ""
    step_in_stage: int = 0
    #: The collocation points the loss is currently evaluated on, by point-group
    #: name. Written by ``on_resample``, read by ``residual_fn`` — and
    #: **checkpointed**, which is why it is a field of its own rather than a
    #: ``scratch`` key.
    #:
    #: It has to be. With ``resample_every`` set, the cloud in force at step *k*
    #: is not recoverable from the RNG stream alone once a sampler is adaptive:
    #: it was drawn against the network as it stood at the last resample, and
    #: that network no longer exists after a resume. A run that resumed onto a
    #: freshly drawn cloud would be a different experiment wearing the same
    #: ``run_id``, and nothing in its metrics would say so.
    points: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Free-form scratch space for hooks. **Not** checkpointed: anything that
    #: must survive a resume belongs in ``points``, in ``extra_params``, or in
    #: the resample hook's own ``state_dict`` (see ``geometry/samplers.py``).
    scratch: dict[str, Any] = field(default_factory=dict)


ResidualFn = Callable[[TrainState], dict[str, torch.Tensor]]
WeightingFn = Callable[[dict[str, torch.Tensor], TrainState], torch.Tensor]
EvalFn = Callable[[TrainState], dict[str, float]]
HookFn = Callable[[TrainState], None]


class Trainer:
    """Runs one :class:`RunConfig` to completion, resumably."""

    def __init__(
        self,
        *,
        cfg: RunConfig,
        ctx: RuntimeContext,
        nets: dict[str, nn.Module],
        residual_fn: ResidualFn,
        weighting: WeightingFn,
        run: Run,
        extra_params: dict[str, torch.Tensor] | None = None,
        eval_fn: EvalFn | None = None,
        on_resample: HookFn | None = None,
        checkpoints: CheckpointManager | None = None,
        allow_config_change: bool = False,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.nets = nets
        self.residual_fn = residual_fn
        self.weighting = weighting
        self.run = run
        self.extra_params = extra_params or {}
        self.eval_fn = eval_fn
        self.on_resample = on_resample
        self.allow_config_change = allow_config_change
        self.checkpoints = checkpoints or CheckpointManager(
            run.checkpoint_dir,
            cfg.checkpoint,
            config_hash=run.config_hash,
            seed=cfg.seed,
            best_mode=cfg.eval.best_mode,
        )

        # A dedicated stream so that sampling reproducibility does not depend on
        # how many global draws happened to occur beforehand.
        self.generator = make_generator(
            derive_seed(cfg.seed, "trainer", run.config_hash), device="cpu"
        )
        self.state = TrainState(
            cfg=cfg,
            nets=nets,
            extra_params=self.extra_params,
            generator=self.generator,
            device=ctx.device,
            dtype=ctx.dtype,
        )

        self._elapsed = 0.0
        self._best_value: float | None = None
        self._best_metrics: dict[str, float] = {}
        self._best_step: int | None = None
        self._timings: dict[str, float] = {}
        self._last_residuals: dict[str, torch.Tensor] = {}
        self._last_metrics: dict[str, float] = {}
        self._current_groups: list[_OptimizerGroup] = []
        self._pending_optimizer_state: list[dict[str, Any]] = []

    # -- parameter plumbing ---------------------------------------------------

    def named_parameters(self) -> dict[str, torch.Tensor]:
        """``{"<net>.<param>": tensor} | {"extra.<key>": tensor}``, trainables only."""
        params: dict[str, torch.Tensor] = {}
        for net_name, net in self.nets.items():
            for pname, param in net.named_parameters():
                if param.requires_grad:
                    params[f"{net_name}.{pname}"] = param
        for key, tensor in self.extra_params.items():
            if tensor.requires_grad:
                params[f"{EXTRA_PREFIX}.{key}"] = tensor
        return params

    def _select(self, spec: OptimizerSpec) -> tuple[list[str], list[torch.Tensor]]:
        pattern = re.compile(spec.params)
        available = self.named_parameters()
        names = [n for n in available if pattern.fullmatch(n)]
        if not names:
            raise ValueError(
                f"optimizer selector {spec.params!r} matched no parameters; "
                f"available: {sorted(available)}. Selectors are full-match "
                "regexes over '<net>.<param>' and 'extra.<key>'."
            )
        return names, [available[n] for n in names]

    def _build_stage(self, stage: StageSpec) -> list[_OptimizerGroup]:
        groups: list[_OptimizerGroup] = []
        claimed: dict[str, str] = {}
        for spec in stage.optimizers:
            names, params = self._select(spec)
            for name in names:
                if name in claimed:
                    raise ValueError(
                        f"parameter {name!r} is claimed by two optimizers in "
                        f"stage {stage.name!r} ({claimed[name]!r} and "
                        f"{spec.params!r}); with per-optimizer directions that "
                        "would be ambiguous, so selectors must be disjoint."
                    )
                claimed[name] = spec.params
            groups.append(
                _OptimizerGroup(
                    spec=spec, params=params, optimizer=build_optimizer(params, spec)
                )
            )

        lbfgs = [g for g in groups if isinstance(g.optimizer, torch.optim.LBFGS)]
        if lbfgs and len(groups) > 1:
            raise ValueError(
                f"stage {stage.name!r} pairs L-BFGS with another optimizer. "
                "L-BFGS is closure-based and full-batch; it re-evaluates the "
                "loss internally, so a concurrent ascent step would see a "
                "different iterate than it stepped from. Give L-BFGS its own "
                "stage."
            )
        if lbfgs and lbfgs[0].spec.direction != "min":
            raise ValueError("L-BFGS supports direction='min' only")
        return groups

    # -- the loop -------------------------------------------------------------

    def fit(self) -> ResultRow:
        """Train to completion (or divergence), writing one result row.

        Any exception is recorded via :meth:`Run.log_failure` before it
        propagates. An OOM, a bad selector or a bug in a residual function is a
        failure *of that configuration*, and a sweep that reports only the runs
        which survived long enough to write a row reports a fiction
        (DESIGN.md §11). Recording it does not finish the run — a crash with a
        checkpoint behind it is still resumable — and the exception is re-raised
        so the caller decides whether to abort the sweep or move on.
        """
        try:
            return self._fit()
        except Exception as exc:
            self._failed(exc)
            raise

    def _fit(self) -> ResultRow:
        start_stage, start_step_in_stage, global_step, resumed = self._restore()
        if not resumed and self.cfg.logging.trace.record_first:
            self._record_baseline(start_stage)

        for stage_index in range(start_stage, len(self.cfg.stages)):
            stage = self.cfg.stages[stage_index]
            groups = self._build_stage(stage)
            self._current_groups = groups
            resuming = stage_index == start_stage and start_step_in_stage > 0
            step_in_stage = start_step_in_stage if resuming else 0

            if resuming:
                # L-BFGS included: its curvature history round-trips through
                # state_dict, so a mid-stage resume is bit-exact like any other
                # (pinned by test_lbfgs_resume_is_bit_exact).
                self._restore_optimizers(groups)
            else:
                self._save(global_step, stage_index, 0)

            self.state.stage_index = stage_index
            self.state.stage_name = stage.name
            log.info(
                "stage %d/%d %r: %d steps from step_in_stage=%d",
                stage_index + 1,
                len(self.cfg.stages),
                stage.name,
                stage.steps,
                step_in_stage,
            )

            stage_seconds = 0.0
            while step_in_stage < stage.steps:
                # Publish the position before any hook runs: a resampler that
                # reads a stale step would place points for the previous
                # iteration.
                self.state.step = global_step
                self.state.step_in_stage = step_in_stage

                if (
                    stage.resample_every
                    and step_in_stage % stage.resample_every == 0
                    and self.on_resample is not None
                ):
                    self.on_resample(self.state)

                step_t0 = time.perf_counter()
                loss = self._step(groups)
                dt = time.perf_counter() - step_t0
                self._elapsed += dt
                stage_seconds += dt
                step_in_stage += 1
                global_step += 1

                if not torch.isfinite(loss) and self.cfg.eval.stop_on_nonfinite:
                    self._timings[_stage_key(stage)] = (
                        self._timings.get(_stage_key(stage), 0.0) + stage_seconds
                    )
                    return self._diverged(global_step, stage_index, step_in_stage, loss)

                # A target the step already computed is checked *every* step, not
                # on the trace schedule: "steps to target" is a reviewer-facing
                # compute-parity number, and quantising it to the trace cadence
                # would make it depend on a field that is deliberately excluded
                # from the config hash. An eval_fn-derived target stays on the
                # schedule — that is what the schedule is for — and records its
                # resolution instead.
                if self._target_is_free_every_step():
                    self._track_target(global_step, self._step_metrics(float(loss)))

                is_last = (
                    step_in_stage == stage.steps
                    and stage_index == len(self.cfg.stages) - 1
                )
                if self.cfg.logging.trace.should_record(global_step, is_last=is_last):
                    self._record(global_step, stage_index, step_in_stage, float(loss))

                if self.checkpoints.due(global_step):
                    self._save(global_step, stage_index, step_in_stage)

            self._timings[_stage_key(stage)] = (
                self._timings.get(_stage_key(stage), 0.0) + stage_seconds
            )
            start_step_in_stage = 0

        self._save(global_step, len(self.cfg.stages) - 1, self.cfg.stages[-1].steps)
        self._timings["train_seconds"] = self._elapsed
        return self.run.finish(
            status=RunStatus.COMPLETED,
            steps_completed=global_step,
            final_metrics=self._last_metrics,
            best_metrics=self._best_metrics,
            timings=self._timings,
        )

    def _step(self, groups: list[_OptimizerGroup]) -> torch.Tensor:
        if _uses_lbfgs(groups):
            return self._step_lbfgs(groups[0])
        return self._step_first_order(groups)

    def _zero_grads(self) -> None:
        """Clear every trainable gradient, not just the ones an optimizer owns.

        A parameter outside all selectors would otherwise accumulate gradients
        forever — harmless for the result, but it silently pins memory and makes
        any later selector change behave strangely.
        """
        for param in self.named_parameters().values():
            param.grad = None

    def _step_first_order(self, groups: list[_OptimizerGroup]) -> torch.Tensor:
        self._zero_grads()
        loss = self._forward()
        loss.backward()
        for group in groups:
            if group.spec.direction == "max":
                # Ascend by flipping this slice's gradients: the min-max /
                # self-adaptive pattern of DESIGN.md §4 with no special casing.
                for param in group.params:
                    if param.grad is not None:
                        param.grad.neg_()
            if group.spec.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(group.params, group.spec.max_grad_norm)
            group.optimizer.step()
        return loss.detach()

    def _step_lbfgs(self, group: _OptimizerGroup) -> torch.Tensor:
        def closure() -> torch.Tensor:
            self._zero_grads()
            loss = self._forward()
            loss.backward()
            if group.spec.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(group.params, group.spec.max_grad_norm)
            return loss

        return torch.as_tensor(group.optimizer.step(closure)).detach()

    def _forward(self) -> torch.Tensor:
        residuals = self.residual_fn(self.state)
        _check_residuals(residuals)
        self._last_residuals = residuals
        loss = self.weighting(residuals, self.state)
        if loss.ndim != 0:
            raise ValueError(
                f"weighting must return a scalar, got shape {tuple(loss.shape)}"
            )
        return loss

    # -- bookkeeping ----------------------------------------------------------

    def _record_baseline(self, stage_index: int) -> None:
        """Trace step 0: the loss before any optimizer has touched a parameter.

        Only fired on a fresh run — a resumed one already recorded step 0 in its
        first session, and re-recording it would put a point in the trace that
        the uninterrupted run does not have.

        This is the one point in the trace where every metric, the loss included,
        is evaluated at the *same* parameters. In the loop the loss is the value
        the step was computed from (pre-update) while ``eval_fn`` and the
        checkpoint see the post-update parameters — the ordinary convention, but
        it means step 0 is the only clean baseline.

        No ``torch.no_grad()``: a PDE residual differentiates through the network
        to build itself, so the graph is required even when nothing is stepping.
        """
        self.state.stage_index = stage_index
        self.state.stage_name = self.cfg.stages[stage_index].name
        self._record(0, stage_index, 0, float(self._forward().detach()))

    def _record(
        self, step: int, stage_index: int, step_in_stage: int, loss: float
    ) -> None:
        metrics: dict[str, float] = {"loss": loss}
        for name, value in self._last_residuals.items():
            metrics[f"residual/{name}"] = float((value.detach() ** 2).mean())
        if self.eval_fn is not None:
            metrics.update({k: float(v) for k, v in self.eval_fn(self.state).items()})
        self._last_metrics = metrics
        self.run.log_metrics(step, metrics, stage=stage_index, wall_time=self._elapsed)

        self._track_target(step, metrics)
        best_metric = self.cfg.eval.best_metric
        if best_metric is None or best_metric not in metrics:
            return
        value = metrics[best_metric]
        if self.checkpoints.is_improvement(value, self._best_value):
            self._best_value = value
            self._best_metrics = dict(metrics)
            self._best_step = step
            self.checkpoints.save_best(self._payload(step, stage_index, step_in_stage))

    def _target_is_free_every_step(self) -> bool:
        """Can the target be checked without paying for ``eval_fn``?

        ``loss`` is computed by the step itself, and every ``residual/<name>`` is
        a mean of squares over a tensor the step already produced. Those cost
        nothing worth counting. A metric from ``eval_fn`` — ``rel_l2`` is the
        one that matters — is a forward pass over the whole evaluation grid, and
        the trace schedule exists precisely to avoid paying it every step.
        """
        metric = self.cfg.eval.target_metric
        return metric == "loss" or (
            metric is not None and metric.startswith("residual/")
        )

    def _target_resolution(self) -> float:
        """The granularity of the recorded time-to-target, in steps.

        Recorded alongside the number itself, because a reader cannot otherwise
        tell "reached at step 300" from "first *observed* at step 300, having
        possibly happened at 201". It also guards a comparability trap:
        ``logging`` is excluded from the config hash (changing trace density
        does not make it a different experiment), so two runs of one condition
        may legitimately trace at different cadences — and their time-to-target
        would then differ for a reason that has nothing to do with training.
        """
        if self._target_is_free_every_step():
            return 1.0
        schedule = self.cfg.logging.trace
        return float(schedule.every) if schedule.every else float("nan")

    def _track_target(self, step: int, metrics: dict[str, float]) -> None:
        spec = self.cfg.eval
        if spec.target_metric is None or "time_to_target_seconds" in self._timings:
            return
        value = metrics.get(spec.target_metric)
        if value is None:
            return
        reached = (
            value <= spec.target_value
            if spec.best_mode == "min"
            else value >= spec.target_value
        )
        if reached:
            # Time-to-target is reported alongside final accuracy (DESIGN.md §8),
            # in both units, because "steps" and "seconds" answer different
            # reviewer questions — and with the resolution of the observation,
            # because an upper bound reported as an exact value is not a
            # compute-parity number.
            self._timings["time_to_target_seconds"] = self._elapsed
            self._timings["time_to_target_steps"] = float(step)
            self._timings["time_to_target_resolution_steps"] = self._target_resolution()

    def _step_metrics(self, loss: float) -> dict[str, float]:
        """The metrics the step already paid for. See ``_target_is_free_every_step``."""
        metrics = {"loss": loss}
        if self.cfg.eval.target_metric != "loss":
            for name, value in self._last_residuals.items():
                metrics[f"residual/{name}"] = float((value.detach() ** 2).mean())
        return metrics

    def _payload(
        self, step: int, stage_index: int, steps_in_stage: int
    ) -> CheckpointPayload:
        rng = capture_rng_state()
        rng["trainer_generator"] = self.generator.get_state()
        return CheckpointPayload(
            points={k: v.detach().clone() for k, v in self.state.points.items()},
            sampler_state=_hook_state(self.on_resample),
            step=step,
            stage_index=stage_index,
            steps_in_stage=steps_in_stage,
            nets={name: net.state_dict() for name, net in self.nets.items()},
            extra_params={k: v.detach().clone() for k, v in self.extra_params.items()},
            optimizers=[g.optimizer.state_dict() for g in self._current_groups],
            rng=rng,
            elapsed=self._elapsed,
            config_hash=self.run.config_hash,
            seed=self.cfg.seed,
            best_value=self._best_value,
            best_metrics=self._best_metrics,
            best_step=self._best_step,
            timings=dict(self._timings),
        )

    def _save(self, step: int, stage_index: int, steps_in_stage: int) -> None:
        self.checkpoints.save_last(self._payload(step, stage_index, steps_in_stage))

    def _restore(self) -> tuple[int, int, int, bool]:
        """Load the last checkpoint, if any.

        Returns ``(stage_index, steps_in_stage, step, loaded)``. ``loaded`` is
        the only reliable signal for "was this a fresh start" — a checkpoint
        saved at step 0 (the stage-boundary save that happens before the first
        training step) makes ``step == 0`` true on a genuine resume too, so
        the caller must not use ``step == 0`` as a proxy for freshness.
        """
        payload = self.checkpoints.load_last(
            allow_config_change=self.allow_config_change
        )
        if payload is None:
            self._current_groups = []
            return 0, 0, 0, False

        for name, net in self.nets.items():
            net.load_state_dict(payload.nets[name])
        for key, tensor in self.extra_params.items():
            with torch.no_grad():
                tensor.copy_(payload.extra_params[key])
        restore_rng_state(payload.rng)
        self.generator.set_state(payload.rng["trainer_generator"])

        # The cloud and the sampler's own state, both drawn *before* the step
        # this checkpoint records. Restoring them is what makes a resumed
        # resampling run continue the same experiment rather than start a new
        # one on a fresh cloud: `build_trainer` has already drawn an initial
        # cloud by now, and this overwrites it, exactly as it overwrites the
        # RNG stream that produced it.
        if payload.points:
            self.state.points = {
                name: tensor.to(device=self.ctx.device, dtype=self.ctx.dtype)
                for name, tensor in payload.points.items()
            }
        if payload.sampler_state and hasattr(self.on_resample, "load_state_dict"):
            self.on_resample.load_state_dict(payload.sampler_state)

        self._elapsed = payload.elapsed
        self._best_value = payload.best_value
        self._best_metrics = dict(payload.best_metrics)
        self._best_step = payload.best_step
        self._timings = dict(payload.timings)
        self._pending_optimizer_state = payload.optimizers
        self._current_groups = []
        log.info(
            "resumed from %s at step %d (stage %d, %d steps into it)",
            self.checkpoints.last_path,
            payload.step,
            payload.stage_index,
            payload.steps_in_stage,
        )
        return payload.stage_index, payload.steps_in_stage, payload.step, True

    def _restore_optimizers(self, groups: list[_OptimizerGroup]) -> None:
        states = self._pending_optimizer_state
        if len(states) != len(groups):
            raise ValueError(
                f"checkpoint holds {len(states)} optimizer states but the stage "
                f"defines {len(groups)}; the config changed under the checkpoint"
            )
        for group, state in zip(groups, states, strict=True):
            group.optimizer.load_state_dict(state)

    def _failed(self, exc: Exception) -> None:
        """Record a crash. Never raises — it must not mask ``exc``."""
        if self.run.is_finished:
            # finish() already wrote the row; whatever blew up came after it.
            return
        log.exception("run %s failed at step %d", self.run.run_id, self.state.step)
        try:
            self.run.log_failure(exc, step=self.state.step)
        except Exception:  # noqa: BLE001 - bookkeeping must not mask the failure
            log.exception("could not record the crash for run %s", self.run.run_id)

    def _diverged(
        self, step: int, stage_index: int, step_in_stage: int, loss: torch.Tensor
    ) -> ResultRow:
        # Divergence is data, not an error (DESIGN.md §11): the row is written so
        # that failure rate can be reported honestly.
        log.warning("run %s diverged at step %d (loss=%s)", self.run.run_id, step, loss)
        self._timings["train_seconds"] = self._elapsed
        self.run.log_metrics(
            step, {"loss": float(loss)}, stage=stage_index, wall_time=self._elapsed
        )
        return self.run.finish(
            status=RunStatus.DIVERGED,
            steps_completed=step,
            final_metrics={"loss": float(loss)},
            best_metrics=self._best_metrics,
            timings=self._timings,
            error=f"non-finite loss at step {step} (stage {stage_index}, "
            f"step_in_stage {step_in_stage})",
        )


@dataclass
class _OptimizerGroup:
    spec: OptimizerSpec
    params: list[torch.Tensor]
    optimizer: torch.optim.Optimizer


def _hook_state(hook: HookFn | None) -> dict[str, Any]:
    """What a resample hook wants carried across a session, if anything.

    Duck-typed on purpose: ``on_resample`` is a plain callable in the escape
    hatch of DESIGN.md §4, and requiring every caller to subclass something to
    pass a lambda would close that hatch. A hook that holds state opts in by
    growing the two methods.
    """
    getter = getattr(hook, "state_dict", None)
    return dict(getter()) if callable(getter) else {}


def _uses_lbfgs(groups: list[_OptimizerGroup]) -> bool:
    return any(isinstance(g.optimizer, torch.optim.LBFGS) for g in groups)


def _stage_key(stage: StageSpec) -> str:
    return f"stage.{stage.name}.seconds"


def _check_residuals(residuals: dict[str, torch.Tensor]) -> None:
    """Enforce DESIGN.md §4 decision 1 at runtime, not by convention."""
    if not residuals:
        raise ValueError("residual_fn returned an empty dict")
    for name, value in residuals.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"residual {name!r} is {type(value).__name__}, not Tensor")
        if value.ndim != 1:
            raise ValueError(
                f"residual {name!r} has shape {tuple(value.shape)}; residuals must "
                "be per-point tensors of shape (N,), never scalars or column "
                "vectors — reduction belongs to the weighting object "
                "(CLAUDE.md rule 5)."
            )


__all__ = ["EvalFn", "HookFn", "ResidualFn", "TrainState", "Trainer", "WeightingFn"]
