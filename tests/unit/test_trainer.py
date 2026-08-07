"""The bare loop: stages, directions, the residual contract, failure as data."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pinnslab.registry.config import (
    CheckpointSpec,
    EvalSpec,
    LoggingSpec,
    OptimizerSpec,
    StageSpec,
)
from pinnslab.registry.run import Run, load_runs
from pinnslab.registry.schema import MetricSchedule, RunStatus
from pinnslab.training.trainer import Trainer
from tests.conftest import linear_residual, make_net, setup_run, toy_config

pytestmark = pytest.mark.unit


def build(cfg, results_root, *, residual_fn=linear_residual, nets=None, **kwargs):
    ctx, default_nets = setup_run(cfg)
    run = Run.create(cfg, results_root)
    return Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets if nets is not None else default_nets,
        residual_fn=residual_fn,
        weighting=lambda residuals, state: sum(
            (v**2).mean() for v in residuals.values()
        ),
        run=run,
        **kwargs,
    )


def test_adam_then_lbfgs_lowers_the_loss(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(name="adam", steps=100, optimizers=[OptimizerSpec(lr=1e-2)]),
            StageSpec(
                name="lbfgs",
                steps=10,
                optimizers=[OptimizerSpec(name="lbfgs", lr=1.0)],
            ),
        ],
        logging=LoggingSpec(trace=MetricSchedule(every=10)),
    )
    trainer = build(cfg, results_root)
    row = trainer.fit()

    assert row.status is RunStatus.COMPLETED
    assert row.steps_completed == 110
    trace = trainer.run.read_trace()
    assert trace[-1].metrics["loss"] < trace[0].metrics["loss"]
    assert set(trainer._timings) >= {"stage.adam.seconds", "stage.lbfgs.seconds"}


def test_stage_boundary_is_checkpointed(results_root):
    """L-BFGS history is not in state_dict, so the boundary is the rewind point."""
    cfg = toy_config(
        stages=[
            StageSpec(name="adam", steps=5, optimizers=[OptimizerSpec(lr=1e-2)]),
            StageSpec(
                name="lbfgs", steps=2, optimizers=[OptimizerSpec(name="lbfgs", lr=1.0)]
            ),
        ],
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=None),
    )
    trainer = build(cfg, results_root)
    trainer.fit()
    assert trainer.checkpoints.last_path.exists()


def test_ascent_direction_maximises(results_root):
    """A 'max' optimizer flips its slice's gradients — the min-max primitive."""
    cfg = toy_config(
        stages=[
            StageSpec(
                name="ascend",
                steps=30,
                optimizers=[OptimizerSpec(lr=1e-2, direction="max")],
            )
        ],
        logging=LoggingSpec(trace=MetricSchedule(every=1)),
    )
    trainer = build(cfg, results_root)
    trainer.fit()
    trace = trainer.run.read_trace()
    assert trace[-1].metrics["loss"] > trace[0].metrics["loss"]


def test_min_and_max_optimizers_coexist_on_disjoint_slices(results_root):
    """Conformance item 3: self-adaptive weights as a second, ascending optimizer."""
    cfg = toy_config(
        stages=[
            StageSpec(
                name="minmax",
                steps=20,
                optimizers=[
                    OptimizerSpec(params=r"u\..*", lr=1e-2, direction="min"),
                    OptimizerSpec(params=r"extra\.lam", lr=1e-2, direction="max"),
                ],
            )
        ]
    )
    lam = torch.ones(1, dtype=torch.float64, requires_grad=True)

    def residual(state):
        base = linear_residual(state)
        return {"fit": state.extra_params["lam"] * base["fit"]}

    trainer = build(
        cfg, results_root, residual_fn=residual, extra_params={"lam": lam}
    )
    row = trainer.fit()
    assert row.status is RunStatus.COMPLETED
    assert lam.item() > 1.0  # the ascending slice grew


def test_multiple_networks_are_supported(results_root):
    """Conformance item 1: per-field / per-subdomain networks, no core edits."""
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=10, optimizers=[OptimizerSpec(lr=1e-2)])]
    )
    ctx, _ = setup_run(cfg)
    nets = {"u": make_net(), "v": make_net()}

    def residual(state):
        x = torch.rand(16, 1, generator=state.generator, dtype=state.dtype)
        return {
            "u": (state.nets["u"](x) - x).squeeze(-1),
            "v": (state.nets["v"](x) + x).squeeze(-1),
        }

    trainer = Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets,
        residual_fn=residual,
        weighting=lambda r, s: sum((v**2).mean() for v in r.values()),
        run=Run.create(cfg, results_root),
    )
    assert set(trainer.named_parameters()) >= {"u.0.weight", "v.0.weight"}
    assert trainer.fit().status is RunStatus.COMPLETED


def test_lbfgs_refuses_to_share_a_stage(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="bad",
                steps=1,
                optimizers=[
                    OptimizerSpec(name="lbfgs", params=r"u\.0\..*", lr=1.0),
                    OptimizerSpec(name="adam", params=r"u\.2\..*", lr=1e-3),
                ],
            )
        ]
    )
    trainer = build(cfg, results_root)
    with pytest.raises(ValueError, match="own stage"):
        trainer.fit()


def test_overlapping_selectors_are_rejected(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="overlap",
                steps=1,
                optimizers=[OptimizerSpec(lr=1e-3), OptimizerSpec(lr=1e-4)],
            )
        ]
    )
    trainer = build(cfg, results_root)
    with pytest.raises(ValueError, match="claimed by two optimizers"):
        trainer.fit()


def test_selector_matching_nothing_is_an_error(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="typo", steps=1, optimizers=[OptimizerSpec(params="nosuchnet.*")]
            )
        ]
    )
    trainer = build(cfg, results_root)
    with pytest.raises(ValueError, match="matched no parameters"):
        trainer.fit()


def test_scalar_residual_is_rejected(results_root):
    """CLAUDE.md rule 5, enforced at runtime rather than by convention."""

    def bad_residual(state):
        return {"fit": (linear_residual(state)["fit"] ** 2).mean()}

    trainer = build(toy_config(), results_root, residual_fn=bad_residual)
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        trainer.fit()


def test_column_vector_residual_is_rejected(results_root):
    def bad_residual(state):
        return {"fit": linear_residual(state)["fit"].unsqueeze(-1)}

    trainer = build(toy_config(), results_root, residual_fn=bad_residual)
    with pytest.raises(ValueError, match="per-point"):
        trainer.fit()


def test_divergence_is_a_row_not_a_traceback(results_root):
    def exploding(state):
        base = linear_residual(state)
        if state.step >= 3:
            return {"fit": base["fit"] * float("nan")}
        return base

    trainer = build(toy_config(), results_root, residual_fn=exploding)
    row = trainer.fit()

    assert row.status is RunStatus.DIVERGED
    assert "non-finite" in row.error
    assert row.steps_completed == 4


def test_a_crash_is_recorded_and_re_raised(results_root):
    """A crash is a failure of that config, not a hole in the results."""

    def exploding(state):
        if state.step >= 3:
            raise RuntimeError("CUDA out of memory (simulated)")
        return linear_residual(state)

    trainer = build(toy_config(), results_root, residual_fn=exploding)
    with pytest.raises(RuntimeError, match="out of memory"):
        trainer.fit()

    # Not finished: an OOM at step 3 with a checkpoint behind it is resumable,
    # and finalising it here would throw that compute away.
    assert not trainer.run.is_finished
    row = load_runs(results_root, include_unfinished=True)[0]
    assert row.status is RunStatus.FAILED
    assert "out of memory" in row.error


def test_a_config_error_is_recorded_too(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(name="typo", steps=1, optimizers=[OptimizerSpec(params="nope.*")])
        ]
    )
    trainer = build(cfg, results_root)
    with pytest.raises(ValueError, match="matched no parameters"):
        trainer.fit()
    rows = load_runs(results_root, include_unfinished=True)
    assert rows[0].status is RunStatus.FAILED


def test_a_resumed_run_that_completes_is_completed_not_failed(results_root):
    """The crash record is history; the outcome is whatever the run ended as."""
    cfg = toy_config()
    ctx, nets = setup_run(cfg)
    transient = {"raise": True}

    def flaky(state):
        if transient["raise"] and state.step >= 3:
            raise RuntimeError("transient")
        return linear_residual(state)

    def trainer_for(run):
        return Trainer(
            cfg=cfg,
            ctx=ctx,
            nets=nets,
            residual_fn=flaky,
            weighting=lambda r, s: sum((v**2).mean() for v in r.values()),
            run=run,
        )

    with pytest.raises(RuntimeError, match="transient"):
        trainer_for(Run.create(cfg, results_root, run_id="flaky")).fit()

    transient["raise"] = False
    row = trainer_for(Run.resume(results_root, "flaky", cfg)).fit()

    assert row.status is RunStatus.COMPLETED
    assert row.steps_completed == cfg.total_steps
    assert [r.status for r in load_runs(results_root)] == [RunStatus.COMPLETED]


def test_time_to_target_is_recorded_in_steps_and_seconds(results_root):
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=300, optimizers=[OptimizerSpec(lr=1e-2)])],
        eval=EvalSpec(
            best_metric="loss", target_metric="loss", target_value=1.0, best_mode="min"
        ),
        logging=LoggingSpec(trace=MetricSchedule(every=10)),
    )
    trainer = build(cfg, results_root)
    row = trainer.fit()

    assert row.timings["time_to_target_steps"] > 0
    assert row.timings["time_to_target_seconds"] >= 0
    assert row.timings["train_seconds"] > 0


def test_resample_hook_fires_on_schedule(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=20,
                resample_every=5,
                optimizers=[OptimizerSpec(lr=1e-2)],
            )
        ]
    )
    calls: list[int] = []
    trainer = build(cfg, results_root, on_resample=lambda s: calls.append(s.step))
    trainer.fit()
    assert calls == [0, 5, 10, 15]


def test_record_first_traces_the_untrained_baseline(results_root):
    """The left-hand end of a log-log convergence plot has to come from somewhere."""
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=20, optimizers=[OptimizerSpec(lr=1e-2)])],
        logging=LoggingSpec(trace=MetricSchedule(every=5)),
    )
    trainer = build(cfg, results_root)
    trainer.fit()

    trace = trainer.run.read_trace()
    assert [p.step for p in trace] == [0, 5, 10, 15, 20]
    # Step 0 is the loss before any optimizer ran, so it must be the worst one.
    assert trace[0].metrics["loss"] > trace[-1].metrics["loss"]


def test_record_first_can_be_turned_off(results_root):
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=20, optimizers=[OptimizerSpec(lr=1e-2)])],
        logging=LoggingSpec(trace=MetricSchedule(every=5, record_first=False)),
    )
    trainer = build(cfg, results_root)
    trainer.fit()
    assert [p.step for p in trainer.run.read_trace()] == [5, 10, 15, 20]


def test_a_resumed_run_does_not_re_record_the_baseline(results_root):
    """A resumed trace must look exactly like an uninterrupted one."""
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=20, optimizers=[OptimizerSpec(lr=1e-2)])],
        logging=LoggingSpec(trace=MetricSchedule(every=5)),
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=5),
    )
    ctx, nets = setup_run(cfg)
    crash_at = {"step": 12}

    def flaky(state):
        if state.step >= crash_at["step"]:
            raise RuntimeError("session died")
        return linear_residual(state)

    def trainer_for(run):
        return Trainer(
            cfg=cfg,
            ctx=ctx,
            nets=nets,
            residual_fn=flaky,
            weighting=lambda r, s: sum((v**2).mean() for v in r.values()),
            run=run,
        )

    with pytest.raises(RuntimeError, match="session died"):
        trainer_for(Run.create(cfg, results_root, run_id="r")).fit()

    crash_at["step"] = 10**9
    trainer = trainer_for(Run.resume(results_root, "r", cfg))
    trainer.fit()

    assert [p.step for p in trainer.run.read_trace()] == [0, 5, 10, 15, 20]


def test_a_crash_before_the_first_step_does_not_duplicate_the_baseline(results_root):
    """A session can die before completing step 0, after only the baseline
    trace point and the stage-boundary checkpoint (at step 0) were written.

    ``last.pt`` then holds ``step=0``, the same value a genuinely fresh run's
    missing checkpoint reports — so freshness must not be inferred from
    ``step == 0`` alone, or the resumed session re-records the baseline.

    The residual_fn is called once for the baseline (before any checkpoint
    exists) and must succeed there; it must only raise on the *second* call —
    the first real training step, which happens after the stage-boundary
    ``last.pt`` save at step 0.
    """
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=20, optimizers=[OptimizerSpec(lr=1e-2)])],
        logging=LoggingSpec(trace=MetricSchedule(every=5)),
    )
    ctx, nets = setup_run(cfg)
    crash = {"enabled": True, "calls": 0}

    def flaky(state):
        crash["calls"] += 1
        if crash["enabled"] and crash["calls"] == 2:
            raise RuntimeError("session died")
        return linear_residual(state)

    def trainer_for(run):
        return Trainer(
            cfg=cfg,
            ctx=ctx,
            nets=nets,
            residual_fn=flaky,
            weighting=lambda r, s: sum((v**2).mean() for v in r.values()),
            run=run,
        )

    with pytest.raises(RuntimeError, match="session died"):
        trainer_for(Run.create(cfg, results_root, run_id="r")).fit()

    crash["enabled"] = False
    trainer = trainer_for(Run.resume(results_root, "r", cfg))
    trainer.fit()

    steps = [p.step for p in trainer.run.read_trace()]
    assert steps == [0, 5, 10, 15, 20]
    assert steps.count(0) == 1


def test_eval_fn_metrics_reach_the_trace(results_root):
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=10, optimizers=[OptimizerSpec(lr=1e-2)])],
        logging=LoggingSpec(trace=MetricSchedule(every=5)),
    )
    trainer = build(cfg, results_root, eval_fn=lambda state: {"rel_l2": 0.25})
    row = trainer.fit()

    assert row.final_metrics["rel_l2"] == 0.25
    assert all("residual/fit" in p.metrics for p in trainer.run.read_trace())


def test_extra_params_are_trainable_and_checkpointed(results_root):
    """Conformance item 2: inverse problems, PDE coefficients as parameters."""
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=100, optimizers=[OptimizerSpec(lr=1e-1)])]
    )
    nu = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

    def residual(state):
        x = torch.rand(16, 1, generator=state.generator, dtype=state.dtype)
        return {"inverse": (state.extra_params["nu"] - 3.0).expand(x.shape[0])}

    trainer = build(cfg, results_root, residual_fn=residual, extra_params={"nu": nu})
    trainer.fit()

    assert "extra.nu" in trainer.named_parameters()
    assert nu.item() == pytest.approx(3.0, abs=0.5)


def test_hard_constraint_output_transform_needs_no_core_support(results_root):
    """Conformance item 5: the transform is just part of the caller's module."""

    class Constrained(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = make_net()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.body(x)  # exactly zero at x = 0

    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=5, optimizers=[OptimizerSpec(lr=1e-2)])]
    )
    ctx, _ = setup_run(cfg)
    net = Constrained().to(dtype=ctx.dtype)
    trainer = build(cfg, results_root, nets={"u": net})
    trainer.fit()

    assert net(torch.zeros(1, 1, dtype=ctx.dtype)).abs().item() == 0.0
