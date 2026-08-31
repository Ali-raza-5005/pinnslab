"""``StageSpec.max_work``: stopping a stage on a work counter, not a step count.

**Why the feature exists.** A step count is not a budget for any optimizer whose
per-step cost depends on the data. Five runs of an identical 1250-step L-BFGS
stage, differing only in seed, consumed 6,855 to 11,574 residual evaluations —
a 1.7x spread. Two arms given "the same 1250 steps" are not at equal compute,
and choosing per-seed step counts to hit a target means reading the budget off
the outcome.

**Why these tests exist in this shape.** Every check below is of something that
fails *silently* if broken: a budget that does not bind, a budget that binds in
the wrong unit, a budget that resets on resume, or a stage that stopped early
for a reason nobody recorded. None of them make a run crash; all of them make a
comparison wrong while the table still looks right.

The counter here is a plain call counter over the residual function, which is
exactly the shape a paper repo supplies (paper-01's ``CountingResidual``). The
library is deliberately ignorant of what the unit means.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.registry.config import (
    CheckpointSpec,
    LoggingSpec,
    OptimizerSpec,
    StageSpec,
)
from pinnslab.registry.schema import MetricSchedule
from pinnslab.registry.run import Run
from pinnslab.registry.schema import RunStatus
from pinnslab.training.trainer import Trainer
from tests.conftest import linear_residual, setup_run, toy_config

pytestmark = pytest.mark.unit


class _Killed(RuntimeError):
    """Stands in for the SIGKILL a preemptible session actually gets."""


class Counter:
    """A residual function that counts its own calls.

    One call is one unit of work, whatever drove it: an Adam step, an L-BFGS
    line-search probe, or one candidate of a population method.
    """

    def __init__(self, inner=linear_residual) -> None:
        self.inner = inner
        self.total = 0

    def __call__(self, state):
        self.total += 1
        return self.inner(state)


def build(cfg, results_root, counter, **kwargs):
    ctx, nets = setup_run(cfg)
    run = Run.create(cfg, results_root)
    return Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets,
        residual_fn=counter,
        weighting=lambda residuals, state: sum(
            (v**2).mean() for v in residuals.values()
        ),
        run=run,
        work_fn=lambda: counter.total,
        **kwargs,
    )


def test_a_budget_stops_the_stage_before_its_steps_run_out(results_root):
    """The whole point: ``steps`` becomes a safety bound, ``max_work`` the budget."""
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=10_000,  # deliberately unreachable
                max_work=50,
                optimizers=[OptimizerSpec(lr=1e-2)],
            )
        ],
    )
    trainer = build(cfg, results_root, counter)
    row = trainer.fit()

    assert row.status is RunStatus.COMPLETED
    assert row.steps_completed < 10_000, "the step bound was reached, not the budget"
    assert trainer._timings["stage.adam.hit_work_budget"] == 1.0


def test_adam_lands_exactly_on_its_budget(results_root):
    """One evaluation per step, so the overshoot is zero and the count is exact.

    This is the case that pins the arithmetic. Anything that costs more than one
    evaluation per step can only be bounded (next test); Adam can be checked to
    the unit, and if this drifts the budget is being measured in the wrong place.
    """
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam", steps=10_000, max_work=64, optimizers=[OptimizerSpec(lr=1e-2)]
            )
        ],
    )
    trainer = build(cfg, results_root, counter)
    trainer.fit()

    assert trainer._timings["stage.adam.work"] == 64.0


def test_overshoot_is_bounded_by_one_step_for_a_multi_evaluation_optimizer(
    results_root,
):
    """L-BFGS probes an unknown number of times, so the budget can only bound.

    A step's cost is not knowable before taking it, so the loop checks after.
    That overshoots by at most one step's worth — which is the honest trade, and
    is what makes arms land within a step of each other instead of 1.7x apart.
    """
    counter = Counter()
    budget = 60
    cfg = toy_config(
        stages=[
            StageSpec(
                name="lbfgs",
                steps=10_000,
                max_work=budget,
                optimizers=[OptimizerSpec(name="lbfgs", lr=1.0)],
            )
        ],
    )
    trainer = build(cfg, results_root, counter)
    trainer.fit()

    spent = trainer._timings["stage.lbfgs.work"]
    assert spent >= budget, "the budget did not bind"
    # A line search is capped by torch's max_eval; one step cannot be unbounded.
    assert spent < 2 * budget, f"overshot by more than a plausible single step: {spent}"
    assert trainer._timings["stage.lbfgs.hit_work_budget"] == 1.0


def test_a_non_binding_budget_is_reported_as_such(results_root):
    """A stage that ran out of steps first was never held to its budget.

    That is a different experiment from one that spent it, and the difference is
    invisible in the results table. Recording the flag is what lets an analysis
    refuse the comparison instead of averaging the two together.
    """
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=5,
                max_work=1_000_000,  # unreachable
                optimizers=[OptimizerSpec(lr=1e-2)],
            )
        ],
    )
    trainer = build(cfg, results_root, counter)
    row = trainer.fit()

    assert row.steps_completed == 5
    assert trainer._timings["stage.adam.hit_work_budget"] == 0.0


def test_each_stage_gets_its_own_budget(results_root):
    """Budgets are per stage, mirroring ``steps``, so a total is a sum.

    Cumulative budgets would make a later stage's allowance depend on what
    earlier ones happened to spend, which for a data-dependent optimizer is
    exactly the varying quantity the feature exists to remove.
    """
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="first", steps=10_000, max_work=30, optimizers=[OptimizerSpec(lr=1e-2)]
            ),
            StageSpec(
                name="second", steps=10_000, max_work=20, optimizers=[OptimizerSpec(lr=1e-2)]
            ),
        ],
    )
    trainer = build(cfg, results_root, counter)
    trainer.fit()

    assert trainer._timings["stage.first.work"] == 30.0
    assert trainer._timings["stage.second.work"] == 20.0


def test_work_is_still_reported_without_a_budget(results_root):
    """The spend is the parity currency; it is worth recording unconditionally."""
    counter = Counter()
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=7, optimizers=[OptimizerSpec(lr=1e-2)])],
    )
    trainer = build(cfg, results_root, counter)
    trainer.fit()

    assert trainer._timings["stage.adam.work"] == 7.0
    assert "stage.adam.hit_work_budget" not in trainer._timings


def test_a_budget_without_a_counter_is_refused(results_root):
    """Fail loudly rather than fall back to the step bound.

    Silently ignoring ``max_work`` would produce a run that looks budgeted and
    is not — the single most expensive way this feature could go wrong, because
    the config, the log and the results table would all agree with each other.
    """
    ctx, nets = setup_run(cfg := toy_config(
        stages=[
            StageSpec(
                name="adam", steps=5, max_work=10, optimizers=[OptimizerSpec(lr=1e-2)]
            )
        ],
    ))
    trainer = Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets,
        residual_fn=linear_residual,
        weighting=lambda residuals, state: sum(
            (v**2).mean() for v in residuals.values()
        ),
        run=Run.create(cfg, results_root),
        # work_fn deliberately omitted
    )
    with pytest.raises(ValueError, match="max_work"):
        trainer.fit()


def test_the_budget_survives_a_resume(results_root):
    """The one that matters on a platform that kills sessions.

    A resumed stage must measure its budget from the origin the interrupted run
    used. Reading the counter fresh on resume would restart the allowance, and
    the runs most likely to be interrupted are the slow ones — so the arms that
    would silently overspend are exactly the expensive ones.
    """
    budget = 40
    die_at = 12
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=10_000,
                max_work=budget,
                optimizers=[OptimizerSpec(lr=1e-2)],
            )
        ],
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=5),
    )

    def leg(counter: Counter, *, kill_at: int | None):
        ctx, nets = setup_run(cfg)

        def residual(state):
            if kill_at is not None and state.step == kill_at:
                raise _Killed
            return counter(state)

        return Trainer(
            cfg=cfg,
            ctx=ctx,
            nets=nets,
            residual_fn=residual,
            weighting=lambda residuals, state: sum(
                (v**2).mean() for v in residuals.values()
            ),
            run=Run.create_or_resume(cfg, results_root, "budget-resume"),
            work_fn=lambda: counter.total,
        )

    # --- leg 1: killed mid-stage, exactly as a Kaggle session dies
    counter_a = Counter()
    with pytest.raises(_Killed):
        leg(counter_a, kill_at=die_at).fit()
    assert 0 < counter_a.total < budget, "setup wrong: leg 1 should die mid-budget"

    # --- leg 2: the paper's counter carries its total across the resume, which
    # is what CountingResidual.load_state_dict does. The stage's allowance must
    # continue from there rather than start again.
    counter_b = Counter()
    counter_b.total = counter_a.total
    resumed = leg(counter_b, kill_at=None)
    resumed.fit()

    assert resumed._timings["stage.adam.work"] == float(budget), (
        "the resumed stage restarted its allowance; it recorded "
        f"{resumed._timings['stage.adam.work']} against a budget of {budget}"
    )
    assert resumed._timings["stage.adam.hit_work_budget"] == 1.0


def test_work_at_stage_start_round_trips_through_the_checkpoint():
    """Guards the field itself, independently of the loop that uses it."""
    from pinnslab.training.checkpoint import CheckpointPayload

    payload = CheckpointPayload(
        step=3,
        stage_index=1,
        steps_in_stage=2,
        nets={},
        extra_params={},
        optimizers=[],
        rng={},
        elapsed=1.0,
        config_hash="abc",
        seed=0,
        work_at_stage_start=1234,
    )
    revived = CheckpointPayload.from_dict(payload.to_dict())
    assert revived.work_at_stage_start == 1234


def test_a_stage_two_budget_is_measured_from_stage_two(results_root):
    """The origin resets at a stage boundary, so stage 2 does not inherit stage 1's spend.

    Without the reset, a second stage would see the first stage's evaluations
    already on the clock and stop immediately — a 500-step L-BFGS phase that
    silently becomes a no-op, which is a failure paper-01 measured happening for
    an unrelated reason and could not see from the loss curve.
    """
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="warm", steps=10_000, max_work=25, optimizers=[OptimizerSpec(lr=1e-2)]
            ),
            StageSpec(
                name="main", steps=10_000, max_work=25, optimizers=[OptimizerSpec(lr=1e-2)]
            ),
        ],
    )
    trainer = build(cfg, results_root, counter)
    row = trainer.fit()

    assert trainer._timings["stage.main.work"] == 25.0
    assert row.steps_completed == 50, "the second stage did not actually run"
    assert torch.isfinite(torch.tensor(counter.total))
    assert counter.total >= 50


def test_the_final_trace_point_is_recorded_when_the_budget_ends_the_stage(
    results_root,
):
    """``record_last`` must fire on a work-bounded stage, not only a step-bounded one.

    The regression this pins: ``is_last`` originally tested
    ``step_in_stage == stage.steps``, and a stage bounded by work never reaches
    that -- ``steps`` is deliberately set unreachably high. So the last trace
    point was never forced, and the run reported metrics from whichever
    scheduled point happened to fire last instead of from the end of training.

    A completed run with a stale final rel-L2 and nothing anywhere saying so is
    the worst failure shape in this repo. Found by the first budgeted L-BFGS run
    on a real problem, where the row said 354 evaluations while the stage had
    demonstrably spent 400.
    """
    counter = Counter()
    cfg = toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=10_000,
                max_work=97,  # not a multiple of the trace cadence, on purpose
                optimizers=[OptimizerSpec(lr=1e-2)],
            )
        ],
        logging=LoggingSpec(
            trace=MetricSchedule(every=10, record_first=True, record_last=True)
        ),
    )
    trainer = build(cfg, results_root, counter)
    row = trainer.fit()

    trace = trainer.run.read_trace()
    assert trace[-1].step == row.steps_completed, (
        f"last trace point is at step {trace[-1].step} but the run ended at "
        f"{row.steps_completed}; record_last did not fire"
    )
    # The row's own numbers must come from that final point, not an earlier one.
    assert row.final_metrics["loss"] == pytest.approx(trace[-1].metrics["loss"])
