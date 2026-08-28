"""Golden regression: 1-D Burgers, end to end (DESIGN.md §9 step 2).

Run these before any commit touching training, physics or losses (CLAUDE.md
rule 10). They go from a YAML file on disk to a trained network and a rel-L2
against an exact solution, so a break anywhere on that path — diffops, the
residual terms, the reference, the optimizer wiring, the eval grid, the config
loader — moves the number far past the threshold.

What the threshold can and cannot catch
---------------------------------------
The config runs at ``nu = 0.1/pi``, ten times the literature-standard viscosity,
and this is the honest reason: at ``0.01/pi`` the solution steepens into a
near-shock, and a run that fits the two-minute CPU budget lands anywhere in
0.064-0.204 rel-L2 across seeds (measured 2026-08-08). A threshold loose enough
to pass that reliably would be ~0.35, which is useless — the closed-domain bug
fixed that same day produced 0.127-0.273 and would have sailed straight through
it. Smoothed, the same budget converges to ~6e-4 with under 1.5x spread across
seeds, and the assertion has teeth.

The cost is stated rather than hidden: **this test is blind to the failure that
actually happened.** Enforcing the PDE on the interior alone scores 0.00081 on
this config, indistinguishable from the correct 0.00059-0.00085. That failure
needed the sharp front. It is guarded instead by the frozen config above (whose
``points`` list is hashed into the run's identity) and by
``test_the_point_groups_enter_the_config_hash`` in the unit suite. If a future
change makes the standard viscosity affordable and reproducible here, this test
should move to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pinnslab.registry.config import load_config
from pinnslab.registry.run import Run
from pinnslab.registry.schema import RunStatus
from pinnslab.training.build import build_trainer
from pinnslab.utils.device import configure_runtime

pytestmark = pytest.mark.golden

CONFIG = Path(__file__).parent / "configs" / "burgers_smooth.yaml"

#: Measured 0.00059 / 0.00061 / 0.00085 over seeds 0-2 on CPU/float64
#: (torch 2.12, 2026-08-08). Set at ~2.4x the worst of those: enough margin for
#: a torch or platform difference, tight enough that any real regression in the
#: physics or the loop blows through it by orders of magnitude.
REL_L2_TARGET = 2e-3

#: The frozen identity of the config above. Not decoration: it is what a result
#: row joins on, and a silent change to the YAML would otherwise re-target the
#: threshold at a different experiment.
CONFIG_HASH = "0ebf401fda6fa1d0"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Train the frozen config once; the assertions below all read this run."""
    cfg = load_config(CONFIG)
    ctx = configure_runtime(cfg)
    run = Run.create(cfg, tmp_path_factory.mktemp("results"))
    row = build_trainer(cfg, ctx, run).fit()
    return cfg, run, row


def test_the_frozen_config_still_has_its_frozen_hash():
    """Guards the other tests: they assert a number about *this* experiment."""
    assert load_config(CONFIG).identity_hash() == CONFIG_HASH


def test_burgers_reaches_its_target_accuracy(result):
    _, _, row = result
    assert row.status == RunStatus.COMPLETED
    assert row.final_metrics["rel_l2"] < REL_L2_TARGET, (
        f"rel-L2 {row.final_metrics['rel_l2']:.2e} exceeds the frozen target "
        f"{REL_L2_TARGET:.0e}. Something on the config -> YAML -> network -> "
        "residual -> metric path regressed; the unit suite localises it."
    )


def test_the_solution_is_accurate_everywhere_not_only_on_average(result):
    """rel-L2 can look fine while the solution is wrong across a front, which is
    exactly the failure mode this benchmark exists to expose."""
    _, _, row = result
    assert row.final_metrics["max_error"] < 2e-2


def test_the_run_records_full_provenance(result):
    """CLAUDE.md rule 7, checked on a real run rather than a constructed row."""
    _, _, row = result
    for field in (
        "pinnslab_version",
        "git_sha",
        "seed",
        "gpu_name",
        "dtype",
        "device_profile",
        "config_hash",
    ):
        assert getattr(row, field) != "", f"{field} is empty on a completed run"
    assert row.dtype == "float64"
    assert row.config_hash == CONFIG_HASH


def test_both_stages_ran_and_are_timed(result):
    """Adam -> L-BFGS is the staged-training conformance case of DESIGN.md §4,
    and per-stage wall-clock is a first-class result (DESIGN.md §11)."""
    cfg, _, row = result
    assert row.steps_completed == cfg.total_steps
    for stage in cfg.stages:
        assert row.timings[f"stage.{stage.name}.seconds"] > 0.0


def test_lbfgs_improves_on_adam(result):
    """If the second stage were silently a no-op the test above would still
    pass. L-BFGS carried this config from ~2e-2 to ~6e-4 when it was chosen."""
    _, run, _ = result
    trace = run.read_trace()
    end_of_adam = max(
        (p for p in trace if p.stage == 0), key=lambda p: p.step
    )
    final = max(trace, key=lambda p: p.step)

    assert final.stage == 1, "the trace never reached the L-BFGS stage"
    assert final.metrics["rel_l2"] < end_of_adam.metrics["rel_l2"]


def test_the_run_is_reproducible(tmp_path):
    """A DESIGN.md §5 non-negotiable, asserted on the whole assembled pipeline
    rather than on the trainer alone: same config, same seed, same numbers.

    Deliberately short — this is about determinism, not accuracy, and running
    the full golden config twice would blow the two-minute budget.
    """
    cfg = load_config(CONFIG)
    quick = cfg.model_copy(
        update={"stages": [cfg.stages[0].model_copy(update={"steps": 40})]}
    )

    finals = []
    for run_id in ("a", "b"):
        ctx = configure_runtime(quick)
        run = Run.create(quick, tmp_path, run_id=run_id)
        finals.append(build_trainer(quick, ctx, run).fit().final_metrics)

    first, second = finals
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == second[key], f"{key} differs between identical runs"


def test_the_reference_is_not_being_compared_against_itself(result):
    """A reference that silently became the prediction would make every
    accuracy number above zero and meaningless."""
    cfg, _, _ = result
    ctx = configure_runtime(cfg)
    from pinnslab.benchmarks.problem import build_problem
    from pinnslab.eval.metrics import relative_l2, uniform_grid

    problem = build_problem(cfg.problem)
    grid = uniform_grid(problem.domain, problem.eval_resolution, dtype=ctx.dtype)
    truth = problem.reference_at(grid)

    assert relative_l2(torch.zeros_like(truth), truth) == pytest.approx(1.0)
    assert float(truth.abs().max()) > 0.5
