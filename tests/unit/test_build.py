"""Config -> Trainer assembly.

This module is what makes CLAUDE.md rule 4 true in practice: if a run can be
built from anything other than a validated, hashed config, then hyperparameters
are back in scripts and nothing downstream can be trusted. So most of what is
tested here is refusal.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.registry.config import (
    CheckpointSpec,
    EvalSpec,
    LoggingSpec,
    NetSpec,
    OptimizerSpec,
    ParamSpec,
    PointSetSpec,
    ProblemSpec,
    ResidualSpec,
    RunConfig,
    SamplingSpec,
    StageSpec,
    WeightingSpec,
)
from pinnslab.registry.run import Run
from pinnslab.registry.schema import MetricSchedule
from pinnslab.training.build import POINTS, _gather, assemble, build_trainer
from pinnslab.utils.device import configure_runtime

pytestmark = pytest.mark.unit


def burgers_config(**overrides) -> RunConfig:
    """The smallest honest Burgers config; steps kept tiny for unit speed."""
    base = {
        "name": "burgers-unit",
        "seed": 0,
        "dtype": "float64",
        "device": "cpu",
        "problem": ProblemSpec(name="burgers1d"),
        "nets": {"u": NetSpec(inputs=2, outputs=1, width=8, depth=2)},
        "residuals": {
            # The PDE holds on the closed domain; interior-only costs ~6x in
            # rel-L2 on this benchmark (see ResidualSpec.points).
            "pde": ResidualSpec(
                kind="burgers1d.pde", points=["interior", "initial", "boundary"]
            ),
            "ic": ResidualSpec(kind="burgers1d.ic", points="initial"),
            "bc": ResidualSpec(kind="burgers1d.bc", points="boundary"),
        },
        "sampling": SamplingSpec(
            points={
                "interior": PointSetSpec(region="interior", n=64),
                "initial": PointSetSpec(region="initial", n=16),
                "boundary": PointSetSpec(region="boundary", n=8),
            }
        ),
        "weighting": WeightingSpec(kind="mean"),
        "stages": [
            StageSpec(
                name="adam", steps=3, optimizers=[OptimizerSpec(name="adam", lr=1e-3)]
            )
        ],
        "eval": EvalSpec(best_metric="rel_l2", best_mode="min"),
        "logging": LoggingSpec(trace=MetricSchedule(every=2)),
        "checkpoint": CheckpointSpec(every_seconds=None, every_steps=10**9),
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def built(results_root):
    cfg = burgers_config()
    ctx = configure_runtime(cfg)
    return cfg, ctx, Run.create(cfg, results_root)


# -- what it builds -----------------------------------------------------------


def test_assemble_builds_every_declared_piece(built):
    cfg, ctx, _ = built
    parts = assemble(cfg, ctx)

    assert parts.problem.name == "burgers1d"
    assert set(parts.nets) == {"u"}
    assert parts.eval_fn is not None


def test_nets_take_the_configs_dtype_and_device(built):
    cfg, ctx, _ = built
    net = assemble(cfg, ctx).nets["u"]
    assert next(net.parameters()).dtype is torch.float64


def test_the_first_point_cloud_is_drawn_before_training(built):
    """The trainer only fires ``on_resample`` when a stage sets
    ``resample_every``; with fixed points it would never fire and the first
    residual evaluation would find an empty scratch."""
    cfg, ctx, run = built
    trainer = build_trainer(cfg, ctx, run)

    points = trainer.state.scratch[POINTS]
    assert set(points) == {"interior", "initial", "boundary"}
    assert points["interior"].shape == (64, 2)
    assert points["initial"].shape == (16, 2)


def test_each_residual_gets_its_own_point_group(built):
    """The IC term evaluated on interior points would be enforcing the initial
    condition at every time — a wrong problem that still trains."""
    cfg, ctx, run = built
    trainer = build_trainer(cfg, ctx, run)

    residuals = trainer.residual_fn(trainer.state)

    assert residuals["pde"].shape == (64 + 16 + 8,)
    assert residuals["ic"].shape == (16,)
    assert residuals["bc"].shape == (8,)


def test_a_multi_group_residual_concatenates_in_declared_order(built):
    """Kept as one ``(N,)`` vector rather than evaluated per group, so per-point
    weighting still sees a single population (CLAUDE.md rule 5)."""
    cfg, ctx, run = built
    trainer = build_trainer(cfg, ctx, run)
    points = trainer.state.scratch[POINTS]

    gathered = _gather(points, ("interior", "initial", "boundary"))

    assert torch.equal(
        gathered,
        torch.cat([points["interior"], points["initial"], points["boundary"]], dim=0),
    )


def test_the_pde_is_enforced_on_the_points_the_config_names(built):
    """The finding this whole mechanism exists for: enforcing the PDE on the
    interior alone costs ~6x in rel-L2 on Burgers while *lowering* the loss,
    because interior-only is simply an easier objective. Making the group list
    explicit means the choice is hashed and visible in the results row rather
    than buried in a framework's point bookkeeping."""
    cfg, ctx, run = built
    pde = cfg.residuals["pde"].model_copy(update={"points": ("interior",)})
    interior_only = cfg.model_copy(
        update={"residuals": {**cfg.residuals, "pde": pde}}
    )
    trainer = build_trainer(interior_only, ctx, run)

    assert trainer.residual_fn(trainer.state)["pde"].shape == (64,)


def test_it_trains_end_to_end(built):
    cfg, ctx, run = built
    row = build_trainer(cfg, ctx, run).fit()

    assert row.status == "completed"
    assert row.steps_completed == 3
    assert "rel_l2" in row.final_metrics
    assert "max_error" in row.final_metrics


def test_extra_params_reach_the_optimizer(results_root):
    """DESIGN.md §4 conformance item 2: an inverse problem is an ordinary run
    whose coefficients happen to be parameters."""
    cfg = burgers_config(
        extra_params={"nu_hat": ParamSpec(init=0.01, trainable=True)},
        stages=[
            StageSpec(
                name="adam",
                steps=2,
                optimizers=[OptimizerSpec(name="adam", lr=1e-3, params=r"extra\..*")],
            )
        ],
    )
    ctx = configure_runtime(cfg)
    trainer = build_trainer(cfg, ctx, Run.create(cfg, results_root))

    assert "extra.nu_hat" in trainer.named_parameters()
    assert trainer.extra_params["nu_hat"].requires_grad


def test_the_eval_grid_is_fixed_across_calls(built):
    """A metric computed on a moving grid would make two runs incomparable."""
    cfg, ctx, run = built
    trainer = build_trainer(cfg, ctx, run)

    first = trainer.eval_fn(trainer.state)
    second = trainer.eval_fn(trainer.state)

    assert first == second


# -- reproducibility ----------------------------------------------------------


def test_one_seed_gives_one_point_cloud(results_root):
    cfg = burgers_config()
    clouds = []
    for run_id in ("a", "b"):
        ctx = configure_runtime(cfg)
        trainer = build_trainer(cfg, ctx, Run.create(cfg, results_root, run_id=run_id))
        clouds.append(trainer.state.scratch[POINTS]["interior"])

    assert torch.equal(*clouds)


def test_different_seeds_give_different_point_clouds(results_root):
    clouds = []
    for run_id, seed in (("a", 1), ("b", 2)):
        cfg = burgers_config(seed=seed)
        ctx = configure_runtime(cfg)
        trainer = build_trainer(cfg, ctx, Run.create(cfg, results_root, run_id=run_id))
        clouds.append(trainer.state.scratch[POINTS]["interior"])

    assert not torch.equal(*clouds)


# -- what it refuses ----------------------------------------------------------


def test_a_config_with_no_residuals_is_refused(results_root):
    """The empty defaults exist for the callable escape hatch, not as a way to
    smuggle hyperparameters back into a script."""
    from tests.conftest import toy_config

    cfg = toy_config()
    with pytest.raises(ValueError, match="nothing to train"):
        assemble(cfg, configure_runtime(cfg))


def test_residuals_without_a_problem_are_refused(built):
    cfg, ctx, _ = built
    with pytest.raises(ValueError, match="no problem"):
        assemble(cfg.model_copy(update={"problem": None}), ctx)


def test_an_unknown_residual_kind_lists_the_registered_ones(built):
    cfg, ctx, _ = built
    broken = cfg.model_copy(
        update={
            "residuals": {"pde": ResidualSpec(kind="burgers2d.pde", points="interior")}
        }
    )
    with pytest.raises(KeyError, match="burgers1d.pde"):
        assemble(broken, ctx)


def test_an_unknown_problem_lists_the_registered_ones(built):
    cfg, ctx, _ = built
    with pytest.raises(KeyError, match="burgers1d"):
        assemble(cfg.model_copy(update={"problem": ProblemSpec(name="kdv")}), ctx)


def test_training_without_the_builder_says_what_to_do(built):
    """Constructing a Trainer directly with a config-driven residual function
    leaves scratch empty; the error has to name the fix."""
    cfg, ctx, run = built
    trainer = build_trainer(cfg, ctx, run)
    trainer.state.scratch.clear()

    with pytest.raises(RuntimeError, match="build_trainer"):
        trainer.residual_fn(trainer.state)
