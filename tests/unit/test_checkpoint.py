"""Resume must be bit-exact, or a session death silently changes the experiment."""

from __future__ import annotations

import pytest
import torch

from pinnslab.registry.config import CheckpointSpec, OptimizerSpec, StageSpec
from pinnslab.registry.run import Run
from pinnslab.training.checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from pinnslab.training.trainer import Trainer
from pinnslab.utils.seeding import capture_rng_state
from tests.conftest import linear_residual, setup_run, toy_config

pytestmark = pytest.mark.unit


class _Boom(RuntimeError):
    """Stands in for a Kaggle session being killed."""


def _long_config(**overrides):
    return toy_config(
        stages=[
            StageSpec(
                name="adam",
                steps=200,
                optimizers=[OptimizerSpec(name="adam", lr=1e-2)],
            )
        ],
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=50),
        **overrides,
    )


def _train(cfg, root, run_id, *, die_at: int | None = None):
    ctx, nets = setup_run(cfg)
    run = Run.create_or_resume(cfg, root, run_id)

    def residual(state):
        if die_at is not None and state.step == die_at:
            raise _Boom
        return linear_residual(state)

    trainer = Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets,
        residual_fn=residual,
        weighting=lambda residuals, state: (residuals["fit"] ** 2).mean(),
        run=run,
    )
    trainer.fit()
    return nets


def test_resume_is_bit_exact(results_root):
    """The load-bearing test of the whole checkpoint layer."""
    cfg = _long_config()

    reference = _train(cfg, results_root, "uninterrupted")

    # Same config, same seed, killed at step 100 and restarted from last.pt.
    with pytest.raises(_Boom):
        _train(cfg, results_root, "killed", die_at=100)
    resumed = _train(cfg, results_root, "killed")

    ref_params = dict(reference["u"].named_parameters())
    for name, param in resumed["u"].named_parameters():
        assert torch.equal(param, ref_params[name]), f"{name} drifted after resume"


def _lbfgs_config(**overrides):
    """L-BFGS with the library's own defaults: strong-Wolfe, max_iter=20."""
    return toy_config(
        stages=[
            StageSpec(
                name="lbfgs",
                steps=12,
                optimizers=[OptimizerSpec(name="lbfgs", lr=0.5)],
            )
        ],
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=4),
        **overrides,
    )


def test_lbfgs_resume_is_bit_exact(results_root):
    """The curvature history round-trips, so an L-BFGS stage is not special.

    This module once claimed the opposite and had the trainer rewind an
    interrupted L-BFGS stage to its boundary, discarding a whole stage per
    session death. If a future torch really does move ``old_dirs`` / ``ro`` /
    ``H_diag`` out of ``state_dict``, this test fails — which is the only way
    that regression would ever be noticed, since a non-bit-exact resume produces
    a plausible run that is quietly a different experiment.
    """
    cfg = _lbfgs_config()

    reference = _train(cfg, results_root, "lbfgs-uninterrupted")

    with pytest.raises(_Boom):
        _train(cfg, results_root, "lbfgs-killed", die_at=6)
    resumed = _train(cfg, results_root, "lbfgs-killed")

    ref_params = dict(reference["u"].named_parameters())
    for name, param in resumed["u"].named_parameters():
        assert torch.equal(param, ref_params[name]), f"{name} drifted after resume"


def test_lbfgs_checkpoints_mid_stage(results_root):
    """Periodic saves are no longer suppressed inside an L-BFGS stage.

    The bit-exactness test above passes under the old rewind policy too — it just
    costs a stage of compute to get there. This is the test that pins the policy
    itself: progress inside the stage must actually reach ``last.pt``.
    """
    cfg = _lbfgs_config()
    with pytest.raises(_Boom):
        _train(cfg, results_root, "lbfgs-mid", die_at=9)

    manager = CheckpointManager(
        results_root / "lbfgs-mid" / "checkpoints",
        cfg.checkpoint,
        config_hash=cfg.identity_hash(),
        seed=cfg.seed,
    )
    payload = manager.load_last()
    assert payload is not None
    assert payload.steps_in_stage > 0, "the L-BFGS stage was rewound to its boundary"
    assert payload.step == 8  # every_steps=4, killed at 9


def test_resume_continues_the_rng_stream(results_root):
    """Weights alone are not enough: the point cloud must continue too."""
    cfg = _long_config()
    with pytest.raises(_Boom):
        _train(cfg, results_root, "k2", die_at=100)

    manager = CheckpointManager(
        (results_root / "k2" / "checkpoints"),
        cfg.checkpoint,
        config_hash=cfg.identity_hash(),
        seed=cfg.seed,
    )
    payload = manager.load_last()
    assert payload is not None
    assert payload.step == 100
    assert "trainer_generator" in payload.rng
    assert payload.rng["numpy"][0] == "MT19937"


def test_checkpoint_loads_with_weights_only(tmp_path):
    """RNG state must survive torch.load's safe loader (torch >= 2.6)."""
    payload = CheckpointPayload(
        step=1,
        stage_index=0,
        steps_in_stage=1,
        nets={"u": {"w": torch.zeros(2)}},
        extra_params={"k": torch.ones(1)},
        optimizers=[{"state": {}, "param_groups": [{"lr": 0.1}]}],
        rng=capture_rng_state(),
        elapsed=1.0,
        config_hash="abc",
        seed=0,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, payload)

    loaded = load_checkpoint(path)  # weights_only=True inside
    assert loaded.step == 1
    assert torch.equal(loaded.extra_params["k"], torch.ones(1))


def test_save_is_atomic_and_leaves_no_temporary(tmp_path):
    payload = CheckpointPayload(
        step=0,
        stage_index=0,
        steps_in_stage=0,
        nets={},
        extra_params={},
        optimizers=[],
        rng=capture_rng_state(),
        elapsed=0.0,
        config_hash="abc",
        seed=0,
    )
    path = tmp_path / "last.pt"
    save_checkpoint(path, payload)
    save_checkpoint(path, payload)
    assert list(tmp_path.iterdir()) == [path]


def test_resume_refuses_a_different_condition(tmp_path):
    payload = CheckpointPayload(
        step=5,
        stage_index=0,
        steps_in_stage=5,
        nets={},
        extra_params={},
        optimizers=[],
        rng=capture_rng_state(),
        elapsed=0.0,
        config_hash="hash-a",
        seed=0,
    )
    save_checkpoint(tmp_path / "last.pt", payload)

    manager = CheckpointManager(
        tmp_path, CheckpointSpec(), config_hash="hash-b", seed=0
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        manager.load_last()
    assert manager.load_last(allow_config_change=True) is not None


def test_resume_refuses_a_different_seed(tmp_path):
    """A run is identified by (config_hash, seed) — both are checked."""
    payload = CheckpointPayload(
        step=5,
        stage_index=0,
        steps_in_stage=5,
        nets={},
        extra_params={},
        optimizers=[],
        rng=capture_rng_state(),
        elapsed=0.0,
        config_hash="hash-a",
        seed=1,
    )
    save_checkpoint(tmp_path / "last.pt", payload)
    manager = CheckpointManager(
        tmp_path, CheckpointSpec(), config_hash="hash-a", seed=2
    )
    with pytest.raises(ValueError, match="seed"):
        manager.load_last()


def test_no_checkpoint_means_no_resume(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSpec(), config_hash="a", seed=0)
    assert manager.load_last() is None


def test_best_mode_direction():
    manager_min = CheckpointManager(".", CheckpointSpec(), config_hash="a", seed=0)
    assert manager_min.is_improvement(0.5, 1.0)
    assert not manager_min.is_improvement(1.5, 1.0)

    manager_max = CheckpointManager(
        ".", CheckpointSpec(), config_hash="a", seed=0, best_mode="max"
    )
    assert manager_max.is_improvement(1.5, 1.0)
    assert manager_max.is_improvement(0.5, None)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_nonfinite_value_is_never_a_best(mode):
    """Otherwise the first NaN latches ``best`` there permanently.

    ``value < nan`` and ``value > nan`` are both False, so nothing could ever
    displace it and ``best.pt`` would keep whichever parameters blew up.
    Reachable only with ``stop_on_nonfinite=False``, which is exactly the
    setting used to study runs that recover from a loss spike.
    """
    manager = CheckpointManager(
        ".", CheckpointSpec(), config_hash="a", seed=0, best_mode=mode
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not manager.is_improvement(bad, None)
        assert not manager.is_improvement(bad, 1.0)

    # ...and a real value still displaces a NaN that predates the guard.
    assert manager.is_improvement(1.0, float("nan"))


def test_to_dict_does_not_copy_the_tensors_it_is_about_to_serialise():
    """``asdict`` would deep-copy the whole model and optimizer state per save.

    The save is synchronous, so references are as good as a snapshot. Pinned
    because reverting to ``dataclasses.asdict`` looks like a tidy-up.
    """
    weight = torch.zeros(4)
    payload = CheckpointPayload(
        step=0,
        stage_index=0,
        steps_in_stage=0,
        nets={"u": {"w": weight}},
        extra_params={},
        optimizers=[],
        rng={},
        elapsed=0.0,
        config_hash="a",
        seed=0,
    )
    assert payload.to_dict()["nets"]["u"]["w"].data_ptr() == weight.data_ptr()
