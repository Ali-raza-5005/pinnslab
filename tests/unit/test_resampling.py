"""The sampler seam, and what makes a resampling run survive a session death.

Two claims are under test here, and they are the two that paper 1 (sampling)
stands on:

1. **A new sampler is one registered file.** It is selected by a config's
   ``strategy:``, built through :mod:`pinnslab.geometry.samplers`, handed the
   live :class:`TrainState`, and used by the real training loop — with no edit
   to anything in ``pinnslab/`` (CLAUDE.md rule 9).
2. **A resampling run resumes onto the cloud it was training on.** Not a fresh
   one: the cloud in force at step *k* is part of the experiment, and for an
   adaptive sampler it is not recoverable from the RNG alone, because it was
   drawn against a network that no longer exists after the resume.

The adaptive sampler used below deliberately lives in the test file rather than
in ``pinnslab``. CLAUDE.md rule 2: method code is born in a paper repo and is
promoted only when a second paper needs it. What the library owns is the seam;
what a paper owns is the sampler. ``examples/rad_sampler.py`` is the same
statement in runnable form.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.components import SAMPLERS
from pinnslab.geometry.samplers import Sampler, register_sampler
from pinnslab.registry.config import (
    CheckpointSpec,
    EvalSpec,
    LoggingSpec,
    NetSpec,
    OptimizerSpec,
    PointSetSpec,
    ProblemSpec,
    ResidualSpec,
    RunConfig,
    SamplingSpec,
    StageSpec,
)
from pinnslab.registry.run import Run
from pinnslab.registry.schema import MetricSchedule
from pinnslab.training.build import build_trainer
from pinnslab.training.checkpoint import load_checkpoint
from pinnslab.utils.device import configure_runtime

pytestmark = pytest.mark.unit


class _Boom(RuntimeError):
    """Stands in for a Kaggle session being killed."""


# -- a sampler written the way a paper repo would write one --------------------


@register_sampler("test.recorder")
class RecordingSampler(Sampler):
    """Uniform points, but it records what the seam actually handed it.

    Everything an adaptive sampler needs is asserted through this: the networks
    (to score candidate points), the step (to anneal), the cloud currently in
    force (to keep part of it), and the trainer's generator (so the draw is
    inside the checkpoint).
    """

    #: Class-level so a test can read it without reaching into the trainer.
    calls: list[dict] = []

    def __init__(self, spec, problem):
        self.spec = spec
        self.problem = problem
        self.draws = 0

    def __call__(self, state, current=None):
        self.draws += 1
        RecordingSampler.calls.append(
            {
                "step": state.step,
                "nets": sorted(state.nets),
                "had_points": None if current is None else tuple(current.shape),
                "draws": self.draws,
            }
        )
        return self.problem.domain.sample(
            self.spec.region,
            self.spec.n,
            generator=state.generator,
            strategy="pseudo",
            dtype=state.dtype,
            device=state.device,
        )

    def state_dict(self):
        return {"draws": self.draws}

    def load_state_dict(self, payload):
        self.draws = int(payload["draws"])


@register_sampler("test.residual_adaptive")
class ResidualAdaptiveSampler(Sampler):
    """A real adaptive sampler in miniature: keeps the worst points, redraws the rest.

    The mechanism is the one paper 1 is about — score candidates by the PDE
    residual and concentrate points where it is large — reduced to the smallest
    version that still exercises every requirement: it reads the network, it
    reads its own current cloud, it draws from the trainer's generator, and it
    carries state across a resume.
    """

    def __init__(self, spec, problem):
        self.spec = spec
        self.problem = problem
        self.keep = int(spec.options.get("keep", 8))
        self.generations = 0

    def __call__(self, state, current=None):
        self.generations += 1
        fresh = self.problem.domain.sample(
            self.spec.region,
            self.spec.n,
            generator=state.generator,
            strategy="pseudo",
            dtype=state.dtype,
            device=state.device,
        )
        if current is None:
            return fresh

        # Score the cloud we are holding by |u| and keep the largest — a stand-in
        # for the residual, and enough to make the draw depend on the network.
        with torch.no_grad():
            score = state.nets["u"](current).abs().squeeze(-1)
        keep = min(self.keep, current.shape[0])
        worst = torch.topk(score, keep).indices
        return torch.cat([current[worst], fresh[keep:]], dim=0)

    def state_dict(self):
        return {"generations": self.generations}

    def load_state_dict(self, payload):
        self.generations = int(payload["generations"])


# -- configs -------------------------------------------------------------------


def resampling_config(*, strategy: str = "pseudo", steps: int = 40, **overrides):
    """A real Burgers run, shrunk, that resamples every 10 steps.

    Checkpointed every 5 steps so a kill at step 27 lands *between* resamples —
    the case where a run that redrew on resume would silently diverge from the
    uninterrupted one.
    """
    base = dict(
        name="resampling",
        seed=3,
        dtype="float64",
        device="cpu",
        problem=ProblemSpec(name="burgers1d", options={"nu": 0.3183098861837907}),
        nets={"u": NetSpec(arch="mlp", inputs=2, outputs=1, width=8, depth=2)},
        residuals={
            "pde": ResidualSpec(kind="burgers1d.pde", points=("interior",)),
            "ic": ResidualSpec(kind="burgers1d.ic", points=("initial",)),
        },
        sampling=SamplingSpec(
            points={
                "interior": PointSetSpec(region="interior", n=32, strategy=strategy),
                "initial": PointSetSpec(region="initial", n=8),
            }
        ),
        stages=[
            StageSpec(
                name="adam",
                steps=steps,
                optimizers=[OptimizerSpec(name="adam", lr=1e-2)],
                resample_every=10,
            )
        ],
        eval=EvalSpec(best_metric="loss", best_mode="min"),
        logging=LoggingSpec(trace=MetricSchedule(every=20)),
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=5),
    )
    base.update(overrides)
    return RunConfig(**base)


def train(cfg, root, run_id, *, die_at: int | None = None):
    """Run to completion, or die at a step the way a killed session does."""
    ctx = configure_runtime(cfg)
    run = Run.create_or_resume(cfg, root, run_id)
    trainer = build_trainer(cfg, ctx, run)

    if die_at is not None:
        inner = trainer.residual_fn

        def residual(state):
            if state.step == die_at:
                raise _Boom
            return inner(state)

        trainer.residual_fn = residual

    trainer.fit()
    return trainer


def final_params(trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.nets["u"].named_parameters()}


# -- the registry --------------------------------------------------------------


def test_every_geometric_strategy_is_registered():
    """The registry is the list of legal ``strategy:`` values, so it has to
    contain the built-ins rather than the config falling through to DeepXDE."""
    assert {"pseudo", "lhs", "halton", "hammersley", "sobol"} <= set(SAMPLERS.keys())


def test_an_unknown_sampler_names_the_registered_ones():
    cfg = resampling_config(strategy="no_such_sampler")
    with pytest.raises(KeyError, match="pseudo"):
        build_trainer(cfg, configure_runtime(cfg), _throwaway_run(cfg))


def test_a_registered_sampler_is_selected_by_the_config(results_root):
    """Rule 9 end to end: one new file, one decorator, zero edits to core."""
    RecordingSampler.calls.clear()
    cfg = resampling_config(strategy="test.recorder", steps=20)

    train(cfg, results_root, "recorder")

    assert RecordingSampler.calls, "the config named it and it was never called"
    assert all(c["nets"] == ["u"] for c in RecordingSampler.calls)


def test_a_sampler_sees_the_step_and_the_cloud_it_is_replacing(results_root):
    """What an adaptive sampler needs: where training is, and what it is
    holding. The first draw has no predecessor and says so with ``None``."""
    RecordingSampler.calls.clear()
    cfg = resampling_config(strategy="test.recorder", steps=30)

    train(cfg, results_root, "sees")

    first, *rest = RecordingSampler.calls
    assert first["had_points"] is None
    assert all(c["had_points"] == (32, 2) for c in rest)
    assert [c["step"] for c in rest] == [0, 10, 20]


def test_the_batched_search_path_hands_samplers_the_candidate_networks():
    """The population evaluator draws each candidate's cloud through the same
    sampler objects, so an adaptive one must find real networks there rather
    than an empty dict — it may score even its first draw with them."""
    from pinnslab.search.evaluate import BatchedEvaluator

    RecordingSampler.calls.clear()
    cfg = resampling_config(strategy="test.recorder", steps=5)

    BatchedEvaluator()([cfg], steps=5)

    assert RecordingSampler.calls, "the batched path never drew a cloud"
    assert RecordingSampler.calls[0]["nets"] == ["u"]


def test_a_geometric_sampler_rejects_options_it_would_ignore():
    """A forwarded option nobody reads is a typo or the wrong sampler name, and
    silently ignoring it means training a different experiment than declared."""
    with pytest.raises(TypeError, match="takes no options"):
        cfg = resampling_config()
        cfg = cfg.model_copy(
            update={
                "sampling": SamplingSpec(
                    points={
                        "interior": PointSetSpec(
                            region="interior", n=32, options={"k": 1.0}
                        ),
                        "initial": PointSetSpec(region="initial", n=8),
                    }
                )
            }
        )
        build_trainer(cfg, configure_runtime(cfg), _throwaway_run(cfg))


# -- resume --------------------------------------------------------------------


def test_the_cloud_is_in_the_checkpoint(results_root):
    """The record that makes the rest of this file possible. Without it a
    resumed run has no way back to the points it was training on."""
    cfg = resampling_config(steps=20)
    trainer = train(cfg, results_root, "cloud")

    payload = load_checkpoint(trainer.checkpoints.last_path)

    assert set(payload.points) == {"interior", "initial"}
    for name, points in payload.points.items():
        assert torch.equal(points, trainer.state.points[name])


def test_a_resampling_run_resumes_bit_exactly(results_root):
    """The load-bearing test for paper 1: a session killed *between* resamples
    must come back on the cloud it was using, not on a fresh draw.

    Killed at step 27 with ``resample_every=10``, so the run resumes from the
    step-25 checkpoint holding the cloud drawn at step 20 and must keep it
    until step 30.
    """
    cfg = resampling_config(steps=40)

    reference = final_params(train(cfg, results_root, "uninterrupted"))

    with pytest.raises(_Boom):
        train(cfg, results_root, "killed", die_at=27)
    resumed = final_params(train(cfg, results_root, "killed"))

    for name, param in resumed.items():
        assert torch.equal(param, reference[name]), f"{name} drifted after resume"


def test_an_adaptive_run_resumes_bit_exactly(results_root):
    """The same claim for a sampler whose draw depends on the *network*, which
    is the case no amount of RNG replay could reconstruct."""
    cfg = resampling_config(strategy="test.residual_adaptive", steps=40)

    reference = final_params(train(cfg, results_root, "adaptive"))

    with pytest.raises(_Boom):
        train(cfg, results_root, "adaptive_killed", die_at=23)
    resumed = final_params(train(cfg, results_root, "adaptive_killed"))

    for name, param in resumed.items():
        assert torch.equal(param, reference[name]), f"{name} drifted after resume"


def test_a_samplers_own_state_survives_the_resume(results_root):
    """A sampler that counts generations must not restart at zero: its counter
    is part of the experiment exactly as the optimizer's moments are."""
    cfg = resampling_config(strategy="test.recorder", steps=40)

    with pytest.raises(_Boom):
        train(cfg, results_root, "stateful", die_at=27)
    resumed = train(cfg, results_root, "stateful")

    sampler = resumed.on_resample.samplers["interior"]
    # 1 initial draw + resamples at 0/10/20 before the kill, then the resumed
    # session's own initial draw and the resample at 30.
    assert sampler.draws > 1
    payload = load_checkpoint(resumed.checkpoints.last_path)
    assert payload.sampler_state["interior"] == {"draws": sampler.draws}


def test_a_checkpoint_from_a_different_sampling_config_is_refused(results_root):
    """Sampler state is keyed by point group; a checkpoint carrying a group
    this config does not declare means sampling changed underneath it."""
    cfg = resampling_config(strategy="test.recorder", steps=10)
    trainer = train(cfg, results_root, "mismatch")

    with pytest.raises(ValueError, match="sampling changed"):
        trainer.on_resample.load_state_dict({"interior": {"draws": 1}, "ghost": {}})


def _throwaway_run(cfg) -> Run:
    import tempfile
    from pathlib import Path

    return Run.create(cfg, Path(tempfile.mkdtemp(prefix="pinnslab_test_")))
