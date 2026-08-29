"""The optimizer seam: what the loop may be handed, and what it must refuse.

Until 2026-08-29 ``Trainer`` chose how to drive an optimizer with
``isinstance(opt, torch.optim.LBFGS)``. Anything else registered through
``@register_optimizer`` was driven by the first-order path, which calls
``step()`` with no objective — so a derivative-free method could be registered,
configured, run to completion, and never once see the loss. Nothing raised.

The toy swarm below is the smallest thing with the shape of the population
methods DESIGN.md §6 names as a research direction (CSO, PSO, DE over network
weights): it scores P candidates per step through the closure, moves losers
toward winners, installs the best, and re-evaluates. It is not a good optimizer
and is not meant to be. It exists so that the *seam* has a test that is not
L-BFGS.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.components import register_optimizer
from pinnslab.registry.config import CheckpointSpec, OptimizerSpec, StageSpec
from pinnslab.registry.run import Run
from pinnslab.training.optimizers import requires_closure, uses_gradients
from pinnslab.training.trainer import Trainer
from tests.conftest import setup_run, toy_config

pytestmark = pytest.mark.unit


# -- a deterministic residual -------------------------------------------------
#
# tests.conftest.linear_residual draws its cloud from ``state.generator``, which
# is exactly what a closure-based optimizer may not have: it calls the closure
# several times per step and needs those calls to be comparable to each other.
# A fixed cloud is the trainer's documented contract for this path.

_X = torch.linspace(0.0, 1.0, 32).unsqueeze(-1)


def fixed_residual(state) -> dict[str, torch.Tensor]:
    x = _X.to(dtype=state.dtype)
    target = 2.0 * x + 1.0
    return {"fit": (state.nets["u"](x) - target).squeeze(-1)}


def _flat(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _write(params: list[torch.Tensor], vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in params:
            size = param.numel()
            param.copy_(vector[offset : offset + size].view_as(param))
            offset += size


class ToySwarm(torch.optim.Optimizer):
    """A population method in the fewest lines that still exercise the seam.

    The two capability attributes are the whole protocol. Everything else here
    is ordinary ``torch.optim.Optimizer``: the population lives in ``self.state``
    so it round-trips through ``state_dict`` with no help from the trainer, which
    is what makes a resumed swarm bit-exact.
    """

    requires_closure = True
    uses_gradients = False

    def __init__(self, params, lr=0.5, pop_size=4, seed=0):
        super().__init__(list(params), {"lr": lr, "pop_size": pop_size, "seed": seed})
        self._params = [p for g in self.param_groups for p in g["params"]]

    def _score(self, closure, candidate: torch.Tensor) -> torch.Tensor:
        _write(self._params, candidate)
        return closure()

    def step(self, closure=None):
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        if "population" not in state:
            base = _flat(self._params)
            spread = torch.randn(
                group["pop_size"],
                base.numel(),
                generator=torch.Generator().manual_seed(group["seed"]),
                dtype=base.dtype,
            )
            state["population"] = base.unsqueeze(0) + 0.1 * spread
            state["population"][0] = base
            state["calls"] = 0

        population = state["population"]
        # Re-derived from (seed, call count) rather than carried as a generator
        # state tensor: an int survives ``load_state_dict``'s casting of state
        # entries, and a ByteTensor's survival is a torch implementation detail
        # we would rather not pin a resume test to.
        generator = torch.Generator().manual_seed(group["seed"] + state["calls"])
        state["calls"] += 1

        fitness = torch.stack([self._score(closure, c) for c in population])

        order = torch.randperm(population.shape[0], generator=generator)
        for left, right in zip(order[0::2], order[1::2], strict=False):
            better = fitness[left] <= fitness[right]
            win, lose = (left, right) if better else (right, left)
            pull = torch.rand(
                population.shape[1], generator=generator, dtype=population.dtype
            )
            gap = population[win] - population[lose]
            population[lose] += group["lr"] * pull * gap

        # Install the winner and re-evaluate it *last*, so the value returned and
        # the trainer's ``_last_residuals`` both describe the parameters this step
        # leaves behind (training/optimizers.py, contract 2).
        best = int(torch.argmin(fitness))
        return self._score(closure, population[best])


class _Silent(ToySwarm):
    """Honours the closure protocol but not the "return the value" half of it."""

    def step(self, closure=None):
        super().step(closure)
        return None


class _Unreachable(torch.optim.SGD):
    """Derivative-free but never asks for the objective — a contradiction."""

    uses_gradients = False


@register_optimizer("toy_swarm")
def _build_toy_swarm(params, spec):
    return ToySwarm(params, lr=spec.lr, **spec.options)


@register_optimizer("toy_silent")
def _build_toy_silent(params, spec):
    return _Silent(params, lr=spec.lr)


@register_optimizer("toy_unreachable")
def _build_toy_unreachable(params, spec):
    return _Unreachable(params, lr=spec.lr)


# -- helpers ------------------------------------------------------------------


def _build(cfg, results_root, run_id="swarm"):
    ctx, nets = setup_run(cfg)
    return Trainer(
        cfg=cfg,
        ctx=ctx,
        nets=nets,
        residual_fn=fixed_residual,
        weighting=lambda residuals, state: (residuals["fit"] ** 2).mean(),
        run=Run.create_or_resume(cfg, results_root, run_id),
    )


def _swarm_stage(steps=15, **overrides):
    spec = {"name": "toy_swarm", "lr": 0.5, "options": {"pop_size": 4, "seed": 3}}
    spec.update(overrides)
    return StageSpec(name="swarm", steps=steps, optimizers=[OptimizerSpec(**spec)])


# -- the seam works -----------------------------------------------------------


def test_the_predicates_read_capability_not_type():
    param = [torch.zeros(2, requires_grad=True)]
    assert not requires_closure(torch.optim.Adam(param))
    assert uses_gradients(torch.optim.Adam(param))
    # The one isinstance check left, and it is a shim for torch's own class.
    assert requires_closure(torch.optim.LBFGS(param))
    assert uses_gradients(torch.optim.LBFGS(param))
    swarm = ToySwarm(param)
    assert requires_closure(swarm)
    assert not uses_gradients(swarm)


def test_a_derivative_free_optimizer_trains_through_the_ordinary_loop(results_root):
    """The point of the seam change: one config, one hash, one Run, no fork."""
    cfg = toy_config(stages=[_swarm_stage()])
    trainer = _build(cfg, results_root)
    row = trainer.fit()

    assert row.status.value == "completed"
    assert row.steps_completed == 15
    trace = trainer.run.read_trace()
    assert trace[-1].metrics["loss"] < trace[0].metrics["loss"]


def test_a_swarm_stage_composes_with_a_gradient_stage(results_root):
    """``stages: [{swarm}, {adam}]`` — the schedule this seam exists to allow."""
    cfg = toy_config(
        stages=[
            _swarm_stage(steps=10),
            StageSpec(name="adam", steps=30, optimizers=[OptimizerSpec(lr=1e-2)]),
        ]
    )
    trainer = _build(cfg, results_root)
    row = trainer.fit()

    assert row.steps_completed == 40
    assert set(trainer._timings) >= {"stage.swarm.seconds", "stage.adam.seconds"}


def test_reported_fitness_belongs_to_the_parameters_left_installed(results_root):
    """PINNSLAB.md §21.9 must not reach this path.

    The first-order loop reports the loss at the parameters it is about to
    update, so step *k*'s ``loss`` is θ(k−1) while its checkpoint is θ(k). A
    derivative-free step re-evaluates its winner after installing it, so both
    describe θ(k) — asserted here by re-running the objective at the parameters
    the run finished on and demanding the *same* number.
    """
    cfg = toy_config(stages=[_swarm_stage()])
    trainer = _build(cfg, results_root)
    trainer.fit()

    final = trainer.run.read_trace()[-1].metrics["loss"]
    assert float(trainer._forward().detach()) == pytest.approx(final, rel=1e-15)


def test_the_first_order_path_still_reports_the_pre_update_loss(results_root):
    """The counterpart, pinned rather than fixed.

    Fixing it costs a second forward pass on every step of every run in the
    repo. This test exists so the off-by-one is a recorded property with a
    failing test behind any silent change, not folklore in an audit file.
    """
    cfg = toy_config(
        stages=[StageSpec(name="adam", steps=20, optimizers=[OptimizerSpec(lr=1e-2)])]
    )
    trainer = _build(cfg, results_root, run_id="first-order")
    trainer.fit()

    final = trainer.run.read_trace()[-1].metrics["loss"]
    assert float(trainer._forward().detach()) != pytest.approx(final, rel=1e-12)


def test_gradient_free_resume_is_bit_exact(results_root):
    """The population is optimizer state, so the generic checkpoint carries it."""

    class _Boom(RuntimeError):
        pass

    cfg = toy_config(
        stages=[_swarm_stage(steps=20)],
        checkpoint=CheckpointSpec(every_seconds=None, every_steps=5),
    )

    def run(run_id, die_at=None):
        ctx, nets = setup_run(cfg)

        def residual(state):
            if die_at is not None and state.step == die_at:
                raise _Boom
            return fixed_residual(state)

        Trainer(
            cfg=cfg,
            ctx=ctx,
            nets=nets,
            residual_fn=residual,
            weighting=lambda residuals, state: (residuals["fit"] ** 2).mean(),
            run=Run.create_or_resume(cfg, results_root, run_id),
        ).fit()
        return nets

    reference = run("swarm-uninterrupted")
    with pytest.raises(_Boom):
        run("swarm-killed", die_at=11)
    resumed = run("swarm-killed")

    expected = dict(reference["u"].named_parameters())
    for name, param in resumed["u"].named_parameters():
        assert torch.equal(param, expected[name]), f"{name} drifted after resume"


# -- the seam refuses ---------------------------------------------------------


def test_a_closure_optimizer_refuses_to_share_a_stage(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="bad",
                steps=1,
                optimizers=[
                    OptimizerSpec(name="toy_swarm", params=r"u\.0\..*", lr=0.5),
                    OptimizerSpec(name="adam", params=r"u\.2\..*", lr=1e-3),
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="own stage"):
        _build(cfg, results_root, run_id="share").fit()


def test_a_closure_optimizer_refuses_ascent(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="bad",
                steps=1,
                optimizers=[OptimizerSpec(name="toy_swarm", lr=0.5, direction="max")],
            )
        ]
    )
    with pytest.raises(ValueError, match="nothing to flip"):
        _build(cfg, results_root, run_id="ascent").fit()


def test_a_derivative_free_optimizer_refuses_grad_clipping(results_root):
    """The refusal DESIGN.md §6 CORRECTION 2 requires: never silently drop it."""
    cfg = toy_config(
        stages=[
            StageSpec(
                name="bad",
                steps=1,
                optimizers=[OptimizerSpec(name="toy_swarm", lr=0.5, max_grad_norm=1.0)],
            )
        ]
    )
    with pytest.raises(ValueError, match="no gradient is ever computed"):
        _build(cfg, results_root, run_id="clip").fit()


def test_a_derivative_free_optimizer_must_ask_for_the_closure(results_root):
    cfg = toy_config(
        stages=[
            StageSpec(
                name="bad",
                steps=1,
                optimizers=[OptimizerSpec(name="toy_unreachable", lr=0.5)],
            )
        ]
    )
    with pytest.raises(ValueError, match="requires_closure=True"):
        _build(cfg, results_root, run_id="unreachable").fit()


def test_a_closure_optimizer_returning_none_is_refused(results_root):
    """Otherwise the trace and the divergence check silently fill with nan."""
    cfg = toy_config(
        stages=[
            StageSpec(
                name="silent",
                steps=1,
                optimizers=[OptimizerSpec(name="toy_silent", lr=0.5)],
            )
        ]
    )
    with pytest.raises(ValueError, match="returned None"):
        _build(cfg, results_root, run_id="silent").fit()
