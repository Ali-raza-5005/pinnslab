"""Shared fixtures.

The toy problem is a linear fit, not a PDE: step 1 is infrastructure, and the
trainer must be testable without ``physics/``, ``models/`` or ``geometry/``
existing. Building the network inline here also proves the trainer has no
dependency on ``models/``.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from pinnslab.losses.weighting import MeanWeighting
from pinnslab.registry.config import (
    CheckpointSpec,
    EvalSpec,
    LoggingSpec,
    OptimizerSpec,
    RunConfig,
    StageSpec,
)
from pinnslab.registry.schema import MetricSchedule
from pinnslab.utils.device import configure_runtime
from pinnslab.utils.seeding import capture_rng_state, restore_rng_state

TRUE_SLOPE = 2.0
TRUE_INTERCEPT = 1.0


@pytest.fixture(autouse=True)
def isolate_global_state():
    """Leave the process exactly as each test found it.

    ``set_seed`` and ``configure_runtime`` both mutate process-global state — RNG
    streams, the determinism flags, the default dtype, an environment variable.
    Without this, tests silently depend on execution order: anything running
    after a trainer test inherits float64 and determinism-on, so a test asserting
    "``deterministic=False`` leaves the flags alone" could not be written at all.
    """
    rng = capture_rng_state()
    dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        restore_rng_state(rng)
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas


def make_net(in_dim: int = 1, width: int = 8) -> nn.Module:
    """A tiny MLP. Deterministic given the ambient torch seed."""
    return nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, 1))


def linear_residual(state) -> dict[str, torch.Tensor]:
    """Per-point residual of shape (N,), drawn from the trainer's own generator.

    Drawing from ``state.generator`` is what makes the resume tests meaningful:
    a resumed run that restored weights but not RNG would produce a different
    point cloud and silently diverge from the uninterrupted run.
    """
    x = torch.rand(32, 1, generator=state.generator, dtype=state.dtype)
    target = TRUE_SLOPE * x + TRUE_INTERCEPT
    return {"fit": (state.nets["u"](x) - target).squeeze(-1)}


def toy_config(**overrides) -> RunConfig:
    base = {
        "name": "toy",
        "seed": 7,
        "dtype": "float64",
        "device": "cpu",
        "stages": [
            StageSpec(
                name="adam",
                steps=20,
                optimizers=[OptimizerSpec(name="adam", lr=1e-2)],
            )
        ],
        "eval": EvalSpec(best_metric="loss", best_mode="min"),
        "logging": LoggingSpec(trace=MetricSchedule(every=5)),
        "checkpoint": CheckpointSpec(every_seconds=None, every_steps=10),
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def cfg() -> RunConfig:
    return toy_config()


@pytest.fixture
def weighting() -> MeanWeighting:
    return MeanWeighting()


@pytest.fixture
def results_root(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    return root


def setup_run(cfg: RunConfig):
    """``configure_runtime`` then build nets — always in that order.

    The seed and dtype must be in force before a single parameter is allocated,
    or two "identical" runs start from different weights.
    """
    ctx = configure_runtime(cfg)
    nets = {"u": make_net().to(device=ctx.device, dtype=ctx.dtype)}
    return ctx, nets
