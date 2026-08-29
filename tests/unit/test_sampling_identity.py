"""The collocation cloud belongs to the sampling, not to the whole config.

The claim under test is the one that makes an optimizer-schedule ablation an
ablation at all: **two runs of one seed that differ only in how they are
optimised must train on the same collocation cloud.**

Until 2026-08-29 they did not. ``Trainer`` derived its sampling stream from
``run.config_hash``, which covers ``stages``, so appending an L-BFGS stage
redrew the cloud that the *Adam* stage before it had trained on. The two
configs

    stages: [adam 15000]
    stages: [adam 15000, lbfgs 500]

therefore did not share their Adam phase. They were two different experiments
that agreed on every field a reader would check, and the difference showed up
as a result rather than as an error: on Burgers at nu=0.01/pi, seed 100,
otherwise identical, rel-L2 at the end of the identical Adam stages was
**0.1405** in one and **0.5644** in the other — a 4x spread from the draw
alone, comfortably larger than the effects such a comparison exists to detect.

It was never biased; both conditions drew from the same distribution of clouds,
so the comparison was valid and merely noisy. But it was noise that pairing
removes for free, and recovering the same signal by adding seeds instead costs
one to two orders of magnitude more compute.

The tests below pin both halves of the rule, because only asserting the first
would be satisfied by a trainer that gave *every* config the same cloud:

1. changing only the optimizer schedule keeps the cloud;
2. changing the geometry, the point counts, the sampling strategy or the
   precision still changes it.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.registry.config import SAMPLING_IDENTITY, RunConfig
from pinnslab.registry.run import Run
from pinnslab.training.build import build_trainer
from pinnslab.utils.device import configure_runtime

pytestmark = pytest.mark.unit


def _config(**overrides) -> RunConfig:
    base = {
        "name": "sampling-identity",
        "seed": 3,
        "dtype": "float64",
        "device": "cpu",
        "problem": {"name": "burgers1d"},
        "nets": {
            "u": {"arch": "mlp", "inputs": 2, "outputs": 1, "width": 8, "depth": 2}
        },
        "residuals": {
            "pde": {"kind": "burgers1d.pde", "points": ["interior", "initial"]},
            "ic": {"kind": "burgers1d.ic", "points": "initial"},
        },
        "sampling": {
            "points": {
                "interior": {"region": "interior", "n": 32, "strategy": "pseudo"},
                "initial": {"region": "initial", "n": 8, "strategy": "pseudo"},
            }
        },
        "stages": [
            {"name": "adam", "steps": 1, "optimizers": [{"name": "adam", "lr": 1e-3}]}
        ],
        "checkpoint": {"every_seconds": None, "every_steps": 1000},
    }
    base.update(overrides)
    return RunConfig(**base)


def _cloud(cfg: RunConfig, root) -> dict[str, torch.Tensor]:
    """The initial collocation cloud this config would train on.

    ``build_trainer`` draws it before the first step, which is exactly the
    quantity in question, so nothing has to be trained to read it.
    """
    ctx = configure_runtime(cfg)
    run = Run.create(cfg, root, run_id=f"r{abs(hash(cfg.identity_hash())) % 10**9}")
    trainer = build_trainer(cfg, ctx, run)
    return {k: v.detach().clone() for k, v in trainer.state.points.items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# -- half one: optimisation must not move the cloud ---------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "stages": [
                    {
                        "name": "adam",
                        "steps": 1,
                        "optimizers": [{"name": "adam", "lr": 1e-3}],
                    },
                    {
                        "name": "lbfgs",
                        "steps": 1,
                        "optimizers": [{"name": "lbfgs", "lr": 1.0}],
                    },
                ]
            },
            id="append-an-lbfgs-stage",
        ),
        pytest.param(
            {
                "stages": [
                    {
                        "name": "adam",
                        "steps": 9,
                        "optimizers": [{"name": "adam", "lr": 0.5}],
                    }
                ]
            },
            id="different-steps-and-lr",
        ),
        pytest.param(
            {
                "nets": {
                    "u": {
                        "arch": "mlp",
                        "inputs": 2,
                        "outputs": 1,
                        "width": 16,
                        "depth": 3,
                    }
                }
            },
            id="different-architecture",
        ),
        pytest.param({"weighting": {"kind": "mean", "coefficients": {"ic": 10.0}}},
                     id="different-loss-weights"),
        pytest.param({"name": "renamed", "tags": {"arm": "treatment"}},
                     id="cosmetic-fields"),
    ],
)
def test_cloud_survives_everything_that_is_not_sampling(tmp_path, overrides):
    """The paired-comparison guarantee. Each id above is a real experimental arm."""
    reference = _cloud(_config(), tmp_path / "ref")
    other = _cloud(_config(**overrides), tmp_path / "other")
    assert _same(reference, other), (
        "changing how a run is optimised redrew its collocation cloud, so two "
        "arms of this comparison do not share the points they train on"
    )


# -- half two: sampling must still move it ------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"seed": 4}, id="seed"),
        pytest.param(
            {
                "sampling": {
                    "points": {
                        "interior": {
                            "region": "interior",
                            "n": 64,
                            "strategy": "pseudo",
                        },
                        "initial": {"region": "initial", "n": 8},
                    }
                }
            },
            id="point-count",
        ),
        pytest.param(
            {
                "sampling": {
                    "points": {
                        "interior": {
                            "region": "interior",
                            "n": 32,
                            "strategy": "hammersley",
                        },
                        "initial": {"region": "initial", "n": 8},
                    }
                }
            },
            id="sampling-strategy",
        ),
        pytest.param(
            {"problem": {"name": "burgers1d", "options": {"nu": 0.05}}},
            id="physical-constant",
        ),
    ],
)
def test_cloud_still_changes_with_the_sampling(tmp_path, overrides):
    """Without this, a trainer that handed every config one cloud would pass."""
    reference = _cloud(_config(), tmp_path / "ref")
    other = _cloud(_config(**overrides), tmp_path / "other")
    assert not _same(reference, other), (
        "a change to the geometry, the point counts, the strategy or the seed "
        "left the cloud untouched; the sampling identity is too narrow"
    )


def test_sampling_identity_excludes_the_optimisation_fields():
    """Guards the constant itself, so widening it is a deliberate act."""
    assert SAMPLING_IDENTITY == ("problem", "sampling", "dtype")
    cfg = _config()
    hashed = cfg.sampling_identity_hash()
    assert hashed == cfg.sampling_identity_hash(), "hash must be stable"
    assert hashed != cfg.identity_hash(), (
        "the sampling identity must be a strictly narrower key than the full "
        "config identity, or nothing has changed"
    )
