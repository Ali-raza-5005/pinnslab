"""The shipped examples, tested as the artefacts a new user copies.

An example that stopped working is worse than no example: it is the first thing
someone runs, and its failure reads as "this repo is broken". So the configs are
loaded and *trained* here (at a smoke-length budget — the real ones take ~30s a
seed), the run matrix is parsed, and the search space is checked against the
config it claims to vary.

The RAD sampler in ``examples/rad_sampler.py`` gets a property test rather than
a smoke test, because "it ran" would pass for a sampler that quietly returned
uniform points — which is precisely the bug an adaptive-sampling paper cannot
afford to ship.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pinnslab.components import SAMPLERS
from pinnslab.registry.config import load_config
from pinnslab.registry.run import Run
from pinnslab.search.space import SearchSpace
from pinnslab.search.spec import load_search_spec
from pinnslab.training.build import build_trainer
from pinnslab.training.queue import config_for, load_matrix
from pinnslab.utils.device import configure_runtime
from pinnslab.utils.plugins import load_plugin

pytestmark = pytest.mark.unit

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
UNIFORM = EXAMPLES / "configs" / "burgers_uniform.yaml"
RAD = EXAMPLES / "configs" / "burgers_rad.yaml"


@pytest.fixture(scope="module", autouse=True)
def registered():
    """Import the example sampler exactly as ``--register`` does."""
    return load_plugin(str(EXAMPLES / "rad_sampler.py"))


def smoke(path: Path, steps: int = 12):
    """The example config, cut to a length a test suite can afford."""
    cfg = load_config(path)
    return cfg.model_copy(
        update={
            "stages": [cfg.stages[0].model_copy(update={"steps": steps})],
            "device": "cpu",
        }
    )


def train(cfg, root, run_id):
    ctx = configure_runtime(cfg)
    return build_trainer(cfg, ctx, Run.create(cfg, root, run_id=run_id))


# -- the configs ---------------------------------------------------------------


def test_both_arms_differ_only_in_how_they_sample():
    """The comparison's whole validity: same points, same cadence, same
    optimizer schedule, same seeds. If anything else drifts apart, the example
    teaches a confounded experiment (DESIGN.md §8)."""
    uniform, rad = load_config(UNIFORM).to_dict(), load_config(RAD).to_dict()

    differing = {k for k in uniform if uniform[k] != rad[k]}
    # name and tags are excluded from the config hash, so they are labels rather
    # than conditions.
    assert differing == {"name", "tags", "sampling"}

    u_points, r_points = uniform["sampling"]["points"], rad["sampling"]["points"]
    assert {k: v["n"] for k, v in u_points.items()} == {
        k: v["n"] for k, v in r_points.items()
    }
    assert [s["resample_every"] for s in uniform["stages"]] == [
        s["resample_every"] for s in rad["stages"]
    ]


def test_the_two_arms_are_different_conditions():
    assert load_config(UNIFORM).identity_hash() != load_config(RAD).identity_hash()


def test_the_run_matrix_names_configs_that_exist_and_validate():
    """A matrix is validated before the first cell trains; on Kaggle a typo
    found two hours in has cost two hours."""
    cells = load_matrix(EXAMPLES / "run_matrix.csv")

    assert len(cells) == 10
    for cell in cells:
        assert cell.config.is_file()
        config_for(cell)  # raises on an invalid config or an unknown component

    per_arm = {}
    for cell in cells:
        per_arm.setdefault(cell.config.name, []).append(cell.seed)
    assert all(len(seeds) >= 5 for seeds in per_arm.values()), (
        "DESIGN.md §8 wants >=5 seeds for a headline claim, and the example "
        "should model that rather than the shortcut"
    )


def test_the_search_space_names_paths_the_base_config_declares():
    """A search whose paths do not exist optimises nothing while producing a
    full set of plausible results, so this is checked before it can run."""
    spec = load_search_spec(EXAMPLES / "search.yaml")
    space = SearchSpace(spec.space)

    space.validate_against(load_config(RAD))  # raises if a path is wrong
    assert not spec.batched, "a search on rel_l2 cannot use the batched path"


def test_the_search_space_produces_configs_that_still_validate():
    spec = load_search_spec(EXAMPLES / "search.yaml")
    space = SearchSpace(spec.space)
    base = load_config(RAD)

    for unit in (0.0, 0.5, 1.0):
        candidate = space.apply(base, [unit] * space.dim)
        assert candidate.identity_hash()
        assert candidate.sampling.points["interior"].strategy == "rad"


# -- the sampler ---------------------------------------------------------------


def test_the_example_sampler_is_registered_under_the_name_the_config_uses():
    assert "rad" in SAMPLERS
    assert load_config(RAD).sampling.points["interior"].strategy == "rad"


def test_the_example_config_trains(results_root):
    """End to end on the real file: config -> problem -> sampler -> loop."""
    trainer = train(smoke(RAD), results_root, "rad")
    row = trainer.fit()

    assert row.status.value == "completed"
    assert row.final_metrics["rel_l2"] > 0.0
    assert trainer.state.points["interior"].shape == (1000, 2)


def test_the_first_rad_draw_is_the_baseline_draw(results_root):
    """Generation 0 has no residual to score, so RAD must make exactly the draw
    the control arm makes: it is a modification of the baseline, not a different
    procedure, which is what makes the k=0 mechanism ablation mean anything.

    Compared stream-for-stream rather than run-for-run, because two *runs* never
    share an RNG stream: the trainer derives it from ``(seed, "trainer",
    config_hash)``, so different conditions are decorrelated by construction —
    which also means the two arms of this example are unpaired, and need their
    five seeds.
    """
    from pinnslab.benchmarks.problem import build_problem
    from pinnslab.geometry.samplers import build_sampler
    from pinnslab.utils.seeding import make_generator

    rad_cfg, uniform_cfg = load_config(RAD), load_config(UNIFORM)
    problem = build_problem(rad_cfg.problem)
    rad = build_sampler(rad_cfg.sampling.points["interior"], problem)
    plain = build_sampler(uniform_cfg.sampling.points["interior"], problem)

    class Stream:
        def __init__(self):
            self.generator = make_generator(11)
            self.dtype, self.device = torch.float64, torch.device("cpu")

    assert torch.equal(rad(Stream(), None), plain(Stream(), None))


def test_rad_puts_points_where_the_residual_is(results_root):
    """The property that makes it *adaptive*. A sampler that quietly returned
    uniform points would pass every other test in this file."""
    cfg = smoke(RAD, steps=40)
    trainer = train(cfg, results_root, "adaptive")
    trainer.fit()

    sampler = trainer.on_resample.samplers["interior"]
    state = trainer.state
    uniform = sampler._uniform(state, 1000)
    adaptive = sampler(state, current=state.points["interior"])

    score = lambda points: sampler.term(state, points).detach().abs().mean()  # noqa: E731
    assert score(adaptive) > score(uniform), (
        "RAD's cloud should carry more residual mass than a uniform draw of the "
        "same size; it is not concentrating anything"
    )
    assert adaptive.shape == uniform.shape


def test_the_sampler_reports_its_generation_count(results_root):
    """It is checkpointed, so a resumed run continues RAD's schedule instead of
    restarting it (see tests/unit/test_resampling.py)."""
    trainer = train(smoke(RAD, steps=30), results_root, "gen")
    trainer.fit()

    sampler = trainer.on_resample.samplers["interior"]
    assert sampler.state_dict() == {"generations": sampler.generations}
    assert sampler.generations >= 1
