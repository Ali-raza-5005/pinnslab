"""The two evaluators, and the claim that they are interchangeable.

`SequentialEvaluator` is the oracle — it drives the ordinary `build_trainer`
path, so whatever it reports is by construction what a reproduction run
reports. `BatchedEvaluator` is the fast path. If the two disagree, the search
optimises something other than what the paper will later measure, and every
number the search produced is off the record it claims to be on.

So the load-bearing test here compares them on a *real* Burgers config with the
library's own registered residual terms, not a stand-in.
"""

from __future__ import annotations

import math

import pytest
import torch

from pinnslab.registry.config import RunConfig
from pinnslab.search.evaluate import (
    BatchedEvaluator,
    SequentialEvaluator,
    with_step_budget,
)
from pinnslab.search.space import SearchSpace

pytestmark = pytest.mark.unit


def base_config(**over) -> RunConfig:
    payload = {
        "name": "evaluable",
        "seed": 3,
        "dtype": "float64",
        "device": "cpu",
        "problem": {"name": "burgers1d", "options": {"nu": 0.3183098861837907}},
        "nets": {
            "u": {"arch": "mlp", "inputs": 2, "outputs": 1, "width": 8, "depth": 2}
        },
        "residuals": {
            "pde": {"kind": "burgers1d.pde", "points": ["interior", "initial"]},
            "ic": {"kind": "burgers1d.ic", "points": "initial"},
        },
        "sampling": {
            "points": {
                "interior": {"region": "interior", "n": 48},
                "initial": {"region": "initial", "n": 12},
            }
        },
        "weighting": {"kind": "mean"},
        "stages": [
            {"name": "adam", "steps": 20, "optimizers": [{"name": "adam", "lr": 1e-3}]}
        ],
        "logging": {"trace": {"every": 10}},
        "checkpoint": {"every_seconds": None, "every_steps": 1000},
    }
    payload.update(over)
    return RunConfig(**payload)


# -- the budget ----------------------------------------------------------------


def test_a_rung_rescales_the_stages_proportionally():
    """An Adam->L-BFGS schedule must keep its shape at every rung, or the rungs
    measure structurally different things and halving between them is
    meaningless."""
    cfg = base_config(
        stages=[
            {"name": "adam", "steps": 900, "optimizers": [{"name": "adam"}]},
            {"name": "lbfgs", "steps": 100, "optimizers": [{"name": "lbfgs"}]},
        ]
    )
    rung = with_step_budget(cfg, 100)

    assert [s.steps for s in rung.stages] == [90, 10]
    assert rung.total_steps == 100


def test_a_stage_that_rounds_to_nothing_is_dropped():
    cfg = base_config(
        stages=[
            {"name": "adam", "steps": 999, "optimizers": [{"name": "adam"}]},
            {"name": "lbfgs", "steps": 1, "optimizers": [{"name": "lbfgs"}]},
        ]
    )
    assert [s.name for s in with_step_budget(cfg, 10).stages] == ["adam"]


def test_the_full_budget_is_the_config_untouched():
    cfg = base_config()
    assert with_step_budget(cfg, cfg.total_steps) is cfg


# -- the sequential oracle -----------------------------------------------------


def test_the_sequential_evaluator_returns_the_runs_metric(tmp_path):
    cfg = base_config()
    scores = SequentialEvaluator(root=tmp_path)([cfg], cfg.total_steps)

    assert len(scores) == 1
    assert math.isfinite(scores[0])
    assert (tmp_path / f"{cfg.identity_hash()[:12]}_s3_n20" / "result.json").exists()


def test_a_candidate_that_cannot_build_scores_nan(tmp_path):
    """A configuration that cannot train is a search result, not a crash of the
    search (DESIGN.md §11: failures are data)."""
    broken = base_config(problem={"name": "no-such-benchmark"})
    assert math.isnan(SequentialEvaluator(root=tmp_path)([broken], 20)[0])


# -- the batched path ----------------------------------------------------------


def test_batched_and_sequential_agree_on_the_training_objective():
    """THE test of this module.

    The batched path reuses the config's registered residual terms unchanged —
    they work on ``(P, N, m)`` only because diffops indexes with ``...``. If
    that broke, or if the point groups were laid out differently between the
    two paths, this diverges immediately.

    Compared at zero training steps so the assertion is about the *objective*
    rather than about two optimisers agreeing over time: an equivalence in the
    loss is what makes the batched fitness comparable with a reproduction run,
    and `test_search_population.py` already pins that training itself matches.
    """
    configs = [base_config(seed=s) for s in (0, 1, 2)]
    batched = BatchedEvaluator()(configs, steps=0)

    expected = [_objective_the_slow_way(cfg) for cfg in configs]

    assert len(batched) == 3
    for got, want in zip(batched, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-9)


def _objective_the_slow_way(cfg: RunConfig) -> float:
    """The same loss through the single-run assembly, one candidate at a time."""
    from pinnslab.training.build import assemble
    from pinnslab.utils.device import configure_runtime

    ctx = configure_runtime(cfg)
    part = assemble(cfg, ctx)

    class State:
        def __init__(self):
            self.nets = part.nets
            self.extra_params = part.extra_params
            self.generator = torch.Generator().manual_seed(cfg.seed)
            self.dtype, self.device = ctx.dtype, ctx.device
            self.points: dict = {}
            self.scratch: dict = {}
            self.step = 0

    state = State()
    part.on_resample(state)
    residuals = part.residual_fn(state)
    # The config's own weighting object, not a re-implementation of it. This
    # used to concatenate every term and take one pooled mean of squares, with
    # a comment asserting that "*is* the mean weighting" — which is true only
    # when every term has the same point count. It never does: here `pde` sees
    # 60 points and `ic` sees 12, and on the shipped Burgers example the two
    # objectives differed by 6.3x with the boundary term weighted 26x too low.
    # So the test passed while pinning the bug, because both sides computed the
    # same wrong number. Calling the real object is what makes this an oracle.
    return float(part.weighting(residuals, state).detach())


def test_the_batched_path_trains_every_candidate():
    configs = [base_config(seed=s) for s in (0, 1, 2)]
    before = BatchedEvaluator()(configs, steps=0)
    after = BatchedEvaluator()(configs, steps=30)

    assert all(a < b for a, b in zip(after, before, strict=True)), (
        "30 Adam steps did not reduce the objective for every candidate"
    )


def test_candidates_are_scored_independently():
    """Different sampling seeds must give different fitness; a coupled batch
    would flatten them."""
    scores = BatchedEvaluator()([base_config(seed=s) for s in (0, 1, 2)], steps=0)
    assert len(set(scores)) == 3


def test_a_space_that_holds_the_point_budget_fixed_batches_cleanly():
    """The shape of paper 1's search: a fixed collocation budget (which
    DESIGN.md §8 requires for fairness anyway) with the other axes free.

    The axis varied here is a **loss weight**, one of DESIGN.md §6's four
    directions and one that genuinely batches: the per-term coefficient enters
    the population residual per candidate. This test used to vary
    `stages.0.optimizers.0.lr`, which does *not* batch — one Adam runs over the
    stacked population with one scalar lr, so every candidate was silently
    trained at candidate 0's learning rate while the archive recorded three
    different ones. That is now refused; see the test below.
    """
    space = SearchSpace(
        {"weighting.coefficients.ic": {
            "kind": "continuous", "low": 1.0, "high": 100.0, "log": True}}
    )
    # The coefficient must already be declared: `_set_path` refuses to create a
    # key the config does not have, so a search cannot introduce a field the
    # schema never saw. Declaring it at 1.0 is the neutral starting point.
    base = base_config(weighting={"kind": "mean", "coefficients": {"ic": 1.0}})
    configs = [space.apply(base, [u]) for u in (0.1, 0.5, 0.9)]

    scores = BatchedEvaluator()(configs, steps=5)

    assert len(scores) == 3
    assert all(math.isfinite(s) for s in scores)
    # Different weights on the IC term are different objectives, so the
    # candidates must not collapse onto one number — which is what would happen
    # if the coefficient were read off configs[0] and applied to everyone.
    assert len(set(scores)) == 3


def test_a_per_candidate_learning_rate_is_refused_rather_than_ignored():
    """One Adam, one scalar lr, P candidates.

    A space over `stages.0.optimizers.0.lr` reads plausibly and used to run
    without complaint, scoring every candidate at the *first* one's lr. The
    archive then held three distinct config hashes for one experiment, and the
    cache would serve those fitnesses back forever. Refusing is the only
    correct answer until the ensemble carries a per-member lr.
    """
    space = SearchSpace(
        {"stages.0.optimizers.0.lr": {
            "kind": "continuous", "low": 1e-4, "high": 1e-2, "log": True}}
    )
    configs = [space.apply(base_config(), [u]) for u in (0.1, 0.9)]

    with pytest.raises(ValueError, match="same optimizer"):
        BatchedEvaluator()(configs, steps=1)


def test_a_multi_stage_schedule_is_refused_rather_than_run_as_adam():
    """An Adam->L-BFGS config evaluated as Adam alone ranks candidates under a
    training procedure no reproduction run performs."""
    staged = base_config(stages=[
        {"name": "adam", "steps": 10, "optimizers": [{"name": "adam", "lr": 1e-3}]},
        {"name": "lbfgs", "steps": 5, "optimizers": [{"name": "lbfgs", "lr": 1.0}]},
    ])
    with pytest.raises(ValueError, match="several stages"):
        BatchedEvaluator()([staged], steps=1)


def test_a_varying_physical_constant_is_refused():
    """Residual terms are built once, from configs[0].problem. A space over nu
    would train the whole population on candidate 0's equation while recording
    each candidate's own value in its config."""
    configs = [
        base_config(problem={"name": "burgers1d", "options": {"nu": 0.3}}),
        base_config(problem={"name": "burgers1d", "options": {"nu": 0.4}}),
    ]
    with pytest.raises(ValueError, match="same problem"):
        BatchedEvaluator()(configs, steps=1)


def test_a_resampling_config_is_refused():
    """train_population draws the cloud once; a search over *resampling* that
    never resamples measures nothing."""
    resampling = base_config(stages=[{
        "name": "adam",
        "steps": 10,
        "resample_every": 5,
        "optimizers": [{"name": "adam", "lr": 1e-3}],
    }])
    with pytest.raises(ValueError, match="resample_every"):
        BatchedEvaluator()([resampling], steps=1)


def test_the_batched_objective_honours_the_declared_coefficients():
    """`weighting.coefficients` are part of the objective, so they must reach
    it. They did not: the batched path pooled every term into one mean and
    dropped the coefficients entirely."""
    plain = base_config()
    weighted = base_config(weighting={"kind": "mean", "coefficients": {"ic": 10.0}})

    scores = BatchedEvaluator()([plain, weighted], steps=0)

    assert scores[0] != scores[1], (
        "a 10x weight on the IC term did not change the objective"
    )


def test_a_candidates_fitness_does_not_depend_on_its_position_in_the_batch():
    """The cache is keyed on (config_hash, steps), so a config must score the
    same wherever it sits. It did not, once: `configure_runtime` reseeds the
    global RNG and `assemble` draws the initial weights from it, so building
    candidates back to back gave candidate k an init that depended on how many
    preceded it."""
    a, b = base_config(seed=0), base_config(seed=1)

    forwards = BatchedEvaluator()([a, b], steps=0)
    backwards = BatchedEvaluator()([b, a], steps=0)
    alone = BatchedEvaluator()([a], steps=0)

    assert forwards[0] == pytest.approx(backwards[1], rel=1e-12)
    assert forwards[0] == pytest.approx(alone[0], rel=1e-12)


# -- what the batched path refuses ---------------------------------------------


def test_a_mixed_point_budget_is_refused():
    """Batching needs one shape. DESIGN.md §8 wants an identical collocation
    count across compared methods anyway, so this is a fairness rule as much as
    a technical one — and it says so rather than padding silently."""
    configs = [
        base_config(),
        base_config(sampling={"points": {
            "interior": {"region": "interior", "n": 96},
            "initial": {"region": "initial", "n": 12},
        }}),
    ]
    with pytest.raises(ValueError, match="one point-group layout"):
        BatchedEvaluator()(configs, steps=1)


def test_a_non_mean_weighting_is_refused():
    """Training under a different objective than the config declares would make
    the search's fitness incomparable with its own reproduction runs."""
    configs = [base_config(weighting={"kind": "sum"})]
    with pytest.raises(ValueError, match="mean weighting"):
        BatchedEvaluator()(configs, steps=1)


def test_differing_residual_structure_is_refused():
    configs = [
        base_config(),
        base_config(residuals={"pde": {"kind": "burgers1d.pde", "points": "interior"}}),
    ]
    with pytest.raises(ValueError, match="same residual terms"):
        BatchedEvaluator()(configs, steps=1)


def test_an_empty_population_is_not_an_error():
    assert BatchedEvaluator()([], steps=10) == []
