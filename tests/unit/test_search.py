"""The search layer: space, spec, algorithms, cache, outer-loop state, driver.

The population evaluator has its own file — its claims are numerical and need a
different kind of test. What is tested here is the machinery around it, and the
three things DESIGN.md §6 says not to skip: the outer-loop checkpoint including
the metaheuristic's RNG, the candidate cache, and multi-fidelity.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinnslab.registry.config import RunConfig
from pinnslab.search.algorithms import (
    SEARCH_ALGORITHMS,
    DifferentialEvolution,
    build_algorithm,
)
from pinnslab.search.cache import CandidateCache
from pinnslab.search.loop import Search
from pinnslab.search.space import (
    Categorical,
    Continuous,
    Integer,
    SearchSpace,
    build_domain,
)
from pinnslab.search.spec import FidelitySchedule, SearchSpec, load_search_spec
from pinnslab.search.state import SearchState, capture_rng, restore_rng

pytestmark = pytest.mark.unit


def base_config(**over) -> RunConfig:
    payload = {
        "name": "searchable",
        "seed": 0,
        "device": "cpu",
        "problem": {"name": "burgers1d"},
        "nets": {"u": {"arch": "mlp", "inputs": 2, "outputs": 1, "width": 16}},
        "residuals": {"pde": {"kind": "burgers1d.pde", "points": "interior"}},
        "sampling": {"points": {"interior": {"region": "interior", "n": 100}}},
        "stages": [
            {"name": "adam", "steps": 10, "optimizers": [{"name": "adam", "lr": 1e-3}]}
        ],
    }
    payload.update(over)
    return RunConfig(**payload)


SPACE = {
    "sampling.points.interior.n": {"kind": "integer", "low": 50, "high": 2000},
    "stages.0.optimizers.0.lr": {
        "kind": "continuous", "low": 1e-5, "high": 1e-2, "log": True,
    },
}


def spec(**over) -> SearchSpec:
    payload = {"space": SPACE, "pop_size": 6, "generations": 2}
    payload.update(over)
    return SearchSpec(**payload)


# -- domains -------------------------------------------------------------------


def test_a_continuous_domain_spans_its_interval():
    d = Continuous(low=-1.0, high=3.0)
    assert d.decode(0.0) == -1.0
    assert d.decode(1.0) == 3.0
    assert d.decode(0.5) == 1.0


def test_a_log_domain_searches_decades_evenly():
    """Uniform on [1e-5, 1e-2] spends 99.9% of proposals in the top decade,
    which is not a search of a learning rate."""
    d = Continuous(low=1e-5, high=1e-2, log=True)
    assert d.decode(0.0) == pytest.approx(1e-5)
    assert d.decode(1.0) == pytest.approx(1e-2)
    assert d.decode(1 / 3) == pytest.approx(1e-4)


def test_a_log_domain_needs_a_positive_lower_bound():
    with pytest.raises(ValueError, match="low > 0"):
        Continuous(low=0.0, high=1.0, log=True)


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="low < high"):
        Continuous(low=2.0, high=1.0)


def test_every_integer_gets_an_equal_slice():
    """Off-by-one here biases the search toward the endpoints for the whole
    campaign, invisibly."""
    d = Integer(low=1, high=4)
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for unit in np.linspace(0, 1, 10_001):
        counts[d.decode(unit)] += 1
    assert max(counts.values()) - min(counts.values()) <= 2


def test_integers_stay_inside_their_range():
    d = Integer(low=3, high=9)
    assert d.decode(0.0) == 3
    assert d.decode(1.0) == 9
    assert d.decode(1.5) == 9  # a proposal outside the box is the normal case
    assert d.decode(-0.4) == 3


def test_categorical_choices_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        Categorical(choices=("a", "b", "a"))


def test_encode_inverts_decode():
    """Needed to seed generation 0 with a hand-tuned incumbent instead of
    starting the search behind it."""
    domains = [
        Continuous(low=1e-5, high=1e-2, log=True),
        Continuous(low=-2.0, high=6.0),
        Integer(low=8, high=256),
        Integer(low=8, high=256, log=True),
        Categorical(choices=("pseudo", "hammersley", "sobol")),
    ]
    for d in domains:
        for unit in (0.05, 0.3, 0.5, 0.77, 0.95):
            value = d.decode(unit)
            assert d.decode(d.encode(value)) == value, f"{d.kind} at {unit}"


def test_an_unknown_domain_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown domain kind"):
        build_domain({"kind": "gaussian", "low": 0, "high": 1})


# -- the space -----------------------------------------------------------------


def test_applying_a_candidate_produces_a_validated_config():
    """A search must not be able to propose something a human could not have
    written by hand: the result goes back through pydantic."""
    space = SearchSpace(SPACE)
    cfg = space.apply(base_config(), [1.0, 0.0])

    assert isinstance(cfg, RunConfig)
    assert cfg.sampling.points["interior"].n == 2000
    assert cfg.stages[0].optimizers[0].lr == pytest.approx(1e-5)


def test_different_candidates_get_different_config_hashes():
    """The cache, the results row and the figure all join on this."""
    space = SearchSpace(SPACE)
    a = space.apply(base_config(), [0.1, 0.2]).identity_hash()
    b = space.apply(base_config(), [0.9, 0.2]).identity_hash()
    assert a != b


def test_two_vectors_that_decode_alike_share_a_hash():
    """Why the cache is keyed on the config, not the vector: an integer axis
    collapses whole regions of the unit cube onto one experiment."""
    axis = "sampling.points.interior.n"
    space = SearchSpace({axis: SPACE[axis]})
    a = space.apply(base_config(), [0.5000]).identity_hash()
    b = space.apply(base_config(), [0.5001]).identity_hash()
    assert a == b


def test_a_path_that_does_not_exist_fails_before_anything_trains():
    """A search that silently optimises nothing is the worst failure mode: it
    produces a complete set of plausible results."""
    space = SearchSpace(
        {"sampling.points.interior.nn": {"kind": "integer", "low": 1, "high": 2}}
    )
    with pytest.raises(KeyError, match="does not exist"):
        space.validate_against(base_config())


def test_a_bad_list_index_names_the_segment_that_failed():
    space = SearchSpace({"stages.7.steps": {"kind": "integer", "low": 1, "high": 2}})
    with pytest.raises(KeyError, match="stages.7"):
        space.validate_against(base_config())


def test_an_empty_space_is_refused():
    with pytest.raises(ValueError, match="nothing to search"):
        SearchSpace({})


def test_a_wrong_length_vector_is_refused():
    with pytest.raises(ValueError, match="length 2"):
        SearchSpace(SPACE).decode([0.5])


# -- the spec ------------------------------------------------------------------


def test_rungs_must_increase():
    with pytest.raises(ValueError, match="strictly increase"):
        FidelitySchedule(rungs=(1000, 500))


def test_the_search_budget_is_computable_before_it_runs():
    """DESIGN.md §8: compute parity including search cost. A number you can
    only get by running the search is no use for planning a fair comparison."""
    schedule = FidelitySchedule(rungs=(100, 1000), keep=0.5)
    # 8 candidates x 100 steps, then 4 survivors x the extra 900.
    assert schedule.cost(8) == 8 * 100 + 4 * 900
    whole = spec(budget=schedule, pop_size=8, generations=3)
    assert whole.total_inner_steps == 3 * 4400


def test_at_least_one_candidate_always_survives_a_rung():
    assert FidelitySchedule(rungs=(10, 20, 30), keep=0.1).survivors(4, 1) == 1


def test_the_spec_hash_ignores_seed_but_not_the_space():
    """Several seeds of one search are one condition (DESIGN.md §4)."""
    assert spec(seed=1).identity_hash() == spec(seed=2).identity_hash()
    assert spec(pop_size=8).identity_hash() != spec(pop_size=6).identity_hash()


def test_a_search_spec_round_trips_through_yaml(tmp_path):
    """A search is a hyperparameter of the research, so it lives in a validated,
    hashed file rather than in a script (CLAUDE.md rule 4) — which needs a
    loader, the mirror of `load_config`."""
    import yaml

    original = spec(pop_size=8)
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(original.model_dump(mode="json")), encoding="utf-8")

    assert load_search_spec(path).identity_hash() == original.identity_hash()


def test_a_search_yaml_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_search_spec(path)


def test_a_search_yaml_with_an_unknown_field_is_refused(tmp_path):
    """`extra="forbid"`, so a typo'd key is a load error rather than a search
    that quietly ignored half of what it was told."""
    path = tmp_path / "search.yaml"
    path.write_text(
        "space:\n  seed: {kind: integer, low: 1, high: 4}\npopulation_size: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="population_size"):
        load_search_spec(path)


# -- algorithms ----------------------------------------------------------------


def test_random_and_de_are_both_registered():
    """Random search is a mandatory baseline, not a placeholder (DESIGN.md §8)."""
    assert "random" in SEARCH_ALGORITHMS
    assert "de" in SEARCH_ALGORITHMS


def test_random_search_is_memoryless():
    """A 'random search' that drifted toward good regions would be a weak
    optimiser masquerading as the control."""
    rng = np.random.default_rng(0)
    algorithm = build_algorithm("random", 3, 5, rng)
    first = algorithm.ask()
    algorithm.tell(first, np.zeros(5))

    reference = build_algorithm("random", 3, 5, np.random.default_rng(0))
    reference.ask()
    assert np.array_equal(algorithm.ask(), reference.ask())


def test_de_generation_zero_is_a_random_sample():
    """So a DE run and a random run at one seed start from the same population
    and the comparison between them is paired."""
    de = build_algorithm("de", 4, 8, np.random.default_rng(3))
    rs = build_algorithm("random", 4, 8, np.random.default_rng(3))
    assert np.array_equal(de.ask(), rs.ask())


def test_de_keeps_only_improvements():
    de = DifferentialEvolution(2, 6, np.random.default_rng(0))
    first = de.ask()
    de.tell(first, np.arange(6, dtype=float))

    trials = de.ask()
    better = np.full(6, -1.0)
    better[0] = 100.0  # candidate 0's trial is worse and must be rejected
    de.tell(trials, better)

    assert np.array_equal(de.population[0], first[0])
    assert np.array_equal(de.population[1], trials[1])


def test_de_needs_enough_candidates_for_distinct_donors():
    with pytest.raises(ValueError, match="pop_size must be >= 4"):
        DifferentialEvolution(2, 3, np.random.default_rng(0))


def test_de_always_takes_at_least_one_axis_from_the_mutant():
    """Otherwise a trial can be an exact copy of its target and the generation
    is wasted evaluating what is already in the population."""
    de = DifferentialEvolution(5, 8, np.random.default_rng(1))
    de.tell(de.ask(), np.arange(8, dtype=float))
    de.f, de.cr = 0.8, 0.0  # crossover off: only the forced axis can change

    trials = de.ask()
    for index in range(8):
        assert not np.array_equal(trials[index], de.population[index])


# -- the cache -----------------------------------------------------------------


def test_the_cache_keys_on_fidelity_as_well_as_config():
    """A fitness at 200 steps is not the same number as one at 20000, and
    conflating them would promote candidates on a cheap evaluation."""
    cache = CandidateCache()
    cache.put("abc", 200, 1e-2)

    assert cache.get("abc", 200) == 1e-2
    assert cache.get("abc", 20_000) is None


def test_the_cache_survives_the_process(tmp_path):
    """A search spans many Kaggle sessions; a cache that did not persist would
    re-pay for every candidate the previous session already evaluated."""
    CandidateCache(tmp_path).put("abc", 200, 1e-2, generation=3)

    reloaded = CandidateCache(tmp_path)
    assert reloaded.get("abc", 200) == 1e-2
    assert len(reloaded) == 1


def test_the_cache_skips_a_torn_line(tmp_path):
    """Append-only and fsynced, so a killed session can only tear the tail."""
    cache = CandidateCache(tmp_path)
    cache.put("abc", 200, 1e-2)
    with (tmp_path / "candidates.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"config_hash": "de')

    reloaded = CandidateCache(tmp_path)
    assert reloaded.get("abc", 200) == 1e-2


def test_the_cache_reports_its_hit_rate():
    cache = CandidateCache()
    cache.put("a", 10, 1.0)
    cache.get("a", 10)
    cache.get("b", 10)
    assert cache.hit_rate == 0.5


# -- outer-loop state ----------------------------------------------------------


def test_the_rng_state_round_trips():
    """DESIGN.md §6's "don't skip" #1, and the one that breaks reproducibility
    invisibly: a resumed search that lost its RNG proposes a different sequence
    of candidates and produces a plausible result nobody can reproduce."""
    rng = np.random.default_rng(7)
    rng.random(5)
    captured = capture_rng(rng)
    expected = rng.random(5)

    restored = np.random.default_rng(0)
    restore_rng(restored, captured)
    assert np.array_equal(restored.random(5), expected)


def test_the_state_round_trips_through_disk(tmp_path):
    state = SearchState(generation=3, spec_hash="abc")
    state.rng_state = capture_rng(np.random.default_rng(1))
    state.save(tmp_path)

    loaded = SearchState.load(tmp_path)
    assert loaded.generation == 3
    assert loaded.spec_hash == "abc"
    assert loaded.rng_state == state.rng_state


def test_no_state_on_disk_is_not_an_error(tmp_path):
    assert SearchState.load(tmp_path) is None


def test_the_incumbent_is_chosen_within_one_fidelity():
    """A 200-step fitness of 1e-3 does not beat a 20000-step fitness of 2e-3 —
    it just cost less. Comparing across rungs crowns candidates that got lucky
    early and were never tested properly."""
    from pinnslab.search.state import Evaluation

    state = SearchState(
        archive=[
            Evaluation(0, [0.1], "cheap", fitness=1e-6, steps=200),
            Evaluation(0, [0.2], "proper", fitness=2e-3, steps=20_000),
            Evaluation(0, [0.3], "worse", fitness=5e-3, steps=20_000),
        ]
    )
    assert state.best().config_hash == "proper"


# -- the driver ----------------------------------------------------------------


def counting_evaluator(seen: list[int]):
    """Fitness = the collocation count, so the optimum is a known corner."""

    def evaluate(configs, steps):
        seen.append(len(configs))
        return [float(c.sampling.points["interior"].n) for c in configs]

    return evaluate


def test_a_search_runs_every_generation():
    seen: list[int] = []
    search = Search(spec(generations=3), base_config(), counting_evaluator(seen))
    state = search.run()

    assert state.generation == 3
    assert len(seen) == 3
    assert state.best() is not None


def test_the_search_improves_on_a_monotone_objective():
    """DE against a trivially monotone fitness must beat its own generation 0."""
    search = Search(
        spec(algorithm="de", pop_size=8, generations=8), base_config(),
        counting_evaluator([]),
    )
    state = search.run()

    first = min(e.fitness for e in state.archive if e.generation == 0)
    last = min(e.fitness for e in state.archive if e.generation == 7)
    assert last < first


def test_a_mistyped_path_fails_before_the_first_candidate():
    bad = spec(
        space={
            "sampling.points.interior.nope": {"kind": "integer", "low": 1, "high": 9}
        }
    )
    with pytest.raises(KeyError, match="does not exist"):
        Search(bad, base_config(), counting_evaluator([]))


def test_repeated_candidates_are_served_from_the_cache():
    """DE's greedy replacement keeps surviving vectors unchanged for many
    generations; paying for them twice is pure waste."""
    calls: list[int] = []

    def evaluate(configs, steps):
        calls.append(len(configs))
        return [1.0] * len(configs)

    # A one-axis integer space far smaller than the population guarantees
    # duplicate configurations within a single generation.
    tiny = spec(
        space={"sampling.points.interior.n": {"kind": "integer", "low": 1, "high": 2}},
        pop_size=12,
        generations=2,
    )
    search = Search(tiny, base_config(), evaluate)
    search.run()

    assert search.cache.hits > 0
    assert sum(calls) < 24, "every candidate was evaluated despite duplicates"


def test_multi_fidelity_narrows_the_population():
    """Cheap runs for everyone, the long budget only for survivors."""
    batches: list[tuple[int, int]] = []

    def evaluate(configs, steps):
        batches.append((len(configs), steps))
        return [float(c.sampling.points["interior"].n) for c in configs]

    search = Search(
        spec(
            budget=FidelitySchedule(rungs=(10, 100, 1000), keep=0.5),
            pop_size=8,
            generations=1,
        ),
        base_config(),
        evaluate,
    )
    search.run()

    assert [steps for _, steps in batches] == [10, 100, 1000]
    counts = [n for n, _ in batches]
    assert counts == [8, 4, 2], f"successive halving did not narrow: {counts}"


def test_a_diverged_candidate_scores_worse_than_every_finite_one():
    """And not by a fixed 1e9: an arbitrary constant among values of 1e-3 makes
    every finite candidate look identical to a difference-based optimiser."""
    def evaluate(configs, steps):
        return [float("nan") if i == 0 else 1e-3 * (i + 1) for i in range(len(configs))]

    search = Search(spec(generations=1, pop_size=4), base_config(), evaluate)
    search.run()

    scores = sorted(e.fitness for e in search.state.archive)
    assert scores[-1] > scores[-2]
    assert np.isfinite(scores[-1]), "a penalty of inf tells DE nothing"


def test_an_evaluator_returning_the_wrong_count_is_an_error():
    search = Search(spec(generations=1), base_config(), lambda configs, steps: [1.0])
    with pytest.raises(ValueError, match="one per config"):
        search.run()


# -- resume --------------------------------------------------------------------


def test_a_resumed_search_continues_where_it_stopped(tmp_path):
    interrupted = Search(
        spec(algorithm="de", generations=4), base_config(),
        counting_evaluator([]), root=tmp_path,
    )
    interrupted.step()
    interrupted.step()

    resumed = Search(
        spec(algorithm="de", generations=4), base_config(),
        counting_evaluator([]), root=tmp_path,
    )
    assert resumed.state.generation == 2
    resumed.run()
    assert resumed.state.generation == 4


def test_a_resumed_search_proposes_what_an_uninterrupted_one_would(tmp_path):
    """The property the RNG checkpoint exists for. Without it the resumed
    search explores a different sequence and no rerun reproduces the result."""
    uninterrupted = Search(
        spec(algorithm="de", generations=4), base_config(),
        counting_evaluator([]), root=tmp_path / "whole",
    )
    uninterrupted.run()

    part = Search(
        spec(algorithm="de", generations=4), base_config(),
        counting_evaluator([]), root=tmp_path / "split",
    )
    part.step()
    part.step()
    rest = Search(
        spec(algorithm="de", generations=4), base_config(),
        counting_evaluator([]), root=tmp_path / "split",
    )
    rest.run()

    assert [e.vector for e in rest.state.archive] == [
        e.vector for e in uninterrupted.state.archive
    ]


def test_resuming_under_a_changed_spec_is_refused(tmp_path):
    """Silently mixing two experiments is worse than losing one."""
    first = Search(
        spec(generations=2), base_config(), counting_evaluator([]), root=tmp_path
    )
    first.step()

    with pytest.raises(ValueError, match="different search"):
        Search(
            spec(generations=2, pop_size=8), base_config(),
            counting_evaluator([]), root=tmp_path,
        )
