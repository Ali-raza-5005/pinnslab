"""The search algorithms, on objectives whose optima are known.

Everything else in the suite drives an algorithm through :class:`Search` with a
stub evaluator, which proves the *plumbing* — ask, tell, cache, resume — and
says nothing about whether the optimiser optimises. That is a real gap for this
repo specifically, because DESIGN.md §8 makes "does the metaheuristic beat
random search at matched budget?" the question a paper lives or dies on, and an
implementation bug in DE would answer it *no* for a reason that has nothing to
do with PINNs.

So these tests run the algorithms directly on standard box-constrained
benchmarks, mapped into the unit cube the algorithm layer actually sees:

* **Sphere** — convex, separable. If DE cannot beat random search here, DE is
  broken. This is the smoke test with teeth.
* **Rosenbrock** — a curved, ill-conditioned valley. Separable per-axis search
  does badly; a difference-vector method should not.
* **Rastrigin** — highly multimodal. Not asserted to be *solved*, only to be
  handled without the population collapsing, because collapse is the failure
  mode a wrong mutation or selection rule produces.

Every assertion is a comparison against random search at an **identical budget**
(same population, same generations, same seed), never against an absolute
threshold — an absolute number would encode this machine and this torch version,
and the comparison is what the science rests on anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinnslab.search.algorithms import build_algorithm

pytestmark = pytest.mark.unit

DIM = 6
POP = 16
GENERATIONS = 80


# -- the benchmarks ------------------------------------------------------------
#
# Each takes a unit-cube vector and returns a scalar to be MINIMISED, which is
# the orientation `Search._orient` guarantees before an algorithm ever sees a
# fitness.


def sphere(unit: np.ndarray) -> float:
    """Optimum 0 at the cube's centre. Mapped to [-5.12, 5.12]^d."""
    x = _rescale(unit, -5.12, 5.12)
    return float(np.sum(x**2))


def rosenbrock(unit: np.ndarray) -> float:
    """Optimum 0 at x = 1. Mapped to [-2.048, 2.048]^d."""
    x = _rescale(unit, -2.048, 2.048)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(unit: np.ndarray) -> float:
    """Optimum 0 at x = 0, with a lattice of local minima around it."""
    x = _rescale(unit, -5.12, 5.12)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def _rescale(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + np.asarray(unit, dtype=float) * (high - low)


BENCHMARKS = {"sphere": sphere, "rosenbrock": rosenbrock, "rastrigin": rastrigin}


# -- the driver ----------------------------------------------------------------


def optimise(
    algorithm: str,
    objective,
    *,
    seed: int = 0,
    generations: int = GENERATIONS,
    pop: int = POP,
    **options,
) -> tuple[float, np.ndarray]:
    """Run one algorithm to a fixed budget. Returns ``(best fitness, history)``.

    Deliberately *not* through :class:`~pinnslab.search.loop.Search`: no cache,
    no fidelity ladder, no config round trip. What is under test is the
    ask/tell contract and nothing else, so a failure here cannot be anything
    but the algorithm.
    """
    rng = np.random.default_rng(seed)
    alg = build_algorithm(algorithm, DIM, pop, rng, **options)

    best = np.inf
    history = []
    for _ in range(generations):
        candidates = alg.ask()
        fitness = np.array([objective(c) for c in candidates])
        alg.tell(candidates, fitness)
        best = min(best, float(fitness.min()))
        history.append(best)
    return best, np.array(history)


# -- box constraints -----------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["random", "de"])
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_every_proposal_stays_inside_the_unit_cube(algorithm, name):
    """The box constraint is the algorithm layer's whole contract with
    :mod:`pinnslab.search.space`, which clips on decode as a second line of
    defence. If proposals leave the cube, that clip silently piles candidates
    onto the boundary and the search reports diversity it does not have.
    """
    rng = np.random.default_rng(1)
    alg = build_algorithm(algorithm, DIM, POP, rng)
    for _ in range(10):
        candidates = alg.ask()
        assert candidates.shape == (POP, DIM)
        assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)
        alg.tell(candidates, np.array([BENCHMARKS[name](c) for c in candidates]))


# -- does it actually optimise -------------------------------------------------


def test_de_beats_random_search_on_a_convex_bowl():
    """The floor. Sphere is convex and separable; a working DE must clear
    random search at an identical budget by a wide margin, and if it does not
    the answer to DESIGN.md §8's mandatory comparison is "no" for reasons that
    have nothing to do with PINNs.

    Median over five seeds, not one, and for the reason §8 gives: a single seed
    of this comparison spans 4x to 40x, so a one-seed threshold is a statement
    about which seed was picked. The measured medians on this implementation
    are 4x at 20 generations, 15x at 40, 790x at 80 and 2e4x at 120 — the
    margin below is deliberately far under the 80-generation figure, because
    what it must catch is DE not searching at all, not DE being slow.
    """
    de = float(np.median([optimise("de", sphere, seed=s)[0] for s in range(5)]))
    rs = float(np.median([optimise("random", sphere, seed=s)[0] for s in range(5)]))

    assert de < rs / 10.0, (
        f"DE reached {de:.4g} and random search {rs:.4g} at the same budget; "
        "DE is not searching"
    )


def test_de_beats_random_search_in_a_curved_valley():
    """Rosenbrock is the case a per-axis search cannot do well on, so it
    separates a real difference-vector method from one whose mutation has
    degenerated into noise."""
    de, _ = optimise("de", rosenbrock, seed=0)
    rs, _ = optimise("random", rosenbrock, seed=0)

    assert de < rs, f"DE {de:.4g} did not beat random search {rs:.4g}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_de_beats_random_search_across_seeds(seed):
    """One seed is an anecdote. DESIGN.md §8 asks for >=5 seeds of any claim,
    and the claim here is about the optimiser rather than about a PINN."""
    de, _ = optimise("de", sphere, seed=seed)
    rs, _ = optimise("random", sphere, seed=seed)
    assert de < rs


def test_de_makes_monotone_progress():
    """Best-so-far can never worsen — DE's replacement is greedy and one-to-one,
    so a generation that made everything worse must simply keep the incumbents.
    A `tell` that overwrote unconditionally would show up exactly here."""
    _, history = optimise("de", sphere, seed=0)
    assert np.all(np.diff(history) <= 0.0)


def test_de_gets_close_to_a_known_optimum():
    """Sphere's optimum is 0 at the cube's centre. Not a tight tolerance — the
    budget is 640 evaluations in 6 dimensions — but far enough below the
    ~50 a random cloud reaches that only a converging search passes."""
    best, _ = optimise("de", sphere, seed=0, generations=120)
    assert best < 1e-3, f"DE stalled at {best:.4g} on a convex bowl"


def test_the_multimodal_case_does_not_collapse():
    """Rastrigin is not asserted to be *solved* — no 640-evaluation budget
    solves it — only to be handled. The failure this catches is population
    collapse: a wrong mutation or selection rule drives every candidate onto one
    point, which still produces a plausible-looking fitness curve.
    """
    rng = np.random.default_rng(0)
    alg = build_algorithm("de", DIM, POP, rng)
    for _ in range(GENERATIONS):
        candidates = alg.ask()
        alg.tell(candidates, np.array([rastrigin(c) for c in candidates]))

    spread = float(np.mean(np.std(alg.population, axis=0)))
    assert spread > 1e-3, (
        f"the population collapsed to a mean per-axis spread of {spread:.2e}; "
        "DE has stopped exploring"
    )


# -- exploration / exploitation mechanics --------------------------------------


def test_the_differential_weight_changes_the_search():
    """F is the exploration knob. If it were being ignored, every setting would
    trace the same path — which is exactly what a mutation that dropped the
    scaling would look like from the outside."""
    timid, _ = optimise("de", rastrigin, seed=0, generations=25,
                        differential_weight=0.1)
    bold, _ = optimise("de", rastrigin, seed=0, generations=25,
                       differential_weight=0.9)
    assert timid != bold


def test_the_crossover_probability_changes_the_search():
    """CR is the other knob, and `ask` forces one axis from the mutant
    regardless — so CR=0 must still move (that forced axis) while differing
    from CR=1 (every axis)."""
    low, _ = optimise("de", rosenbrock, seed=0, generations=25,
                      crossover_probability=0.05)
    high, _ = optimise("de", rosenbrock, seed=0, generations=25,
                       crossover_probability=0.95)
    assert low != high


def test_donors_are_distinct_and_exclude_the_target():
    """DE/rand/1 needs a target plus three *distinct* donors. A donor equal to
    the target collapses the difference vector toward zero, so the mutation
    quietly stops mutating."""
    rng = np.random.default_rng(0)
    alg = build_algorithm("de", DIM, POP, rng)
    alg.tell(alg.ask(), rng.random(POP))

    for target in range(POP):
        for _ in range(50):
            donors = alg._donors(target)
            assert len(set(donors)) == 3
            assert target not in donors


# -- reproducibility -----------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["random", "de"])
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_a_seeded_run_is_reproducible(algorithm, name):
    """Same seed, same trajectory — the property every other reproducibility
    claim in the library rests on, asserted on the whole history rather than on
    the final number, because two different searches can land on one value.

    A short budget on purpose: a seed leak shows up in the first few
    generations or not at all, and this runs six times.
    """
    first = optimise(algorithm, BENCHMARKS[name], seed=7, generations=15)[1]
    second = optimise(algorithm, BENCHMARKS[name], seed=7, generations=15)[1]
    assert np.array_equal(first, second)


@pytest.mark.parametrize("algorithm", ["random", "de"])
def test_different_seeds_explore_differently(algorithm):
    """The other half: a "reproducible" algorithm that ignored its seed would
    pass the test above and be useless for the multi-seed statistics of
    DESIGN.md §8."""
    assert not np.array_equal(
        optimise(algorithm, sphere, seed=1, generations=15)[1],
        optimise(algorithm, sphere, seed=2, generations=15)[1],
    )


def test_random_search_does_not_learn():
    """It is the control, so it has to be the honest zero: `tell` must not
    steer it. A random search that quietly biased toward good regions would
    make the mandatory §8 baseline weaker than it claims, and every "we beat
    random search" number would be inflated."""
    rng_a = np.random.default_rng(5)
    told = build_algorithm("random", DIM, POP, rng_a)
    first = told.ask()
    told.tell(first, np.zeros(POP))
    after_telling = told.ask()

    rng_b = np.random.default_rng(5)
    untold = build_algorithm("random", DIM, POP, rng_b)
    untold.ask()
    after_silence = untold.ask()

    assert np.array_equal(after_telling, after_silence)


# -- resume --------------------------------------------------------------------


def test_de_resumes_onto_the_same_trajectory():
    """A search spans Kaggle sessions. `state`/`load_state` carry the
    population and its fitness; the RNG is restored by the loop. Restore one
    without the other and the resumed search proposes a different sequence
    while looking perfectly healthy."""
    rng = np.random.default_rng(3)
    original = build_algorithm("de", DIM, POP, rng)
    for _ in range(5):
        candidates = original.ask()
        original.tell(candidates, np.array([sphere(c) for c in candidates]))

    saved = original.state()
    saved_rng = rng.bit_generator.state

    resumed_rng = np.random.default_rng(0)
    resumed_rng.bit_generator.state = saved_rng
    resumed = build_algorithm("de", DIM, POP, resumed_rng)
    resumed.load_state(saved)

    assert np.array_equal(original.ask(), resumed.ask())
