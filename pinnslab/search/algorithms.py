"""The optimisers that propose candidates, one registered class each.

Every algorithm sees only a population of unit-cube vectors and their fitness.
It never learns what an axis means — :mod:`pinnslab.search.space` owns that —
which is what lets the same machinery run the four research directions of
DESIGN.md §6 by changing a config field.

Two are shipped, deliberately:

* ``random`` — DESIGN.md §8 makes random search at matched budget a
  **mandatory** baseline. If the metaheuristic does not beat it at equal
  budget there is no result, and that is a thing to discover in P1 rather than
  from Reviewer 2. It is the first algorithm here for that reason, not as a
  placeholder.
* ``de`` — differential evolution, the "serious optimizer" baseline §8 asks
  for. Framing the contribution as "metaheuristic search improves PINN
  sampling" rather than "GWO is the right algorithm" is what lets the
  algorithm be swapped without changing the claim, and it sidesteps the real
  community backlash against reskinned nature-inspired metaheuristics.

PSO, GA and GWO are **not** here yet. CLAUDE.md rule 2: a method is born in a
paper repo and promoted when a second paper needs it. Each is one file
implementing :meth:`Algorithm.ask` / :meth:`Algorithm.tell` and one
``@register_search`` line, with no edits to anything else.
"""

from __future__ import annotations

import numpy as np

from pinnslab.components import Registry

#: name -> algorithm class. Separate from ``components.py``'s registries only
#: because this one is imported by the search layer alone.
SEARCH_ALGORITHMS: Registry[type[Algorithm]] = Registry("search algorithm")
register_search = SEARCH_ALGORITHMS.register


class Algorithm:
    """Ask/tell over unit-cube vectors.

    ``ask`` returns ``(n, dim)`` proposals; ``tell`` receives the fitness of
    exactly those proposals, already oriented so that **lower is better**. The
    loop normalises direction once so no algorithm has to carry a sign.

    ``state``/``load_state`` carry everything a resumed search needs *except*
    the RNG, which the loop checkpoints centrally — DESIGN.md §6 calls out
    metaheuristic RNG state as one of the three things not to skip, and having
    one owner for it is how that stays true.
    """

    def __init__(self, dim: int, pop_size: int, rng: np.random.Generator, **options):
        self.dim = dim
        self.pop_size = pop_size
        self.rng = rng
        if options:
            raise TypeError(
                f"{type(self).__name__} got unexpected options {sorted(options)}"
            )

    def ask(self) -> np.ndarray:
        raise NotImplementedError

    def tell(self, candidates: np.ndarray, fitness: np.ndarray) -> None:
        raise NotImplementedError

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


@register_search("random")
class RandomSearch(Algorithm):
    """Independent uniform samples. The baseline everything is measured against.

    Deliberately memoryless: ``tell`` does nothing. A "random search" that
    quietly biased toward good regions would be a weak optimiser masquerading
    as the control, and the whole point of this baseline is that it is the
    honest zero.
    """

    def ask(self) -> np.ndarray:
        return self.rng.random((self.pop_size, self.dim))

    def tell(self, candidates: np.ndarray, fitness: np.ndarray) -> None:
        return None


@register_search("de")
class DifferentialEvolution(Algorithm):
    """Classic DE/rand/1/bin with a greedy one-to-one replacement.

    Chosen over CMA-ES as the first serious optimiser because it is
    box-constrained by construction, has two interpretable knobs, and does not
    need a covariance estimate to be meaningful at the population sizes a PINN
    search can afford (16-50, where CMA-ES is still warming up).

    The one-to-one replacement matters for the cache: a trial that loses is
    discarded entirely, so the population only ever contains configurations
    that have actually been evaluated.
    """

    def __init__(
        self,
        dim: int,
        pop_size: int,
        rng: np.random.Generator,
        *,
        differential_weight: float = 0.8,
        crossover_probability: float = 0.9,
    ):
        super().__init__(dim, pop_size, rng)
        if pop_size < 4:
            raise ValueError(
                f"DE/rand/1 needs a target plus three distinct donors, so "
                f"pop_size must be >= 4, got {pop_size}"
            )
        self.f = float(differential_weight)
        self.cr = float(crossover_probability)
        self.population: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self._trials: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        if self.population is None:
            # Generation 0 *is* a random sample, which is why a DE run and a
            # random run at the same seed start from the same population and
            # the comparison between them is paired.
            self._trials = self.rng.random((self.pop_size, self.dim))
            return self._trials

        trials = np.empty_like(self.population)
        for i in range(self.pop_size):
            a, b, c = self._donors(i)
            mutant = np.clip(
                self.population[a]
                + self.f * (self.population[b] - self.population[c]),
                0.0,
                1.0,
            )
            crossover = self.rng.random(self.dim) < self.cr
            # At least one axis always comes from the mutant, or a trial could
            # be an exact copy of its target and the generation would be wasted.
            crossover[self.rng.integers(self.dim)] = True
            trials[i] = np.where(crossover, mutant, self.population[i])
        self._trials = trials
        return trials

    def _donors(self, target: int) -> tuple[int, int, int]:
        pool = [i for i in range(self.pop_size) if i != target]
        return tuple(self.rng.choice(pool, size=3, replace=False))  # type: ignore[return-value]

    def tell(self, candidates: np.ndarray, fitness: np.ndarray) -> None:
        if self.population is None:
            self.population, self.fitness = candidates.copy(), fitness.copy()
            return
        assert self.fitness is not None
        improved = fitness < self.fitness
        self.population[improved] = candidates[improved]
        self.fitness[improved] = fitness[improved]

    def state(self) -> dict:
        return {
            "population": None if self.population is None else self.population.tolist(),
            "fitness": None if self.fitness is None else self.fitness.tolist(),
        }

    def load_state(self, state: dict) -> None:
        population, fitness = state.get("population"), state.get("fitness")
        self.population = None if population is None else np.array(population)
        self.fitness = None if fitness is None else np.array(fitness)


def build_algorithm(
    name: str, dim: int, pop_size: int, rng: np.random.Generator, **options
) -> Algorithm:
    return SEARCH_ALGORITHMS.get(name)(dim, pop_size, rng, **options)


__all__ = [
    "SEARCH_ALGORITHMS",
    "Algorithm",
    "DifferentialEvolution",
    "RandomSearch",
    "build_algorithm",
    "register_search",
]
