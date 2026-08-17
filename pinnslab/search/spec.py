"""The validated description of a search (DESIGN.md §6).

A search is a hyperparameter of the research, so CLAUDE.md rule 4 applies to it
exactly as it does to a run: no number lives in a script. This is the YAML that
a paper's ``search.yaml`` validates into, and it hashes for provenance the same
way a :class:`RunConfig` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from pinnslab.registry.hashing import config_hash
from pinnslab.registry.schema import Spec


class FidelitySchedule(Spec):
    """Successive halving: cheap runs for everyone, long runs for survivors.

    DESIGN.md §6 puts this in from day one rather than bolting it on, because
    retrofitting it changes what a "generation" costs and therefore invalidates
    every compute-parity number already measured.

    ``rungs`` are inner training steps. Every candidate is evaluated at
    ``rungs[0]``; the best ``keep`` fraction continue to ``rungs[1]``, and so
    on. A candidate eliminated at a rung keeps its last measured fitness, and
    the rung it reached is recorded — comparing a 200-step fitness against a
    20000-step one as though they were the same number is the mistake this
    structure exists to make visible.
    """

    rungs: tuple[int, ...] = Field(default=(1000,), min_length=1)
    keep: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _increasing(self) -> FidelitySchedule:
        if any(b <= a for a, b in zip(self.rungs, self.rungs[1:], strict=False)):
            raise ValueError(
                f"rungs must strictly increase, got {self.rungs}; a later rung "
                "that is not longer is not a higher fidelity"
            )
        if any(r <= 0 for r in self.rungs):
            raise ValueError(f"every rung needs > 0 steps, got {self.rungs}")
        return self

    @property
    def max_steps(self) -> int:
        return self.rungs[-1]

    def survivors(self, population: int, rung: int) -> int:
        """How many candidates continue past ``rung``. Never below one."""
        return max(1, int(population * self.keep ** (rung + 1)))

    def cost(self, population: int) -> int:
        """Total inner steps one generation spends, for compute parity.

        The number DESIGN.md §8 demands be reported alongside any comparison:
        a search that wins while burning 3000x the compute has not won, and
        this is what makes that budget computable *before* the sweep runs.
        """
        total, alive, previous = 0, population, 0
        for index, steps in enumerate(self.rungs):
            total += alive * (steps - previous)
            previous = steps
            alive = self.survivors(population, index)
        return total


class FitnessSpec(Spec):
    """What the search is optimising, and what counts as a failed candidate."""

    #: A metric key produced by the run's ``eval_fn`` or its residuals.
    metric: str = "rel_l2"
    direction: Literal["min", "max"] = "min"
    #: What a diverged candidate scores. ``None`` means "worst in generation",
    #: which keeps the population's scale sane; a fixed number is only right
    #: when the metric has a known ceiling.
    penalty: float | None = None

    def is_better(self, value: float, than: float) -> bool:
        return value < than if self.direction == "min" else value > than


class SearchSpec(Spec):
    """One search: a space, an algorithm, a budget, and a fitness.

    ``space`` is a mapping of **config path -> domain**, e.g.::

        space:
          sampling.points.interior.n: {kind: integer, low: 500, high: 8000}
          stages.0.optimizers.0.lr:
            {kind: continuous, low: 1e-5, high: 1e-2, log: true}
    """

    name: str = "search"
    seed: int = Field(default=0, ge=0)

    space: dict[str, dict[str, Any]] = Field(min_length=1)
    #: A registered algorithm. ``random`` is not a placeholder: DESIGN.md §8
    #: makes random search at matched budget a mandatory baseline, and
    #: discovering that the metaheuristic does not beat it is a P1 result, not
    #: a Reviewer 2 result.
    algorithm: str = "random"
    algorithm_options: dict[str, float | int | bool | str] = Field(
        default_factory=dict
    )
    pop_size: int = Field(default=16, gt=1)
    generations: int = Field(default=10, gt=0)

    budget: FidelitySchedule = Field(default_factory=FidelitySchedule)
    fitness: FitnessSpec = Field(default_factory=FitnessSpec)

    #: Evaluate the population as one batched graph rather than one run at a
    #: time. See :mod:`pinnslab.search.population` for what that requires of
    #: the space — chiefly that every candidate's network has the same shape.
    batched: bool = True

    @property
    def total_inner_steps(self) -> int:
        """Whole-search compute, in inner training steps."""
        return self.generations * self.budget.cost(self.pop_size)

    def identity(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        # seed is excluded for the same reason RunConfig excludes it: several
        # seeds of one search are one condition (DESIGN.md §4).
        payload.pop("seed", None)
        payload.pop("name", None)
        return payload

    def identity_hash(self) -> str:
        return config_hash(self.identity())


def load_search_spec(path: str | Path) -> SearchSpec:
    """``search.yaml`` -> validated :class:`SearchSpec`.

    The mirror of :func:`pinnslab.registry.config.load_config`, and for the same
    reason: a search is a hyperparameter of the research, so it lives in a file
    that is validated and hashed rather than in a script.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping, got {type(raw).__name__}"
        )
    return SearchSpec(**raw)


__all__ = ["FidelitySchedule", "FitnessSpec", "SearchSpec", "load_search_spec"]
