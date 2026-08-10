"""The search driver: generations, fidelity rungs, cache, checkpoint.

Ties together the four pieces DESIGN.md §6 asks for. The loop owns the RNG (so
there is one owner of the thing whose loss breaks reproducibility invisibly),
the cache lookup (so a re-proposed configuration is free), and the fidelity
schedule (so cheap runs filter and only survivors get the long budget).

The evaluator is injected. That is what makes the same loop run:

* a **sequential** evaluation, one candidate through the ordinary
  ``build_trainer`` path — slow, but it is the oracle every faster path is
  checked against, and it is what a search over architectures (which does not
  batch) must use;
* a **batched** evaluation through :mod:`pinnslab.search.population`.

A search is resumable at generation granularity, not mid-generation. An
interrupted generation is re-evaluated on resume, and the cache makes that
nearly free for any candidate that already finished — which is the cheapest
correct answer, since a half-evaluated population has no meaning to an
ask/tell algorithm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnslab.registry.config import RunConfig
from pinnslab.search.algorithms import Algorithm, build_algorithm
from pinnslab.search.cache import CandidateCache
from pinnslab.search.space import SearchSpace
from pinnslab.search.spec import SearchSpec
from pinnslab.search.state import (
    Evaluation,
    SearchState,
    capture_rng,
    restore_rng,
)
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

#: ``(configs, steps) -> one fitness per config``. Lower is always better by
#: the time the loop sees it; :class:`SearchSpec.fitness` direction is applied
#: once, here, so no algorithm carries a sign.
Evaluator = Callable[[list[RunConfig], int], list[float]]


@dataclass
class GenerationReport:
    generation: int
    best: float
    median: float
    evaluated: int
    cached: int
    inner_steps: int


class Search:
    """One search over one base config.

    ``root`` is where the outer-loop checkpoint and the candidate cache live.
    Both are append-only or atomically replaced, so a killed session costs at
    most the generation in flight.
    """

    def __init__(
        self,
        spec: SearchSpec,
        base: RunConfig,
        evaluator: Evaluator,
        *,
        root: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.base = base
        self.evaluator = evaluator
        self.root = Path(root) if root else None

        self.space = SearchSpace(spec.space)
        # Fail before the first candidate trains: a mistyped path would
        # otherwise produce a full set of plausible results from a search that
        # optimised nothing.
        self.space.validate_against(base)

        self.rng = np.random.default_rng(spec.seed)
        self.algorithm: Algorithm = build_algorithm(
            spec.algorithm,
            self.space.dim,
            spec.pop_size,
            self.rng,
            **spec.algorithm_options,
        )
        self.cache = CandidateCache(self.root)
        self.state = SearchState(spec_hash=spec.identity_hash())
        self._restore()

    # -- resume ----------------------------------------------------------------

    def _restore(self) -> None:
        if self.root is None:
            return
        stored = SearchState.load(self.root)
        if stored is None:
            return
        if stored.spec_hash != self.state.spec_hash:
            raise ValueError(
                f"the search state in {self.root} was written by a different "
                f"search ({stored.spec_hash} on disk, {self.state.spec_hash} "
                "now). Resuming under a changed space or algorithm would "
                "silently mix two experiments."
            )
        self.state = stored
        self.algorithm.load_state(stored.algorithm_state)
        if stored.rng_state:
            restore_rng(self.rng, stored.rng_state)
        log.info(
            "resumed search at generation %d with %d archived evaluation(s)",
            self.state.generation,
            len(self.state.archive),
        )

    def _checkpoint(self) -> None:
        if self.root is None:
            return
        self.state.algorithm_state = self.algorithm.state()
        self.state.rng_state = capture_rng(self.rng)
        self.state.save(self.root)

    # -- the loop --------------------------------------------------------------

    def run(self) -> SearchState:
        """Work through the remaining generations."""
        while self.state.generation < self.spec.generations:
            report = self.step()
            log.info(
                "gen %d/%d: best %.4g median %.4g (%d evaluated, %d cached, "
                "%d inner steps)",
                report.generation + 1,
                self.spec.generations,
                report.best,
                report.median,
                report.evaluated,
                report.cached,
                report.inner_steps,
            )
        return self.state

    def step(self) -> GenerationReport:
        """One generation: ask, run the fidelity ladder, tell, checkpoint."""
        generation = self.state.generation
        candidates = self.algorithm.ask()
        fitness, evaluated, cached, spent = self._ladder(candidates, generation)

        self.algorithm.tell(candidates, fitness)
        self.state.generation = generation + 1
        self._checkpoint()

        return GenerationReport(
            generation=generation,
            best=float(np.min(fitness)),
            median=float(np.median(fitness)),
            evaluated=evaluated,
            cached=cached,
            inner_steps=spent,
        )

    def _ladder(
        self, candidates: np.ndarray, generation: int
    ) -> tuple[np.ndarray, int, int, int]:
        """Successive halving over the fidelity rungs.

        Every candidate is evaluated at the first rung; the best ``keep``
        fraction continue. A candidate cut at a rung keeps the fitness it had
        there, and the archive records which rung it reached — comparing a
        cheap number against an expensive one as though they were the same is
        the mistake the rung field exists to make visible.
        """
        alive = list(range(len(candidates)))
        fitness = np.full(len(candidates), np.inf)
        evaluated = cached = spent = 0

        for index, steps in enumerate(self.spec.budget.rungs):
            configs = [self.space.apply(self.base, candidates[i]) for i in alive]
            scores, hits, ran = self._evaluate(configs, steps, generation)
            evaluated += ran
            cached += hits
            spent += ran * steps

            for slot, score, config in zip(alive, scores, configs, strict=True):
                fitness[slot] = score
                self.state.archive.append(
                    Evaluation(
                        generation=generation,
                        vector=[float(v) for v in candidates[slot]],
                        config_hash=config.identity_hash(),
                        fitness=float(score),
                        steps=steps,
                    )
                )

            if index + 1 < len(self.spec.budget.rungs):
                keep = self.spec.budget.survivors(len(candidates), index)
                alive = [i for i in sorted(alive, key=lambda s: fitness[s])[:keep]]

        return fitness, evaluated, cached, spent

    def _evaluate(
        self, configs: list[RunConfig], steps: int, generation: int
    ) -> tuple[list[float], int, int]:
        """Cache lookup, then one call to the evaluator for whatever is left."""
        hashes = [c.identity_hash() for c in configs]
        scores: list[float | None] = [self.cache.get(h, steps) for h in hashes]

        pending = [i for i, s in enumerate(scores) if s is None]
        if pending:
            raw = self.evaluator([configs[i] for i in pending], steps)
            if len(raw) != len(pending):
                raise ValueError(
                    f"the evaluator returned {len(raw)} fitness values for "
                    f"{len(pending)} configs; it must return one per config, "
                    "in order"
                )
            for slot, value in zip(pending, raw, strict=True):
                score = self._orient(value)
                scores[slot] = score
                self.cache.put(
                    hashes[slot],
                    steps,
                    score,
                    generation=generation,
                    seed=self.spec.seed,
                )

        resolved = [float(s) for s in scores if s is not None]
        penalty = self._penalty(resolved)
        return (
            [penalty if not np.isfinite(s) else s for s in resolved],
            len(configs) - len(pending),
            len(pending),
        )

    def _orient(self, value: float) -> float:
        """Make lower always better, once, so no algorithm carries a sign."""
        value = float(value)
        if not np.isfinite(value):
            return np.inf
        return -value if self.spec.fitness.direction == "max" else value

    def _penalty(self, scores: list[float]) -> float:
        """What a diverged candidate scores.

        A configured value when there is one; otherwise strictly worse than the
        worst finite score in this batch. Not a fixed large constant: an
        arbitrary 1e9 among values of 1e-3 makes every finite candidate look
        identical to a difference-based optimiser like DE, which is exactly the
        population it needs to distinguish.
        """
        if self.spec.fitness.penalty is not None:
            return float(self.spec.fitness.penalty)
        finite = [s for s in scores if np.isfinite(s)]
        return 10.0 * max(finite) if finite else 1.0


__all__ = ["Evaluator", "GenerationReport", "Search"]
