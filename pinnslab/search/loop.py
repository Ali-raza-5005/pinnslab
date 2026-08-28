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

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnslab.registry.config import RunConfig
from pinnslab.registry.provenance import collect_provenance
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
    #: Wall-clock for this generation, in seconds — the ladder, the evaluator
    #: and the cache lookups. DESIGN.md §11 wants generation time reported
    #: alongside the search's total, because a metaheuristic that wins on
    #: fitness while costing 100x the wall clock has not won.
    seconds: float = 0.0


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
        self._cached_this_rung: set[str] = set()
        self._restore()
        if not self.state.provenance:
            # CLAUDE.md rule 7 applies to a search as much as to a run: it
            # produces numbers a paper quotes. Captured once, on the first
            # session, so a resumed search keeps the identity it started with.
            self.state.provenance = collect_provenance(
                seed=spec.seed, dtype=base.dtype
            ).model_dump(mode="json")

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
                "%d inner steps, %.1fs; %.1fs total)",
                report.generation + 1,
                self.spec.generations,
                report.best,
                report.median,
                report.evaluated,
                report.cached,
                report.inner_steps,
                report.seconds,
                self.state.total_seconds,
            )
        return self.state

    def step(self) -> GenerationReport:
        """One generation: ask, run the fidelity ladder, tell, checkpoint."""
        generation = self.state.generation
        started = time.perf_counter()
        candidates = self.algorithm.ask()
        fitness, evaluated, cached, spent = self._ladder(candidates, generation)
        seconds = time.perf_counter() - started

        self.algorithm.tell(candidates, fitness)
        self.state.generation = generation + 1
        # Accumulated into the state *before* the checkpoint, so a resumed
        # search continues the running total instead of restarting the clock —
        # which on Kaggle, where a search spans sessions, would make the
        # reported search cost the cost of the last session only.
        self.state.total_seconds += seconds
        self.state.total_inner_steps += spent
        self._checkpoint()

        return GenerationReport(
            generation=generation,
            best=float(np.min(fitness)),
            median=float(np.median(fitness)),
            evaluated=evaluated,
            cached=cached,
            inner_steps=spent,
            seconds=seconds,
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
            rung_started = time.perf_counter()
            scores, hits, ran = self._evaluate(configs, steps, generation)
            rung_seconds = time.perf_counter() - rung_started
            evaluated += ran
            cached += hits
            spent += ran * steps
            # Attributed across the candidates that were actually trained; a
            # cache hit cost nothing. See Evaluation.seconds for why this is an
            # attribution rather than a measurement.
            each = rung_seconds / ran if ran else 0.0

            for slot, score, config in zip(alive, scores, configs, strict=True):
                fitness[slot] = score
                config_hash = config.identity_hash()
                self.state.archive.append(
                    Evaluation(
                        generation=generation,
                        vector=[float(v) for v in candidates[slot]],
                        config_hash=config_hash,
                        fitness=float(score),
                        steps=steps,
                        cached=config_hash in self._cached_this_rung,
                        seconds=0.0 if config_hash in self._cached_this_rung else each,
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
        # Which of these came off the cache, so the archive can mark them and
        # charge them no time. Recorded here rather than returned because the
        # ladder needs it per candidate, not as a count.
        self._cached_this_rung = {
            h for i, h in enumerate(hashes) if i not in set(pending)
        }
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

        # Indexed, not filtered: every slot is populated by now, and a
        # comprehension that dropped a None would silently shift every fitness
        # onto the wrong candidate.
        resolved = [float(scores[i]) for i in range(len(configs))]  # type: ignore[arg-type]
        penalty = self._penalty(resolved)
        return (
            [penalty if not np.isfinite(v) else v for v in resolved],
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
        """What a diverged candidate scores: strictly worse than every real one.

        A configured value when there is one. Otherwise a value derived from
        this batch, because a fixed large constant is its own bug — an
        arbitrary 1e9 among values of 1e-3 makes every finite candidate look
        identical to a difference-based optimiser like DE, which is exactly the
        population it needs to distinguish.

        The derivation must survive a **negative** score, and the obvious
        formula does not. This returned ``10 * max(finite)``, which for a
        maximised fitness — oriented to ``[-0.9, -0.8]`` by :meth:`_orient`, so
        that lower is better — gives ``-8.0``: an order of magnitude *better*
        than any candidate that actually trained, so DE would drive the whole
        population toward configurations that diverge. It also tied with the
        best when every score was 0.0.

        So: step past the worst finite score by a margin taken from the
        batch's own spread, falling back to the worst score's magnitude and
        then to 1.0. That is strictly worse than everything finite for any
        sign, and stays on the batch's scale.
        """
        if self.spec.fitness.penalty is not None:
            return float(self.spec.fitness.penalty)
        finite = [s for s in scores if np.isfinite(s)]
        if not finite:
            return 1.0
        worst, best = max(finite), min(finite)
        margin = (worst - best) or abs(worst) or 1.0
        return float(worst + 10.0 * margin)


__all__ = ["Evaluator", "GenerationReport", "Search"]
