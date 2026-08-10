"""The candidate cache — config hash -> fitness (DESIGN.md §6, "don't skip" #2).

Metaheuristics re-evaluate duplicates constantly: DE's greedy replacement keeps
a surviving vector unchanged for many generations, and any integer or
categorical axis collapses whole regions of the unit cube onto the *same*
configuration. Two different vectors that decode to one config are one
experiment, and paying for it twice is pure waste.

Keyed by ``RunConfig.identity_hash()`` rather than by the vector, which is what
makes that true: identity is the condition, not the coordinates that produced
it (DESIGN.md §4).

**Fidelity is part of the key.** A fitness measured at 200 inner steps is not
the same number as one measured at 20000, and a cache that conflated them would
quietly promote candidates on the strength of a cheap evaluation. The key is
therefore ``(config_hash, steps)``.

On disk as append-only JSONL, for the same reason ``results/`` is: a search
spans many Kaggle sessions, and the file being rewritten is the file a killed
session corrupts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

CACHE_FILE = "candidates.jsonl"


@dataclass(frozen=True)
class CacheEntry:
    config_hash: str
    steps: int
    fitness: float
    generation: int
    seed: int


class CandidateCache:
    """An append-only ``(config_hash, steps) -> fitness`` store.

    Deliberately not an LRU or a bounded cache: the whole history of a search
    is small (one short line per candidate) and is itself a result — it is the
    record of what was tried, which a reviewer asking "how much did the search
    cost?" needs.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) / CACHE_FILE if path else None
        self._entries: dict[tuple[str, int], CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        skipped = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = CacheEntry(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue
            self._entries[(entry.config_hash, entry.steps)] = entry
        log.info(
            "candidate cache: %d entr(ies) from %s%s",
            len(self._entries),
            self.path,
            f" ({skipped} unparseable line(s) skipped)" if skipped else "",
        )

    def get(self, config_hash: str, steps: int) -> float | None:
        entry = self._entries.get((config_hash, steps))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry.fitness

    def put(
        self,
        config_hash: str,
        steps: int,
        fitness: float,
        *,
        generation: int = 0,
        seed: int = 0,
    ) -> None:
        entry = CacheEntry(config_hash, steps, float(fitness), generation, seed)
        self._entries[(config_hash, steps)] = entry
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        looked_up = self.hits + self.misses
        return 0.0 if looked_up == 0 else self.hits / looked_up

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())


__all__ = ["CACHE_FILE", "CacheEntry", "CandidateCache"]
