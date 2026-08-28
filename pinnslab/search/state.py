"""Outer-loop checkpointing (DESIGN.md §6, "don't skip" #1).

A search spans many Kaggle sessions, so the generation loop needs the same
survive-a-kill property the training loop already has. What must round-trip:

* the population and its fitness (via the algorithm's own ``state``),
* the generation counter,
* the archive — every candidate ever evaluated, which is the search's result,
* **the metaheuristic's RNG state.**

The last one is the one that gets skipped, and it is the one that breaks
reproducibility invisibly. Restore weights and generation but not the RNG and
the resumed search proposes a *different* sequence of candidates from the one
an uninterrupted run would — producing a perfectly plausible result that no
rerun will reproduce. It is checkpointed here rather than by each algorithm so
there is exactly one owner (see :class:`~pinnslab.search.algorithms.Algorithm`).

JSON rather than ``torch.save``: this state is small, and being able to read a
dead search's archive with ``jq`` on a phone is worth more than the microsecond.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

STATE_FILE = "search_state.json"


@dataclass
class Evaluation:
    """One candidate, evaluated once, at one fidelity."""

    generation: int
    vector: list[float]
    config_hash: str
    fitness: float
    steps: int
    cached: bool = False
    #: Wall-clock attributed to this candidate, in seconds. DESIGN.md §11 makes
    #: timing a first-class experimental result rather than metadata, because
    #: compute parity *including search cost* is a core reviewer defence.
    #:
    #: An **attribution, not a measurement**, and the difference matters: the
    #: batched evaluator trains the whole population simultaneously, so there is
    #: no per-candidate time to measure. This is the evaluator call's wall clock
    #: divided by the number of candidates it actually evaluated. A cache hit
    #: costs nothing and records 0.0, which is why the search's total is the sum
    #: of the generation times and not of these.
    seconds: float = 0.0


@dataclass
class SearchState:
    """Everything needed to resume a search exactly where it stopped."""

    generation: int = 0
    spec_hash: str = ""
    algorithm_state: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)
    archive: list[Evaluation] = field(default_factory=list)
    #: Wall-clock the search has spent across **every** session, in seconds.
    #: Accumulated rather than measured from the current process, because a
    #: search spans many Kaggle sessions and "how long did the search take" is
    #: a number the paper reports (DESIGN.md §11, ``total_search_time``).
    total_seconds: float = 0.0
    #: Inner training steps actually spent, summed over generations. The
    #: measured counterpart to ``SearchSpec.total_inner_steps``, which is a
    #: bound: the cache makes the real number smaller, and a reviewer asking
    #: what the search cost should be given the one that happened.
    total_inner_steps: int = 0
    #: Rule 7 provenance for the search itself, captured on the first write.
    #: A search produces numbers that go in a paper, so it records the version,
    #: SHA, hardware and precision that produced them exactly as a run does.
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- the incumbent ---------------------------------------------------------

    def best(self) -> Evaluation | None:
        """The best candidate seen, at the highest fidelity it was seen at.

        Compared within a fidelity, never across one: a 200-step fitness of
        1e-3 does not beat a 20000-step fitness of 2e-3, it just cost less to
        obtain. Picking across rungs would systematically crown candidates that
        got lucky early and were never tested properly.
        """
        if not self.archive:
            return None
        top_rung = max(e.steps for e in self.archive)
        return min(
            (e for e in self.archive if e.steps == top_rung), key=lambda e: e.fitness
        )

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "spec_hash": self.spec_hash,
            "algorithm_state": self.algorithm_state,
            "rng_state": self.rng_state,
            "archive": [e.__dict__ for e in self.archive],
            "total_seconds": self.total_seconds,
            "total_inner_steps": self.total_inner_steps,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchState:
        # Every field added since the first version is read with a default, so
        # a state file written by an older tag still resumes (the same
        # forward-compatibility rule ResultRow follows).
        return cls(
            generation=payload["generation"],
            spec_hash=payload.get("spec_hash", ""),
            algorithm_state=payload.get("algorithm_state", {}),
            rng_state=payload.get("rng_state", {}),
            archive=[Evaluation(**e) for e in payload.get("archive", [])],
            total_seconds=float(payload.get("total_seconds", 0.0)),
            total_inner_steps=int(payload.get("total_inner_steps", 0)),
            provenance=payload.get("provenance", {}),
        )

    def save(self, directory: str | Path) -> Path:
        """Atomically, because the process may die during the write."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STATE_FILE
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> SearchState | None:
        path = Path(directory) / STATE_FILE
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError) as exc:
            # The atomic write above means this should be unreachable; if it
            # happens, restarting the search is far better than resuming from
            # a half-understood state and reporting the result as reproducible.
            raise ValueError(
                f"{path} is unreadable ({exc}). The write is atomic, so this "
                "means external corruption; delete it to restart the search."
            ) from None


def capture_rng(rng: np.random.Generator) -> dict[str, Any]:
    """numpy's bit-generator state, as JSON-native types."""
    return json.loads(json.dumps(rng.bit_generator.state, default=_jsonable))


def restore_rng(rng: np.random.Generator, state: dict[str, Any]) -> None:
    """Put a generator back exactly where :func:`capture_rng` found it.

    ``state["state"]["key"]`` comes back from JSON as a list and must be an
    ndarray of uint32, or numpy raises — the round trip is the whole point, so
    it is converted here rather than left for the caller to discover.
    """
    payload = dict(state)
    inner = dict(payload.get("state", {}))
    if "key" in inner:
        inner["key"] = np.array(inner["key"], dtype=np.uint32)
    payload["state"] = inner
    rng.bit_generator.state = payload


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__} into search state")


__all__ = [
    "STATE_FILE",
    "Evaluation",
    "SearchState",
    "capture_rng",
    "restore_rng",
]
