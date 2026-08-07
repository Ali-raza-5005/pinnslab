"""The results schema — what every run is obliged to record.

CLAUDE.md rule 7 is enforced structurally here: the provenance fields are
*required* on :class:`ResultRow`, so a row that omits them cannot be constructed.

DESIGN.md §11: a diverged run is data, not garbage. Failures get a row with
``status=diverged`` and a reason, because failure rate is a reported metric.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

#: How a non-finite float is spelled on disk. JSON has no ``NaN``/``Infinity``
#: (RFC 8259) but a diverged run's final loss is exactly that, so the two must
#: be reconciled somewhere. See :func:`json_safe`.
NONFINITE_TAGS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

#: Version of the :class:`ResultRow` contract. Bump when a field is **renamed,
#: removed, or changes meaning** — adding one with a default is backward
#: compatible and needs no bump. Rows on disk are permanent (CLAUDE.md rule 6),
#: so this is the only handle a future reader has for telling eras apart.
#:
#: Version 1 predates the field itself, which is why it is also the default:
#: a row without ``schema_version`` *is* a v1 row.
RESULT_SCHEMA_VERSION = 1


def json_safe(payload: Any) -> Any:
    """Replace non-finite floats with their string tags, recursively.

    ``json.dump`` emits a bare ``NaN`` token happily and reads it back happily,
    so a diverged row round-trips fine *in Python* while being invalid JSON to
    ``jq`` and to every non-Python consumer. Since ``results/`` is append-only
    (CLAUDE.md rule 6), a row written wrong today cannot be repaired later —
    so every registry writer passes its payload through here first.

    :data:`Metrics` is the inverse, applied on the way back in.
    """
    if isinstance(payload, float) and not math.isfinite(payload):
        if math.isnan(payload):
            return "nan"
        return "inf" if payload > 0 else "-inf"
    if isinstance(payload, dict):
        return {k: json_safe(v) for k, v in payload.items()}
    if isinstance(payload, list | tuple):
        return [json_safe(v) for v in payload]
    return payload


def _read_nonfinite(value: Any) -> Any:
    """``{"loss": "nan"}`` -> ``{"loss": nan}`` when a row is loaded back."""
    if not isinstance(value, dict):
        return value
    return {
        k: NONFINITE_TAGS.get(v, v) if isinstance(v, str) else v
        for k, v in value.items()
    }


#: A metric mapping that survives the non-finite round-trip in both directions.
#: Stated explicitly rather than leaning on pydantic's lax ``str`` -> ``float``
#: coercion, because reading diverged rows back is load-bearing for the
#: failure-rate number in a paper (DESIGN.md §11).
Metrics = Annotated[dict[str, float], BeforeValidator(_read_nonfinite)]


class Spec(BaseModel):
    """Base for every config/schema model: immutable and strict.

    ``extra="forbid"`` turns a typo in a YAML key into a load-time error rather
    than a silently ignored hyperparameter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    DIVERGED = "diverged"
    FAILED = "failed"


class MetricSchedule(Spec):
    """When to append a point to the convergence trace.

    Traces are downsampled by construction (DESIGN.md §11): a full per-step trace
    across 1e4-1e5 runs is gigabytes of noise, while log-spaced points capture the
    curve shape. Stateless by design so that a resumed run records the same steps
    an uninterrupted one would.

    ``record_first`` asks for a point at step 0 — the loss before any optimizer
    has run. It costs one extra forward pass and it is the left-hand end of a
    log-log convergence plot, without which the first decade of the curve has
    nothing to start from.
    """

    every: int | None = Field(default=100, gt=0)
    n_per_decade: int | None = Field(default=None, gt=0)
    record_first: bool = True
    record_last: bool = True

    @model_validator(mode="after")
    def _records_something(self) -> MetricSchedule:
        if (
            self.every is None
            and self.n_per_decade is None
            and not self.record_first
            and not self.record_last
        ):
            raise ValueError(
                "this MetricSchedule can never record anything, so the run would "
                "produce an empty trace. Set at least one of every, n_per_decade, "
                "record_first, record_last."
            )
        return self

    def should_record(self, step: int, *, is_last: bool = False) -> bool:
        if is_last and self.record_last:
            return True
        if step == 0:
            return self.record_first
        if self.every is not None and step % self.every == 0:
            return True
        return self.n_per_decade is not None and _is_log_spaced_step(
            step, self.n_per_decade
        )


def _is_log_spaced_step(step: int, n_per_decade: int) -> bool:
    """True on ~``n_per_decade`` steps per decade, without carrying state."""
    if step < 1:
        return False
    k = round(n_per_decade * math.log10(step))
    return step == max(1, round(10 ** (k / n_per_decade)))


class TracePoint(Spec):
    """One downsampled row of the convergence trace."""

    step: int = Field(ge=0)
    stage: int = Field(default=0, ge=0)
    wall_time: float = Field(ge=0.0)
    metrics: Metrics = Field(default_factory=dict)


class Provenance(Spec):
    """Everything needed to place a result in space, time and hardware."""

    pinnslab_version: str
    git_sha: str
    git_dirty: bool
    git_source: str  # how the sha was resolved: "git" | "direct_url" | "unknown"
    gpu_name: str
    device_profile: str
    dtype: str
    seed: int
    timestamp_utc: str
    hostname: str
    python_version: str
    torch_version: str
    platform: str


class ResultRow(Spec):
    """One run, one row. The unit of everything downstream.

    ``timings`` is a first-class result, not metadata: compute parity including
    search cost is a core reviewer defence (DESIGN.md §8, §11).

    Unlike the rest of :class:`Spec`, this model **ignores** unknown keys. A row
    is only ever built in code by :meth:`from_provenance`, never parsed from
    hand-written YAML, so ``extra="forbid"`` catches no typos here — all it does
    is guarantee that the day a field is added, every row already on disk becomes
    unloadable by the code that reads it. Rows are permanent (CLAUDE.md rule 6),
    and a Kaggle session pinned to an older tag reading rows written by a newer
    one is the ordinary case, not an edge case.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: int = Field(default=RESULT_SCHEMA_VERSION, ge=1)

    run_id: str
    config_hash: str
    status: RunStatus

    # --- CLAUDE.md rule 7, non-negotiable -------------------------------------
    pinnslab_version: str
    git_sha: str
    git_dirty: bool
    seed: int
    gpu_name: str
    dtype: str
    device_profile: str
    # --------------------------------------------------------------------------

    timestamp_utc: str
    steps_completed: int = Field(default=0, ge=0)
    final_metrics: Metrics = Field(default_factory=dict)
    best_metrics: Metrics = Field(default_factory=dict)
    timings: Metrics = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    config: dict | None = None
    error: str | None = None

    @classmethod
    def from_provenance(
        cls, *, run_id: str, config_hash: str, status: RunStatus, prov: Provenance, **kw
    ) -> ResultRow:
        return cls(
            run_id=run_id,
            config_hash=config_hash,
            status=status,
            pinnslab_version=prov.pinnslab_version,
            git_sha=prov.git_sha,
            git_dirty=prov.git_dirty,
            seed=prov.seed,
            gpu_name=prov.gpu_name,
            dtype=prov.dtype,
            device_profile=prov.device_profile,
            timestamp_utc=prov.timestamp_utc,
            **kw,
        )


__all__ = [
    "NONFINITE_TAGS",
    "RESULT_SCHEMA_VERSION",
    "MetricSchedule",
    "Metrics",
    "Provenance",
    "ResultRow",
    "RunStatus",
    "Spec",
    "TracePoint",
    "json_safe",
]
