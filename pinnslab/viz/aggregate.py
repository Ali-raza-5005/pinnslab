"""``results/`` -> the numbers a figure or table is allowed to show.

Everything DESIGN.md §8 requires of a headline claim is enforced here rather
than left to whoever writes the plotting script:

* **median + IQR, never mean ± std.** PINN errors are heavy-tailed; one
  divergent seed moves a mean by orders of magnitude and moves a median not at
  all. The 2026-08-08 Burgers bring-up produced exactly this — one seed at
  0.435 against siblings at 0.030-0.044.
* **failure rate is reported, not hidden.** A summary carries ``n_total`` and
  ``n_used``, so "3/5 seeds converged" is available to every caller and a
  figure cannot quietly average over the survivors.
* **one comparison group, one hardware.** :func:`assert_comparable` refuses to
  aggregate rows that span GPUs or dtypes (DESIGN.md §5). Timings across a T4
  and a P100 are not comparable, and neither is float32 accuracy against
  float64.

Deliberately numpy-only. The aggregation a paper needs is a groupby and two
percentiles; pandas would buy little and would put the ``analysis`` extra
between a Kaggle session and a mid-sweep sanity check.

Reads raw, writes nothing. Derived files belong in ``analysis/``, written by
the caller (DESIGN.md §11).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnslab.registry.run import load_runs, read_trace
from pinnslab.registry.schema import ResultRow, RunStatus, TracePoint
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

#: How a run is labelled on a figure when the caller gives no rule. The config
#: hash, because that *is* the condition (DESIGN.md §4) — a name would be a
#: second, unhashed identity that can drift out of step with the experiment.
DEFAULT_LABEL = "config_hash"


@dataclass(frozen=True)
class RunRecord:
    """One run: its row, and its convergence trace."""

    row: ResultRow
    trace: tuple[TracePoint, ...]

    @property
    def converged(self) -> bool:
        return self.row.status is RunStatus.COMPLETED


@dataclass(frozen=True)
class Summary:
    """A metric across the seeds of one condition.

    ``n_total`` counts every run of the condition, including the ones that
    failed; ``n_used`` counts those that produced a finite value. Reporting the
    first without the second is how a failure rate disappears from a paper.
    """

    label: str
    metric: str
    median: float
    q25: float
    q75: float
    n_total: int
    n_used: int

    @property
    def iqr(self) -> float:
        return self.q75 - self.q25

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.n_total == 0 else 1.0 - self.n_used / self.n_total

    def __str__(self) -> str:
        return (
            f"{self.label}: {self.metric} median {self.median:.3g} "
            f"[{self.q25:.3g}, {self.q75:.3g}] over {self.n_used}/{self.n_total} seeds"
        )


def load_records(
    root: str | Path, *, include_unfinished: bool = True
) -> list[RunRecord]:
    """Every run under ``root``, with its trace.

    ``include_unfinished`` defaults to **True** here, unlike
    :func:`pinnslab.registry.run.load_runs`. A figure almost always reports a
    rate of some kind, and a rate computed only over the runs that survived
    long enough to write a row is not that rate.
    """
    root = Path(root)
    rows = load_runs(root, include_unfinished=include_unfinished)
    records = [
        RunRecord(row=row, trace=tuple(read_trace(root / row.run_id))) for row in rows
    ]
    log.info(
        "loaded %d run(s) from %s (%d completed)",
        len(records),
        root,
        sum(r.converged for r in records),
    )
    return sorted(records, key=lambda r: (r.row.config_hash, r.row.seed))


def assert_comparable(records: Sequence[RunRecord]) -> None:
    """Refuse to plot a comparison group that spans hardware or precision.

    DESIGN.md §5 makes this the code's job, not the author's memory: an entire
    comparison group — every method × seed in one figure — must run on one GPU
    type and one dtype. Mixed timings are meaningless, and float32 hits a
    residual noise floor around 1e-4-1e-5, exactly where accuracy claims live.
    """
    if not records:
        return
    for field_name in ("gpu_name", "dtype"):
        values = {getattr(r.row, field_name) for r in records}
        if len(values) > 1:
            raise ValueError(
                f"this comparison group spans {len(values)} values of "
                f"{field_name}: {sorted(values)}. Results from different "
                "hardware or precision are not comparable and must not share a "
                "figure (DESIGN.md §5). Filter to one before plotting."
            )


def group(
    records: Iterable[RunRecord],
    by: str | Callable[[ResultRow], str] = DEFAULT_LABEL,
) -> dict[str, list[RunRecord]]:
    """Bucket runs into conditions.

    ``by`` is a row attribute name, a ``tags`` key, or a callable. Seeds of one
    condition land together, which is what the median-and-IQR over ≥5 seeds
    needs to group on.
    """
    label = by if callable(by) else _attr_or_tag(by)
    buckets: dict[str, list[RunRecord]] = {}
    for record in records:
        buckets.setdefault(label(record.row), []).append(record)
    return buckets


def _attr_or_tag(name: str) -> Callable[[ResultRow], str]:
    def label(row: ResultRow) -> str:
        if hasattr(row, name):
            return str(getattr(row, name))
        if name in row.tags:
            return row.tags[name]
        raise KeyError(
            f"run {row.run_id} has no attribute or tag {name!r}; tags are "
            f"{sorted(row.tags)}. Tag the runs in their config, or group by "
            "config_hash."
        )

    return label


def summarise(
    records: Sequence[RunRecord],
    metric: str,
    *,
    label: str = "",
    best: bool = False,
) -> Summary:
    """Median and IQR of a final (or best) metric across a condition's seeds."""
    source = "best_metrics" if best else "final_metrics"
    values = [
        value
        for record in records
        if record.converged
        and np.isfinite(value := getattr(record.row, source).get(metric, np.nan))
    ]
    if not values:
        return Summary(label, metric, np.nan, np.nan, np.nan, len(records), 0)

    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return Summary(
        label=label,
        metric=metric,
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        n_total=len(records),
        n_used=len(values),
    )


@dataclass(frozen=True)
class Band:
    """A median convergence curve with its IQR, across the seeds of one condition."""

    label: str
    metric: str
    x: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    n_total: int
    n_used: int


def band(
    records: Sequence[RunRecord],
    metric: str,
    *,
    label: str = "",
    x: str = "step",
    include_diverged: bool = False,
) -> Band:
    """Median-and-IQR convergence curve over seeds, on their common steps.

    **Diverged and failed runs are excluded by default**, and this is not a
    detail. A diverged seed's trace is real data — it is what the failure rate
    is computed from — but it is not part of "the error this method reaches",
    and letting it into the median produces a figure whose legend says ``n=5``
    while its own failure annotation says one of the five failed. The two
    numbers have to come from the same population. ``include_diverged=True``
    is there for the figure whose subject *is* the divergence.

    Seeds of one condition share a step grid exactly — :class:`MetricSchedule`
    is stateless, so a resumed run records the steps an uninterrupted one
    would. Where they do not (a seed that stopped early), the curve is drawn on
    the steps **all** contributing seeds reached, and the shortfall is reported
    through ``n_used`` rather than papered over by averaging a shrinking
    population down the x-axis — a band that silently loses seeds as it goes
    right gets tighter exactly where the runs are failing.
    """
    eligible = [r for r in records if include_diverged or r.converged]
    usable = [r for r in eligible if r.trace]
    if not usable:
        raise ValueError(
            f"no {'run' if eligible else 'converged run'} in this group has a "
            f"trace, so there is no {metric} curve to draw. "
            + (
                "Check that the runs logged metrics at all."
                if eligible
                else f"All {len(records)} run(s) diverged or failed; pass "
                "include_diverged=True to plot them anyway."
            )
        )

    per_run = []
    for record in usable:
        points = {
            _x_of(point, x): point.metrics[metric]
            for point in record.trace
            if metric in point.metrics
        }
        if points:
            per_run.append(points)
    if not per_run:
        raise ValueError(
            f"metric {metric!r} is in no trace point; available metrics are "
            f"{sorted({k for r in usable for p in r.trace for k in p.metrics})}"
        )

    common = sorted(set.intersection(*(set(p) for p in per_run)))
    if not common:
        raise ValueError(f"the runs share no common {x} at which to compare {metric}")
    dropped = max(len(p) for p in per_run) - len(common)
    if dropped:
        log.warning(
            "%r: %d trace point(s) past the shortest run are excluded from the "
            "band; %d/%d seeds contribute",
            label,
            dropped,
            len(per_run),
            len(records),
        )

    values = np.array([[p[k] for k in common] for p in per_run], dtype=float)
    q25, median, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
    return Band(
        label=label,
        metric=metric,
        x=np.asarray(common, dtype=float),
        median=median,
        q25=q25,
        q75=q75,
        n_total=len(records),
        n_used=len(per_run),
    )


def _x_of(point: TracePoint, x: str) -> float:
    if x == "step":
        return float(point.step)
    if x == "wall_time":
        return float(point.wall_time)
    raise ValueError(f"x must be 'step' or 'wall_time', got {x!r}")


__all__ = [
    "DEFAULT_LABEL",
    "Band",
    "RunRecord",
    "Summary",
    "assert_comparable",
    "band",
    "group",
    "load_records",
    "summarise",
]
