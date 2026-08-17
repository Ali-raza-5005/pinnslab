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
* **against wall-clock, seeds are interpolated onto a common time grid.** The
  per-second figure is a §8 reviewer defence, and it cannot be built by
  intersecting exact timestamps: no two seeds finish a step at the same float
  second, so the intersection is ``{0.0}``. See :func:`band`.

Deliberately numpy-only. The aggregation a paper needs is a groupby and two
percentiles; pandas would buy little and would put the ``analysis`` extra
between a Kaggle session and a mid-sweep sanity check.

Reads raw, writes nothing. Derived files belong in ``analysis/``, written by
the caller (DESIGN.md §11).
"""

from __future__ import annotations

import math
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

#: x axes whose values are continuous, so two seeds never share one exactly and
#: :func:`band` must interpolate rather than intersect. ``step`` is deliberately
#: not here: seeds share it exactly, and interpolating a grid every seed already
#: lands on would invent points nobody measured.
CONTINUOUS_X = frozenset({"wall_time"})


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

    Two x axes, two alignment rules, because the axes are different kinds of
    number (see :data:`CONTINUOUS_X`):

    * **``step`` is shared exactly.** :class:`MetricSchedule` is stateless, so a
      resumed run records the steps an uninterrupted one would and every seed
      lands on the same grid. The band is drawn on the intersection. Where a
      seed stopped early, the curve is drawn on the steps **all** contributing
      seeds reached, and the shortfall is reported through ``n_used`` rather
      than papered over by averaging a shrinking population down the x-axis — a
      band that silently loses seeds as it goes right gets tighter exactly where
      the runs are failing.
    * **``wall_time`` is never shared.** Two seeds do not finish a step at the
      same float second, so intersecting exact times leaves only ``t=0`` — a
      one-point band, and an empty figure once a log axis drops it. Seeds are
      interpolated onto a common time grid instead; see
      :func:`_interpolate_onto_common_grid` for what that does and refuses to
      do.
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

    if x in CONTINUOUS_X:
        grid, values, n_used = _interpolate_onto_common_grid(per_run, x=x, label=label)
    else:
        grid, values, n_used = _intersect_exactly(
            per_run, x=x, metric=metric, label=label, n_records=len(records)
        )

    q25, median, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
    return Band(
        label=label,
        metric=metric,
        x=grid,
        median=median,
        q25=q25,
        q75=q75,
        n_total=len(records),
        n_used=n_used,
    )


def _intersect_exactly(
    per_run: Sequence[dict[float, float]],
    *,
    x: str,
    metric: str,
    label: str,
    n_records: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The shared grid for an x every seed lands on exactly (``step``)."""
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
            n_records,
        )
    values = np.array([[p[k] for k in common] for p in per_run], dtype=float)
    return np.asarray(common, dtype=float), values, len(per_run)


def _interpolate_onto_common_grid(
    per_run: Sequence[dict[float, float]], *, x: str, label: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """The shared grid for a continuous x (``wall_time``), by interpolation.

    Five decisions, each of which is a way to get a per-second figure wrong:

    * **The grid is the union of the seeds' own timestamps**, not an arbitrary
      linspace. There is no resolution knob to pick, and the band keeps the
      density the trace schedule already chose — dense early, sparse late,
      which is where a log time axis wants its points anyway.
    * **Clipped to the overlap ``[max(first), min(last)]``: no extrapolation.**
      Past the shortest seed's last timestamp the median would be over a
      shrinking population, which is the same failure the ``step`` path refuses.
      The clip is reported, because "the band stops at 40s while one seed ran
      for 90s" is information about the comparison.
    * **Interpolation is linear in log(metric)** when a seed's values are all
      positive, and linear otherwise. The figure's y axis is logarithmic, so a
      straight line *on the plot* is a log-linear one; interpolating linearly
      there would draw a curve that bulges above every point it connects.
    * **Non-finite points are dropped per seed** before interpolating. A NaN
      inside a trace would otherwise poison a whole interval of the median.
      A seed left with fewer than two finite points cannot be interpolated and
      leaves the band, which ``n_used`` then reports.
    * **Duplicate timestamps keep the last value.** A coarse clock can stamp
      two trace points identically; ``np.interp`` needs a strictly increasing
      x. (``per_run`` is already keyed on x, so this is the dict's doing.)
    """
    curves = []
    for points in per_run:
        finite = {t: v for t, v in points.items() if math.isfinite(v)}
        if len(finite) < 2:
            continue
        times = np.array(sorted(finite), dtype=float)
        curves.append((times, np.array([finite[t] for t in times], dtype=float)))

    if len(curves) != len(per_run):
        log.warning(
            "%r: %d seed(s) have fewer than two finite trace points and cannot "
            "be interpolated onto a %s grid; they are excluded from the band",
            label,
            len(per_run) - len(curves),
            x,
        )
    if not curves:
        raise ValueError(
            f"no seed has two finite trace points, so there is no {x} interval "
            "over which to compare them"
        )

    lo = max(float(times[0]) for times, _ in curves)
    hi = min(float(times[-1]) for times, _ in curves)
    if not hi > lo:
        raise ValueError(
            f"the runs share no common {x} interval: they overlap only at "
            f"{lo:g}. Every seed must have run long enough to be compared with "
            "the shortest one."
        )

    longest = max(float(times[-1]) for times, _ in curves)
    if longest > hi:
        log.warning(
            "%r: the %s band ends at %.4g (the shortest seed's last point); "
            "the longest seed reached %.4g and its tail is not extrapolated "
            "over",
            label,
            x,
            hi,
            longest,
        )

    grid = np.unique(np.concatenate([times for times, _ in curves]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    values = np.array([_interp(grid, times, ys) for times, ys in curves])
    return grid, values, len(curves)


def _interp(grid: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate onto ``grid``, in log space where the metric allows it."""
    if np.all(values > 0.0):
        return np.exp(np.interp(grid, times, np.log(values)))
    return np.interp(grid, times, values)


def _x_of(point: TracePoint, x: str) -> float:
    if x == "step":
        return float(point.step)
    if x == "wall_time":
        return float(point.wall_time)
    raise ValueError(f"x must be 'step' or 'wall_time', got {x!r}")


__all__ = [
    "CONTINUOUS_X",
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
