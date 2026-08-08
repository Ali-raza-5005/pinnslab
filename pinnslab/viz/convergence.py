"""The convergence figure (DESIGN.md §9 step 4).

The one plot every PINN methods paper opens with: error against training
budget, several conditions, several seeds each. What it must not do is show a
single seed per condition — with PINN error spreads of the size measured on
2026-08-08 (0.030 to 0.435 across seeds of one *correct* config), a
single-seed convergence plot is a plot of which seed the author picked.

So the primitive here is a **band, not a line**: median across seeds with the
interquartile range shaded, and the seed count stated in the legend. Failure is
on the figure too — a condition where seeds diverged is annotated rather than
silently drawn from the survivors.

The x-axis is a choice with consequences (DESIGN.md §8): ``step`` compares
methods at equal iterations, ``wall_time`` at equal compute. A method that wins
per-step and loses per-second has not won, and the reviewer will ask. Both are
one argument apart here so that both get looked at.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from pinnslab.registry.schema import ResultRow
from pinnslab.utils.logging import get_logger
from pinnslab.viz import style
from pinnslab.viz.aggregate import (
    Band,
    RunRecord,
    assert_comparable,
    band,
    group,
    summarise,
)

log = get_logger(__name__)

#: Axis labels for the metrics the library records. A metric not in here is
#: labelled by its raw key, which is a prompt to add it rather than an error.
AXIS_LABELS = {
    "rel_l2": r"relative $L_2$ error",
    "max_error": r"max error $\|u - u^\star\|_\infty$",
    "loss": "training loss",
    "step": "training step",
    "wall_time": "wall-clock time (s)",
}


def label_for(key: str) -> str:
    return AXIS_LABELS.get(key, key.replace("_", " "))


def plot_bands(
    ax: plt.Axes,
    bands: Sequence[Band],
    *,
    show_iqr: bool = True,
    positive_x_only: bool = False,
) -> None:
    """Draw median curves with IQR shading onto an existing axes.

    Colors are assigned by position in ``bands`` and never cycled, so a
    condition keeps its identity between figures as long as the caller keeps
    its order. The band is drawn *under* its line and at low alpha: it is
    context for the median, not a mark competing with it.

    ``positive_x_only`` drops ``x <= 0`` for a log x-axis. Matplotlib does that
    silently; here it is done deliberately and reported, because the point
    being dropped is usually step 0 — the untrained baseline that
    ``MetricSchedule.record_first`` exists to capture, and the left-hand end of
    the curve.
    """
    for index, b in enumerate(bands):
        hue = style.color(index)
        x, q25, q75, median = b.x, b.q25, b.q75, b.median
        if positive_x_only and (keep := x > 0).sum() != x.size:
            log.warning(
                "%r: dropping %d point(s) at x <= 0 from a log axis (step 0 is "
                "the untrained baseline); use xscale='linear' to keep it",
                b.label or b.metric,
                int((~keep).sum()),
            )
            x, q25, q75, median = x[keep], q25[keep], q75[keep], median[keep]

        if show_iqr and b.n_used > 1:
            ax.fill_between(
                x, q25, q75, color=hue, alpha=0.16, linewidth=0, zorder=1
            )
        ax.plot(
            x,
            median,
            label=_legend_label(b),
            zorder=2,
            **style.series_kwargs(index),
        )


def _legend_label(b: Band) -> str:
    """The seed count rides in the legend, because a median over 2 seeds and a
    median over 5 are different claims and the figure should say which."""
    if b.n_used == b.n_total:
        return f"{b.label} ($n={b.n_total}$)"
    return f"{b.label} ({b.n_used}/{b.n_total} seeds)"


def convergence_figure(
    records: Sequence[RunRecord],
    *,
    metric: str = "rel_l2",
    x: str = "step",
    by: str | Callable[[ResultRow], str] = "config_hash",
    order: Sequence[str] | None = None,
    title: str = "",
    width: str = "single",
    show_iqr: bool = True,
    xscale: str = "log",
    include_diverged: bool = False,
) -> plt.Figure:
    """A publication-ready convergence figure straight from ``results/``.

    ``order`` fixes which condition gets which color; without it the order is
    whatever :func:`group` produced, which is stable but arbitrary. Pass it for
    any figure that will appear alongside another — and put the method being
    argued for in slot 0, since that is the hue a reader will carry between
    figures.

    ``xscale`` defaults to ``"log"``: the trace is log-spaced by construction
    and the early decades are where sampling strategies separate. ``"linear"``
    is the other common convention and is the one that can show step 0.
    """
    assert_comparable(records)
    groups = group(records, by)
    keys = list(order) if order else list(groups)
    missing = [k for k in keys if k not in groups]
    if missing:
        raise KeyError(f"order names conditions with no runs: {missing}")

    bands = [
        band(groups[k], metric, label=k, x=x, include_diverged=include_diverged)
        for k in keys
    ]

    with style.use_style():
        fig, ax = plt.subplots(figsize=style.figsize(width))
        plot_bands(ax, bands, show_iqr=show_iqr, positive_x_only=xscale == "log")

        ax.set_xlabel(label_for(x))
        ax.set_ylabel(label_for(metric))
        style.log_axis(ax, "y")
        if xscale == "log":
            style.log_axis(ax, "x")
        elif xscale != "linear":
            raise ValueError(f"xscale must be 'log' or 'linear', got {xscale!r}")
        else:
            ax.grid(True, which="major", axis="x")
        if title:
            ax.set_title(title)
        if len(bands) > 1:
            ax.legend(loc="best")
        _note_failures(ax, groups, keys, metric)
    return fig


def _note_failures(
    ax: plt.Axes,
    groups: dict[str, list[RunRecord]],
    keys: Sequence[str],
    metric: str,
) -> None:
    """Put the failure rate on the figure when it is not zero.

    DESIGN.md §8 requires it reported; a footnote in the caption is where it
    goes to be forgotten. If every seed of every condition converged, this
    draws nothing — an annotation that is always present stops being read.
    """
    summaries = [(k, summarise(groups[k], metric, label=k)) for k in keys]
    failed = [(k, s) for k, s in summaries if s.n_used != s.n_total]
    if not failed:
        return
    text = "; ".join(
        f"{k}: {s.n_total - s.n_used}/{s.n_total} failed" for k, s in failed
    )
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=style.MUTED_INK,
    )


def write_convergence_figure(
    results_root: str | Path,
    out: str | Path,
    **kwargs,
) -> Path:
    """``results/`` -> a PDF on disk, in one call and with no manual steps.

    This is the DESIGN.md §9 step 4 loop closed: a config produced those runs,
    and nothing between them and the figure is done by hand.
    """
    from pinnslab.viz.aggregate import load_records

    fig = convergence_figure(load_records(results_root), **kwargs)
    path = style.save(fig, out)
    plt.close(fig)
    return path


__all__ = [
    "AXIS_LABELS",
    "convergence_figure",
    "label_for",
    "plot_bands",
    "write_convergence_figure",
]
