"""Result tables as LaTeX, generated — never typed (DESIGN.md §8).

A hand-typed table is the single easiest place in a paper to publish a number
that no longer matches the run that produced it. These emit `booktabs` from the
same :class:`~pinnslab.viz.aggregate.Summary` objects the figures use, so a
table and the plot beside it cannot disagree.

House conventions, matching the figures:

* **median [q25, q75]**, never mean ± std — PINN error distributions are
  heavy-tailed (DESIGN.md §8).
* **the seed count is a column**, and so is the failure count when any
  condition has one. "3/5 converged" is a result.
* `booktabs` rules only: `\\toprule`, `\\midrule`, `\\bottomrule`. No vertical
  rules and no `\\hline` between every row — that is the house style of every
  journal that publishes tables well.
* numbers are formatted once, here, so significant figures are consistent down
  a column.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pinnslab.viz.aggregate import Summary

#: `\usepackage` lines a caller's preamble needs for the output to compile.
REQUIRED_PACKAGES = ("booktabs",)


def format_number(value: float, *, sig: int = 3, scientific: bool | None = None) -> str:
    """One number, LaTeX math.

    ``nan`` becomes an em dash rather than the string "nan": a condition where
    every seed failed has no median, and printing one would be a lie whereas a
    dash is the truth the failure column then explains.

    ``scientific`` is normally decided once for a whole column by
    :func:`column_format` — see the note there on why per-value decisions are
    wrong.
    """
    body = _bare_number(value, sig=sig, scientific=scientific)
    return body if body == "---" else f"${body}$"


def column_format(values: Sequence[float], *, sig: int = 3) -> bool:
    """Whether a column of numbers should be written in scientific notation.

    One decision for the whole column, not one per value. Deciding per value
    puts ``9.34 \\times 10^{-4}`` directly above ``0.00111`` — the same
    quantity, two notations, and a reader comparing them has to convert in
    their head. Which is exactly the moment a reader misreads a result table.
    """
    finite = [abs(v) for v in values if v is not None and math.isfinite(v) and v != 0]
    if not finite:
        return False
    return min(finite) < 1e-2 or max(finite) >= 1e4


def _bare_number(
    value: float, *, sig: int = 3, scientific: bool | None = None
) -> str:
    """:func:`format_number` without the surrounding ``$``, for composite cells."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "---" if value is None or math.isnan(value) else r"\infty"
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if scientific is None:
        scientific = not (-3 <= exponent < 4)
    if not scientific:
        return f"{value:.{max(0, sig - 1 - exponent)}f}"
    mantissa = value / 10**exponent
    return rf"{mantissa:.{sig - 1}f} \times 10^{{{exponent}}}"


def summary_table(
    summaries: Sequence[Summary],
    *,
    caption: str = "",
    label: str = "",
    condition_header: str = "Method",
    metric_header: str | None = None,
) -> str:
    """A `booktabs` table of median [IQR] per condition, as a LaTeX string.

    The failure column appears only when something failed — an all-zero column
    is noise, and its absence is itself readable as "nothing diverged".
    """
    if not summaries:
        raise ValueError("no summaries to tabulate")

    metric = metric_header or _metric_header(summaries)
    show_failures = any(s.n_used != s.n_total for s in summaries)
    # One notation for every number in the table: median and quartiles are the
    # same quantity and must not be written two different ways.
    sci = column_format(
        [v for s in summaries for v in (s.median, s.q25, s.q75)]
    )

    columns = "lrrr" + ("r" if show_failures else "")
    header = [condition_header, metric, "IQR", "$n$"]
    if show_failures:
        header.append("Failed")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for s in summaries:
        # The IQR cell holds two numbers under one pair of ``$``; going through
        # format_number twice would nest math mode and not compile.
        iqr = (
            rf"$[{_bare_number(s.q25, scientific=sci)},\, "
            rf"{_bare_number(s.q75, scientific=sci)}]$"
        )
        row = [
            _escape(s.label),
            format_number(s.median, scientific=sci),
            "---" if math.isnan(s.q25) else iqr,
            str(s.n_total),
        ]
        if show_failures:
            row.append(str(s.n_total - s.n_used))
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        # A \label argument is an identifier, not prose, so an underscore is
        # sanitised rather than escaped — `\_` inside \label breaks hyperref's
        # anchor names, and `tab:rel_l2` breaks plain LaTeX outright.
        lines.append(rf"\label{{{label.replace('_', '-')}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _metric_header(summaries: Sequence[Summary]) -> str:
    metrics = {s.metric for s in summaries}
    if len(metrics) > 1:
        raise ValueError(
            f"a summary table holds one metric per column, got {sorted(metrics)}; "
            "build one table per metric, or pass metric_header explicitly"
        )
    from pinnslab.viz.convergence import label_for

    return label_for(metrics.pop())


def _escape(text: str) -> str:
    """LaTeX-escape a condition label.

    Config hashes are hex and harmless, but tags are free text, and an
    underscore in one is a compile error three days before a deadline.
    Backslash goes first and to a command rather than to ``\\\\``, which LaTeX
    reads as a line break rather than a literal backslash.
    """
    text = text.replace("\\", r"\textbackslash{}")
    for char in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, rf"\{char}")
    return text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


__all__ = ["REQUIRED_PACKAGES", "column_format", "format_number", "summary_table"]
