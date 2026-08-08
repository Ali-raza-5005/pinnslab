"""``results/`` -> every figure and table, with zero manual steps.

The DESIGN.md §9 step 4 loop, closed: a config produced the runs, the runs are
in ``results/``, and this script turns them into the PDFs and ``.tex`` fragments
a paper `\\input`s. Nothing between them is done by hand, so regenerating after
another seed lands is one command and there is no such thing as a figure that
is out of date with the data.

Derived output goes to ``analysis/`` and is disposable — ``results/`` is
append-only and is never touched here (CLAUDE.md rule 6, DESIGN.md §11).

    python scripts/make_figures.py results/ --out analysis/ --by method

A paper repo's own ``phases/p3_analysis/figures/`` script is expected to look
almost exactly like this one, importing the same primitives. This is the
worked example, not a framework.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinnslab.viz import style
from pinnslab.viz.aggregate import assert_comparable, group, load_records, summarise
from pinnslab.viz.convergence import convergence_figure, label_for
from pinnslab.viz.tables import summary_table


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    records = load_records(args.results)
    if not records:
        print(f"no runs under {args.results}", file=sys.stderr)
        return 1

    # Fail before drawing anything rather than after: a figure mixing a T4 and
    # a P100 is not a figure with a caveat, it is not a result (DESIGN.md §5).
    assert_comparable(records)

    groups = group(records, args.by)
    order = args.order or sorted(groups)
    unknown = [k for k in order if k not in groups]
    if unknown:
        print(f"--order names conditions with no runs: {unknown}", file=sys.stderr)
        return 1
    print(f"{len(records)} run(s) in {len(groups)} condition(s): {', '.join(order)}")

    for metric in args.metrics:
        for x in args.x:
            fig = convergence_figure(
                records,
                metric=metric,
                x=x,
                by=args.by,
                order=order,
                width=args.width,
                xscale=args.xscale,
            )
            style.save(fig, args.out / "figures" / f"convergence_{metric}_vs_{x}")

        summaries = [summarise(groups[k], metric, label=k) for k in order]
        for s in summaries:
            print(f"  {s}")

        tex = args.out / "tables" / f"{metric}.tex"
        tex.parent.mkdir(parents=True, exist_ok=True)
        # label_for, not the raw key: `rel_l2` in a caption is an unescaped
        # underscore, i.e. a LaTeX compile error in the paper that \input's it.
        tex.write_text(
            summary_table(
                summaries,
                caption=(
                    f"Median and interquartile range of {label_for(metric)} "
                    f"over seeds, grouped by {args.by}."
                ),
                label=f"tab:{metric}",
            ),
            encoding="utf-8",
        )
        print(f"  wrote {tex}")

    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="an append-only results/ root")
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument(
        "--by",
        default="config_hash",
        help="row attribute or tag naming the condition (e.g. 'method')",
    )
    parser.add_argument("--metrics", nargs="+", default=["rel_l2"])
    parser.add_argument(
        "--order",
        nargs="+",
        default=None,
        help="condition order, which fixes the colors. Put the method being "
        "argued for first: that is the hue a reader carries between figures. "
        "Alphabetical if omitted.",
    )
    parser.add_argument("--xscale", default="log", choices=["log", "linear"])
    parser.add_argument(
        "--x",
        nargs="+",
        default=["step", "wall_time"],
        help="both by default: a method that wins per-step and loses per-second "
        "has not won, and the reviewer will ask (DESIGN.md §8)",
    )
    parser.add_argument("--width", default="single", choices=sorted(style.WIDTHS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
