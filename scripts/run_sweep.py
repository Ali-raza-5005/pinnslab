"""A whole run matrix, resumably. The local twin of the Kaggle notebook.

    python scripts/run_sweep.py examples/run_matrix.csv --results results/

Same code as ``notebooks/kaggle_runner.py`` cell 3, so what runs on a laptop and
what runs in a session are the same thing. Re-running is always safe: finished
cells are skipped, an interrupted one resumes from its last checkpoint, and the
result is bit-identical to an uninterrupted sweep
(``tests/unit/test_queue_survives_a_killed_session.py``).

Two GPUs, or two terminals, are ``--worker 0 --workers 2`` and ``--worker 1
--workers 2``: the matrix is partitioned statically by row, so the workers never
consider the same cell and need no coordination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

from pinnslab.training.queue import load_matrix, run_queue, statuses
from pinnslab.utils.plugins import load_plugins


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_plugins(args.register)

    cells = load_matrix(args.matrix)
    report = run_queue(
        cells,
        args.results,
        worker=args.worker,
        workers=args.workers,
        deadline_seconds=args.deadline,
    )

    print(report)
    for cell, status in statuses(cells, args.results):
        print(f"  {status.value:10} {cell.config.name} seed={cell.seed}")
    # A deferred cell is not a failure: it is the next session's work.
    return 1 if report.failed else 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="a run_matrix.csv")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--deadline",
        type=float,
        default=None,
        metavar="SECONDS",
        help="stop claiming new cells once less than the longest cell's "
        "runtime is left (Kaggle's wall clock, minus room to publish)",
    )
    parser.add_argument(
        "--register",
        action="append",
        metavar="MODULE_OR_PATH",
        help="import a module that registers components. Repeatable.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
