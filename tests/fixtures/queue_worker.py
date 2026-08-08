"""A Kaggle session, as a subprocess that can be killed for real.

Run as::

    python queue_worker.py <matrix.csv> <results_root> [<run_id> <step>]

With the optional pair, the process calls ``os._exit`` the moment run
``<run_id>`` reaches ``<step>``. That is deliberately not an exception: an
exception unwinds, runs ``finally`` blocks and lets Python flush its buffers,
which is the one thing a killed Kaggle session does *not* do. ``os._exit``
skips all of it, so whatever is on disk afterwards is exactly what a real
session death leaves — including a half-written trailing line in an
append-only file.

The kill is injected by wrapping the trainer's ``residual_fn`` rather than by
adding a hook to the library: the queue must be provably resumable as shipped,
not as instrumented.
"""

from __future__ import annotations

import os
import sys

import pinnslab.training.queue as queue_module
from pinnslab.training.queue import load_matrix, run_queue


def install_kill(run_id: str, step: int) -> None:
    real_build = queue_module.build_trainer

    def build_and_arm(cfg, ctx, run, **kwargs):
        trainer = real_build(cfg, ctx, run, **kwargs)
        if run.run_id != run_id:
            return trainer

        inner = trainer.residual_fn

        def residual(state):
            if state.step >= step:
                sys.stderr.write(f"KILLING {run_id} at step {state.step}\n")
                sys.stderr.flush()
                os._exit(137)
            return inner(state)

        trainer.residual_fn = residual
        return trainer

    queue_module.build_trainer = build_and_arm


def main(argv: list[str]) -> int:
    matrix, root = argv[1], argv[2]
    if len(argv) > 3:
        install_kill(argv[3], int(argv[4]))

    report = run_queue(load_matrix(matrix), root)
    print(f"COMPLETED {len(report.completed)} FAILED {len(report.failed)}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
