"""One config, one seed, one run directory. The smallest complete experiment.

    python scripts/run.py examples/configs/burgers_baseline.yaml --results results/

Everything that identifies the run — the condition, the seed, the precision, the
hardware — is either in the config or recorded from the machine, so this script
takes no hyperparameter of its own (CLAUDE.md rule 4). ``--seed`` is the one
exception, and only because a run *is* the pair ``(config_hash, seed)``: it is
excluded from the hash by design, so overriding it here still names the same
condition (DESIGN.md §4).

Resuming is not a mode. Point the script at the same results root with the same
config and seed and it continues from ``last.pt``; the run id is derived from
the condition, so there is nothing to remember and nothing to pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

from pinnslab.registry.config import load_config
from pinnslab.registry.run import Run
from pinnslab.training.build import build_trainer
from pinnslab.training.queue import CellStatus, run_id_for, status_of
from pinnslab.utils.device import configure_runtime
from pinnslab.utils.plugins import load_plugins


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_plugins(args.register)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    run_id = run_id_for(cfg)
    # A finished run is immutable (CLAUDE.md rule 6), so re-running this command
    # is a no-op rather than an error: "resume" and "already done" are the same
    # gesture from the user's side, and the derived status is what tells them
    # apart. Deriving it here rather than catching the exception keeps one
    # definition of "done" for the script and for the sweep.
    if status_of(args.results, run_id) is CellStatus.DONE:
        print(f"already done: {run_id}\n  -> {Path(args.results) / run_id}")
        return 0

    # configure_runtime first, always: it applies the seed and the dtype, and
    # both must be in force before a single parameter is allocated.
    ctx = configure_runtime(cfg)
    run = Run.create_or_resume(cfg, args.results, run_id)
    row = build_trainer(cfg, ctx, run).fit()

    print(f"{row.status.value} {row.run_id} after {row.steps_completed} steps")
    for name, value in sorted(row.final_metrics.items()):
        print(f"  {name}: {value:.6g}")
    print(f"  -> {Path(args.results) / row.run_id}")
    return 0 if row.status.value == "completed" else 1


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="a run config YAML")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="append-only results root (default: results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the config's seed; the condition is unchanged",
    )
    parser.add_argument(
        "--register",
        action="append",
        metavar="MODULE_OR_PATH",
        help="import a module that registers components (a paper's sampler, "
        "weighting, ...). Repeatable.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
