"""A metaheuristic search over a base config (DESIGN.md §6).

    python scripts/run_search.py examples/search.yaml \
        --base examples/configs/burgers_baseline.yaml --root analysis/search

The search's own hyperparameters — space, algorithm, population, generations,
fidelity rungs, fitness — live in the YAML and are hashed, exactly as a run's
are. ``--root`` holds the outer-loop checkpoint and the candidate cache, so an
interrupted search resumes at generation granularity and a re-proposed
configuration is free.

Which evaluator runs is a property of the search, not a flag: ``batched: true``
in the spec trains the whole population in one graph and scores the *training
objective*; ``batched: false`` runs each candidate through the ordinary
single-run path, which is what a fitness measured against ground truth
(``rel_l2``) requires. Passing ``--results`` under the sequential evaluator
turns every candidate into a real, auditable run directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

from pinnslab.registry.config import load_config
from pinnslab.search.evaluate import BatchedEvaluator, SequentialEvaluator
from pinnslab.search.loop import Search
from pinnslab.search.spec import load_search_spec
from pinnslab.utils.plugins import load_plugins


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_plugins(args.register)

    spec = load_search_spec(args.spec)
    base = load_config(args.base)

    if spec.batched and spec.fitness.metric == "rel_l2":
        raise SystemExit(
            "this spec asks for fitness 'rel_l2' with batched: true, but the "
            "batched evaluator never touches a reference solution — it scores "
            "the training objective. Set batched: false, or search on a metric "
            "the objective produces (see pinnslab/search/evaluate.py)."
        )

    evaluator = (
        BatchedEvaluator()
        if spec.batched
        else SequentialEvaluator(spec.fitness, root=args.results)
    )
    search = Search(spec, base, evaluator, root=args.root)

    print(
        f"{spec.name}: {spec.algorithm} over {len(spec.space)} axes, "
        f"pop {spec.pop_size} x {spec.generations} generations, "
        f"{spec.total_inner_steps} inner steps of compute"
    )
    state = search.run()

    # The measured cost next to the declared bound. DESIGN.md §8 makes compute
    # parity *including search cost* a reviewer defence, so these are results,
    # not progress output: the bound is what a protocol promised, the measured
    # numbers are what happened, and the cache is why they differ.
    bound = spec.total_inner_steps
    saved = bound - state.total_inner_steps
    print(
        f"search cost: {state.total_inner_steps} inner steps in "
        f"{state.total_seconds:.1f}s"
        + (
            f" (bound {bound}; the cache saved {saved})"
            if saved
            else f" (bound {bound}, no cache hits)"
        )
    )

    best = state.best()
    if best is None:
        print("no candidate was evaluated")
        return 1
    print(f"best fitness {best.fitness:.6g} at {best.steps} steps")
    for path, value in sorted(search.space.decode(best.vector).items()):
        print(f"  {path}: {value}")
    print(f"  config hash {best.config_hash}")
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="a search YAML")
    parser.add_argument(
        "--base", type=Path, required=True, help="the config the space varies"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("analysis/search"),
        help="where the outer-loop checkpoint and candidate cache live",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="sequential evaluator only: keep every candidate as a real run",
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
