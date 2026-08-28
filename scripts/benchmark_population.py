"""Measure the batched population evaluator's speedup — on YOUR hardware.

    python scripts/benchmark_population.py
    python scripts/benchmark_population.py --sizes 4 8 16 32 50 --steps 25

DESIGN.md §6 claims batching a population into one graph is worth "20-50x on a
T4". **That number has never been measured.** It is a GPU claim about kernel
launch overhead dominating for tiny networks, and the only hardware this repo
has run on is a laptop CPU, where the measured speedup is 1.7x at P=4, 2.8x at
P=8, 3.4x at P=16 and ~2.2x by P=50 (2026-08-08, torch 2.12, float64, a real
Burgers residual). The curve is not monotone: more candidates is not more
speedup, and pop_size 50 may cost more per candidate than 16.

An unverified speedup is a hole in the compute-parity defence of §8, so this
script exists to close it with one command wherever a GPU turns up. Run it, then
put the numbers — with the device, the dtype and the population sizes — into
LOG.md and §6. Until then, do not quote §6's figure in a paper.

What is compared
----------------
Both arms train P *independent* candidates for the same steps, from the same
initial weights, on the same clouds, through the same
:func:`~pinnslab.search.population.train_population`. The only difference is
whether the population shares one graph. The final losses are printed side by
side because a fast wrong answer is not a speedup; that they agree is pinned
properly in ``tests/unit/test_search_population.py``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)
import torch

from pinnslab.registry.config import load_config
from pinnslab.search.evaluate import with_step_budget
from pinnslab.search.population import Ensemble, train_population
from pinnslab.training.build import assemble
from pinnslab.utils.device import configure_runtime

# Internal helpers used on purpose: the benchmark must measure the path the
# search actually takes, and re-implementing the point stacking here would be a
# second, drifting copy of it.
from pinnslab.search.evaluate import (  # isort: skip
    _population_residual,
    _stack_points,
)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "configs"
    / "burgers_uniform.yaml"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    cfg = load_config(args.config)
    updates = {}
    if args.device:
        updates["device"] = args.device
    if args.dtype:
        updates["dtype"] = args.dtype
    cfg = with_step_budget(cfg.model_copy(update=updates), args.steps)
    ctx = configure_runtime(cfg)

    print(
        f"device={ctx.device} gpu={ctx.gpu_name or 'none'} dtype={ctx.dtype_name} "
        f"steps={args.steps} config={Path(args.config).name}"
    )
    print(f"{'P':>4} {'batched (s)':>12} {'separate (s)':>13} {'speedup':>9}  loss")

    for size in args.sizes:
        print(f"{size:>4} {_row(cfg, ctx, size, args.steps)}")

    print(
        "\nRecord these with the device name before quoting any speedup "
        "(DESIGN.md §6, §8)."
    )
    return 0


def _row(cfg, ctx, size: int, steps: int) -> str:
    configs = [cfg] * size
    parts = [assemble(cfg, ctx) for _ in range(size)]
    net_name = parts[0].problem.solution_net
    points, offsets = _stack_points(parts, configs, ctx)
    residual = _population_residual(configs, net_name, offsets, ctx)
    lr = cfg.stages[0].optimizers[0].lr

    nets = [part.nets[net_name] for part in parts]
    weights = [
        {k: v.detach().clone() for k, v in net.state_dict().items()} for net in nets
    ]

    batched_seconds, batched = _time(
        lambda: train_population(
            Ensemble(nets), points, residual, steps=steps, lr=lr
        )
    )

    # Same starting weights for the second arm, or it measures a different
    # optimisation problem.
    for net, state in zip(nets, weights, strict=True):
        net.load_state_dict(state)

    # A residual per candidate, not the population's: the per-term scales in
    # _population_residual are (P, 1), so handing a one-candidate call the
    # population's residual would broadcast into P rows. Building one each is
    # also the honest comparison — it is what evaluating a candidate alone
    # actually costs.
    singles = [
        _population_residual([configs[i]], net_name, offsets, ctx)
        for i in range(len(nets))
    ]
    separate_seconds, separate = _time(
        lambda: torch.stack(
            [
                train_population(
                    Ensemble([net]),
                    points[index : index + 1],
                    singles[index],
                    steps=steps,
                    lr=lr,
                ).losses[0]
                for index, net in enumerate(nets)
            ]
        )
    )

    speedup = separate_seconds / batched_seconds if batched_seconds else float("nan")
    return (
        f"{batched_seconds:>12.2f} {separate_seconds:>13.2f} {speedup:>8.2f}x  "
        f"{float(batched.losses.mean()):.6g} vs {float(separate.mean()):.6g}"
    )


def _time(fn):
    """Wall clock around ``fn``, with CUDA's queue drained on both sides."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start, result


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:1")
    parser.add_argument("--dtype", default=None, choices=["float32", "float64"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
