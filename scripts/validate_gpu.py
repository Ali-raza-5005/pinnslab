"""Validate pinnslab on a real GPU, and print a report you can paste back.

    python scripts/validate_gpu.py                       # everything
    python scripts/validate_gpu.py --skip speedup        # the fast subset
    python scripts/validate_gpu.py --json report.json    # machine-readable too

Why this script exists
----------------------
README.md's honest limit is that **no GPU has ever run this code**. Every claim
that depends on one is therefore unverified: DESIGN.md §5's determinism switches
(``use_deterministic_algorithms`` refuses some CUDA kernels outright, and a run
that dies two hours into a Kaggle session because of it has cost two hours),
§5's precision-by-GPU rule (T4 FP64 ~1/32 of FP32 — a ratio, i.e. a
measurement), §6's "20-50x on a T4" for the batched population evaluator, and
§7's two-GPU strategy.

None of those can be closed from a laptop, and none should be quoted in a paper
until they are. So this is one command that closes them wherever a GPU turns up,
and prints a report whose numbers go into LOG.md and DESIGN.md.

It validates rather than benchmarks: every check either passes, fails with the
reason, or reports a number that was previously a guess.

On Kaggle
---------
Add a cell to a GPU notebook (T4 x2 or P100)::

    !pip install -q git+https://github.com/Ali-raza-5005/pinnslab@<tag>
    !git clone -q https://github.com/Ali-raza-5005/pinnslab /kaggle/working/src
    !cd /kaggle/working/src && python scripts/validate_gpu.py \
        --json /kaggle/working/gpu_report.json

then paste the printed report back. ``--two-gpu`` additionally re-executes this
script once per visible device to check §7's "two independent configs, one per
GPU" concurrency; it is skipped automatically on a single-GPU machine.

What CANNOT be validated here
-----------------------------
Session death. The queue's kill-resilience is pinned on CPU by
``tests/unit/test_queue_survives_a_killed_session.py`` with a real ``os._exit``;
reproducing that on Kaggle means letting a session actually expire, which is a
manual experiment rather than a script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)
import torch

import pinnslab
from pinnslab.registry.config import RunConfig, load_config
from pinnslab.registry.run import Run
from pinnslab.search.evaluate import _population_residual, _stack_points
from pinnslab.search.population import Ensemble, train_population
from pinnslab.training.build import assemble, build_trainer
from pinnslab.utils.device import configure_runtime, device_profile, gpu_name

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "configs" / "burgers_smooth.yaml"

#: The golden config's CPU float64 result, from tests/golden/test_burgers.py.
#: A GPU run is not expected to match it bit for bit — different kernels reduce
#: in a different order — but a GPU that lands an order of magnitude away is
#: reporting a different experiment, not a rounding difference.
CPU_REL_L2 = 6e-4
REL_L2_CEILING = 2e-3

CHECKS = ("environment", "determinism", "precision", "accuracy", "resume", "speedup")


# -- the checks ----------------------------------------------------------------


def check_environment() -> dict[str, Any]:
    """What hardware this is, and whether the library agrees about it.

    ``device_profile`` is recorded on every result row and aggregation refuses
    to plot a group that spans more than one value of it (DESIGN.md §5), so a
    profile string that is wrong here is a fairness check that silently passes
    everywhere.
    """
    report: dict[str, Any] = {
        "pinnslab": pinnslab.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": gpu_name(),
        "device_profile": device_profile(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        report["gpu_memory_total_gb"] = round(total / 1024**3, 2)
        report["gpu_memory_free_gb"] = round(free / 1024**3, 2)
        report["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return report


def check_determinism(dtype: str) -> dict[str, Any]:
    """Two runs of one ``(config, seed)`` must agree **bit for bit**.

    The check DESIGN.md §5 spends throughput to buy, and the one that has never
    run on CUDA. ``torch.use_deterministic_algorithms(True)`` raises rather than
    degrades when a kernel has no deterministic implementation, so the
    interesting outcome here is not a small difference — it is an exception,
    naming the op, hours into a session.
    """
    cfg = _config(dtype=dtype, steps=40)
    first = _train_to_parameters(cfg, "determinism-a")
    second = _train_to_parameters(cfg, "determinism-b")

    drift = max(
        float((a - b).abs().max()) for a, b in zip(first, second, strict=True)
    )
    return {
        "dtype": dtype,
        "max_parameter_difference": drift,
        "bit_exact": drift == 0.0,
        "ok": drift == 0.0,
    }


def check_precision() -> dict[str, Any]:
    """Measure the FP64/FP32 cost ratio that DESIGN.md §5's rule is built on.

    §5 routes float32 work to the T4 and float64 work to the P100 on the
    strength of "T4 FP64 is ~1/32 of FP32, P100 ~1/2". Those are vendor numbers
    about raw FLOPs; what matters here is the ratio on *this* workload, where a
    tiny MLP is usually launch-bound rather than FLOP-bound and the gap can be
    far smaller than 32x. Whichever it is, the precision-by-GPU rule should
    follow the measurement.
    """
    out: dict[str, Any] = {}
    for dtype in ("float32", "float64"):
        seconds = _time(_timed_train(_config(dtype=dtype, steps=100), dtype))[0]
        out[f"{dtype}_seconds"] = round(seconds, 3)
    ratio = out["float64_seconds"] / out["float32_seconds"]
    out["fp64_slowdown"] = round(ratio, 2)
    out["ok"] = True  # a measurement, not a pass/fail
    return out


def _timed_train(cfg: RunConfig, tag: str):
    """A zero-argument thunk for :func:`_time`, with its config bound now.

    A ``lambda`` over the loop variable would close over the *name*, so a
    future edit that deferred the call would time the last iteration twice.
    """
    return lambda: _train_to_parameters(cfg, f"precision-{tag}")


def check_accuracy(dtype: str) -> dict[str, Any]:
    """The golden Burgers config, end to end, on the GPU.

    The whole config -> network -> residual -> metric path, against a Cole-Hopf
    reference. If autograd, the geometry adapter or the eval grid behaves
    differently on CUDA, this is where it shows up as a number rather than as a
    silently different experiment.
    """
    cfg = load_config(GOLDEN).model_copy(update={"device": "auto", "dtype": dtype})
    ctx = configure_runtime(cfg)
    root = _scratch("accuracy")
    row = build_trainer(cfg, ctx, Run.create(cfg, root, run_id=f"golden-{dtype}")).fit()
    rel_l2 = float(row.best_metrics.get("rel_l2", float("nan")))
    return {
        "dtype": dtype,
        "rel_l2": rel_l2,
        "cpu_reference": CPU_REL_L2,
        "ceiling": REL_L2_CEILING,
        "train_seconds": round(row.timings.get("train_seconds", 0.0), 2),
        "status": row.status.value,
        "gpu_name": row.gpu_name,
        "device_profile": row.device_profile,
        "ok": rel_l2 < REL_L2_CEILING,
    }


class _Killed(RuntimeError):
    """Stands in for a Kaggle session ending. Raised from inside the loop."""


def check_resume(dtype: str) -> dict[str, Any]:
    """Interrupt a run, resume it, and demand the parameters match exactly.

    The same shape as ``tests/unit/test_checkpoint.py::test_resume_is_bit_exact``
    — kill mid-stage, re-issue the identical command, compare against an
    uninterrupted run — but on CUDA, which that test cannot reach. Two code
    paths here are dead to the CPU suite and exist precisely to stop a run
    silently continuing as a different experiment (TESTS_TODO.md):
    ``capture_rng_state`` storing ``torch.cuda.get_rng_state_all()``, and
    ``restore_rng_state`` refusing a device-count mismatch.

    A GPU that resumes to anything other than a bit-identical state makes every
    multi-session Kaggle run unreproducible, and nothing in the metrics says so.
    """
    cfg = _config(dtype=dtype, steps=60)
    root = _scratch("resume")

    reference = _train_at(cfg, root, "uninterrupted")

    # Killed at step 30 — after several checkpoints, mid-stage.
    try:
        _train_at(cfg, root, "killed", die_at=30)
    except _Killed:
        pass
    else:
        return {"ok": False, "error": "the injected kill never fired"}

    resumed = _train_at(cfg, root, "killed")

    drift = max(
        float((a - b).abs().max())
        for a, b in zip(reference, resumed, strict=True)
    )
    return {
        "dtype": dtype,
        "max_parameter_difference": drift,
        "bit_exact": drift == 0.0,
        "ok": drift == 0.0,
    }


def check_speedup(sizes: list[int], steps: int, dtype: str) -> dict[str, Any]:
    """Close DESIGN.md §6's open number: is batching worth 20-50x on a T4?

    Measured on CPU at 1.7x (P=4), 2.8x (P=8), 3.4x (P=16), ~2.2x (P=50) — the
    curve is not monotone, so the population size is part of the answer and is
    reported alongside it. An unverified speedup is a hole in the
    compute-parity defence of §8, so whatever comes out here is what a paper is
    allowed to say.
    """
    cfg = load_config(ROOT / "examples" / "configs" / "burgers_uniform.yaml")
    cfg = cfg.model_copy(
        update={
            "device": "auto",
            "dtype": dtype,
            "stages": [cfg.stages[0].model_copy(update={"steps": steps,
                                                        "resample_every": None})],
        }
    )
    results = {}
    for size in sizes:
        results[str(size)] = _one_size(cfg, size, steps)
    return {"dtype": dtype, "steps": steps, "by_population_size": results, "ok": True}


def _one_size(cfg: RunConfig, size: int, steps: int) -> dict[str, Any]:
    """One population size, batched against one-at-a-time, same weights."""
    configs = [cfg.model_copy(update={"seed": s}) for s in range(size)]
    ctx = None
    parts = []
    for candidate in configs:
        ctx = configure_runtime(candidate)
        parts.append(assemble(candidate, ctx))
    nets = [p.nets["u"] for p in parts]
    weights = [{k: v.clone() for k, v in n.state_dict().items()} for n in nets]
    points, offsets = _stack_points(parts, configs, ctx)
    residual = _population_residual(configs, "u", offsets, ctx)
    lr = configs[0].stages[0].optimizers[0].lr

    batched_seconds, _ = _time(
        lambda: train_population(Ensemble(nets), points, residual, steps=steps, lr=lr)
    )

    # Same starting weights for the second arm, or it measures a different
    # optimisation problem.
    for net, state in zip(nets, weights, strict=True):
        net.load_state_dict(state)

    # One residual per candidate: the population's carries (P, 1) per-term
    # scales that would broadcast a single-candidate call into P rows.
    singles = [
        _population_residual([configs[i]], "u", offsets, ctx) for i in range(len(nets))
    ]
    separate_seconds, _ = _time(
        lambda: [
            train_population(
                Ensemble([net]), points[i : i + 1], singles[i], steps=steps, lr=lr
            )
            for i, net in enumerate(nets)
        ]
    )
    return {
        "batched_seconds": round(batched_seconds, 3),
        "separate_seconds": round(separate_seconds, 3),
        "speedup": round(separate_seconds / batched_seconds, 2)
        if batched_seconds
        else None,
    }


def check_two_gpu() -> dict[str, Any]:
    """DESIGN.md §7: two independent configs, one per GPU, no DDP.

    Re-executes this script's ``--single-gpu-probe`` on each visible device in
    parallel. What it proves is narrow and is exactly what §7 needs: that two
    concurrent pinnslab processes each pinned to one device both finish, which
    is the whole of the two-GPU strategy (the queue's static partitioning
    guarantees they never touch the same cell, and that is pinned on CPU).
    """
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < 2:
        return {"skipped": f"needs 2 GPUs, this machine has {count}", "ok": True}

    started = time.perf_counter()
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--single-gpu-probe"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for device in range(count)
    ]
    outputs = [p.communicate()[0] for p in processes]
    codes = [p.returncode for p in processes]
    return {
        "devices": count,
        "exit_codes": codes,
        "seconds": round(time.perf_counter() - started, 2),
        "output_tail": [
            o.strip().splitlines()[-1] if o.strip() else "" for o in outputs
        ],
        "ok": all(code == 0 for code in codes),
    }


# -- helpers -------------------------------------------------------------------


def _config(*, dtype: str, steps: int) -> RunConfig:
    """A small, honest Burgers run: real residuals, real geometry, few steps."""
    cfg = load_config(GOLDEN)
    return cfg.model_copy(
        update={
            "device": "auto",
            "dtype": dtype,
            "stages": [cfg.stages[0].model_copy(update={"steps": steps})],
            "checkpoint": cfg.checkpoint.model_copy(
                update={"every_seconds": None, "every_steps": 10}
            ),
        }
    )


def _train_to_parameters(cfg: RunConfig, tag: str) -> list[torch.Tensor]:
    return _train_at(cfg, _scratch(tag), tag)


def _train_at(
    cfg: RunConfig, root: Path, run_id: str, *, die_at: int | None = None
) -> list[torch.Tensor]:
    """Train ``cfg`` in ``root/run_id``, resuming it if it is already there.

    ``die_at`` injects a session death through ``eval_fn``, which the trainer
    calls on the trace schedule — the cheapest hook into the loop that does not
    require the trainer to know it is being tested.
    """
    ctx = configure_runtime(cfg)
    run = Run.create_or_resume(cfg, root, run_id)

    def eval_fn(state):
        if die_at is not None and state.step >= die_at:
            raise _Killed(f"session ended at step {state.step}")
        return {}

    trainer = build_trainer(cfg, ctx, run)
    if die_at is not None:
        # Replaced rather than passed: build_trainer supplies the benchmark's
        # own eval_fn, so handing it a second one is a duplicate keyword.
        trainer.eval_fn = eval_fn
    trainer.fit()
    return [p.detach().cpu().clone() for p in trainer.nets["u"].parameters()]


def _scratch(tag: str) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix=f"pinnslab_gpu_{tag}_"))


def _time(fn):
    """Wall clock around ``fn``, with CUDA's queue drained on both sides.

    Without the synchronise a CUDA timing measures how fast work was *queued*,
    which for tiny kernels is most of the apparent speedup.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start, result


# -- driver --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    if args.single_gpu_probe:
        # The child half of check_two_gpu: one short real run, pinned to
        # whichever device CUDA_VISIBLE_DEVICES left visible.
        result = check_accuracy(args.dtype)
        print(f"probe on {result['gpu_name']}: rel_l2={result['rel_l2']:.3g}")
        return 0 if result["ok"] else 1

    report: dict[str, Any] = {"checks": {}}
    report["checks"]["environment"] = check_environment()

    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            "CUDA is not available, so nothing here would validate what it "
            "claims to. Run this on a GPU machine (Kaggle T4 x2 or P100), or "
            "pass --allow-cpu to smoke-test the script itself.",
            file=sys.stderr,
        )
        _emit(report, args)
        return 1

    plan = [c for c in CHECKS if c != "environment" and c not in args.skip]
    for name in plan:
        print(f"\n=== {name} ===", flush=True)
        try:
            if name == "speedup":
                result = check_speedup(args.sizes, args.speedup_steps, args.dtype)
            elif name == "precision":
                result = check_precision()
            else:
                result = globals()[f"check_{name}"](args.dtype)
        except Exception as exc:  # noqa: BLE001 - a failed check is the finding
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["checks"][name] = result
        print(json.dumps(result, indent=2, default=str), flush=True)

    if args.two_gpu:
        print("\n=== two_gpu ===", flush=True)
        report["checks"]["two_gpu"] = check_two_gpu()
        print(json.dumps(report["checks"]["two_gpu"], indent=2), flush=True)

    failed = [n for n, r in report["checks"].items() if r.get("ok") is False]
    report["failed"] = failed
    _emit(report, args)

    print("\n" + "=" * 70)
    print("PASTE EVERYTHING BELOW THIS LINE BACK")
    print("=" * 70)
    print(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


def _emit(report: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {args.json}", file=sys.stderr)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=CHECKS, help="checks to leave out"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="float32 by default: DESIGN.md §5 routes T4 work to float32, and "
        "the T4 is the machine this most needs to run on",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--speedup-steps", type=int, default=25)
    parser.add_argument("--two-gpu", action="store_true")
    parser.add_argument("--json", default=None, help="also write the report here")
    parser.add_argument(
        "--single-gpu-probe", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run the checks on CPU. They validate nothing about a GPU there — "
        "this exists so the script itself can be smoke-tested before it is "
        "carried to Kaggle, where a crash costs a session.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
