"""Seeding and determinism (DESIGN.md §5).

Determinism costs throughput. We take the cost: a PINN result that cannot be
reproduced bit-for-bit cannot be defended in rebuttal.

Two responsibilities live here:

1. :func:`set_seed` — put the process into a deterministic state at run start.
2. :func:`capture_rng_state` / :func:`restore_rng_state` — round-trip every RNG
   stream through a checkpoint. A resumed run that continues from a *fresh* RNG
   is not the same experiment as the uninterrupted one, and the difference is
   invisible in the metrics.

Samplers should draw from an explicit :func:`make_generator` rather than the
global RNG, so that resume reproducibility does not depend on how many global
draws happened to occur before them.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

#: Required by cuBLAS for deterministic reductions; must be set before the CUDA
#: context is created (DESIGN.md §5).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

#: numpy's legacy seeder and torch's CPU generator differ in accepted range;
#: this bound is safe for both.
_SEED_MAX = 2**31


def set_seed(seed: int, *, deterministic: bool = True, warn_only: bool = False) -> None:
    """Seed every RNG stream and switch torch into deterministic mode.

    Call once per run, before any tensor is created and before CUDA is touched.

    Raises:
        RuntimeError: if CUDA is already initialised and the cuBLAS workspace
            config is not set. Setting it after context creation silently does
            nothing, which would leave reductions nondeterministic while every
            log line claims otherwise.
    """
    if not 0 <= seed < 2**63:
        raise ValueError(f"seed must be a non-negative int64, got {seed!r}")

    if deterministic:
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing != CUBLAS_WORKSPACE_CONFIG:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "CUDA is already initialised but CUBLAS_WORKSPACE_CONFIG is "
                    f"{existing!r}. Setting it now would be silently ignored and "
                    "cuBLAS reductions would stay nondeterministic. Set "
                    f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} in the "
                    "environment before importing torch, or call set_seed() "
                    "earlier."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_seed(base: int, *tags: object) -> int:
    """Derive a stable sub-seed from a base seed and arbitrary tags.

    Deterministic across processes and Python versions (``hash()`` is not, which
    is why this exists). Use for per-stage / per-sampler / per-population-member
    streams so they do not interfere with one another.
    """
    payload = "|".join([str(base), *(str(t) for t in tags)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MAX


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """An explicit RNG stream, decoupled from the global one."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG stream this library can affect.

    The numpy state is stored as plain Python ints rather than an ``ndarray``:
    ``torch.load(..., weights_only=True)`` refuses numpy arrays on torch >= 2.6,
    and we want checkpoints to load without disabling that safety switch.
    """
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (
            np_state[0],
            [int(x) for x in np_state[1]],
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state`.

    CUDA state is restored only when the device count matches what was captured;
    a mismatch means the run moved hardware, which violates the
    hardware-uniformity rule (DESIGN.md §5) and is reported rather than papered
    over.
    """
    random.setstate(_as_nested_tuple(state["python"]))

    name, keys, pos, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state(
        (
            name,
            np.array(keys, dtype=np.uint32),
            int(pos),
            int(has_gauss),
            float(cached_gauss),
        )
    )

    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))

    cuda_state = state.get("cuda")
    if cuda_state is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "checkpoint carries CUDA RNG state but this process has no CUDA "
            "device; resuming would silently change the experiment"
        )
    if len(cuda_state) != torch.cuda.device_count():
        raise RuntimeError(
            f"checkpoint carries CUDA RNG state for {len(cuda_state)} device(s) "
            f"but this process sees {torch.cuda.device_count()}; a comparison "
            "group must not span hardware configurations (DESIGN.md §5)"
        )
    torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda_state])


def _as_nested_tuple(obj: Any) -> Any:
    """``random.setstate`` demands tuples; serialisation may hand back lists."""
    if isinstance(obj, (list, tuple)):
        return tuple(_as_nested_tuple(x) for x in obj)
    return obj


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "capture_rng_state",
    "derive_seed",
    "make_generator",
    "restore_rng_state",
    "set_seed",
]
