"""Checkpoint / resume (DESIGN.md §5, §7, §11).

A Kaggle session dies without warning. Every run must therefore be resumable, and
resumption must be *bit-exact* — a resumed run that continues from a fresh RNG is
a different experiment wearing the same run_id, and nothing in the metrics will
tell you that happened. So the payload carries the RNG state of every stream
alongside the weights and optimizer state.

Two hazards handled here explicitly:

**Torn writes.** A process killed inside ``torch.save`` leaves a truncated file.
We write to a temporary path in the same directory and ``os.replace`` it, which
is atomic on POSIX and on Windows, so ``last.pt`` is either the old checkpoint or
the new one and never a corpse.

**L-BFGS.** ``torch.optim.LBFGS`` *does* carry its curvature history in
``state_dict`` — ``old_dirs``, ``old_stps``, ``ro``, ``H_diag``,
``prev_flat_grad``, ``d``, ``t`` — so a mid-L-BFGS resume is bit-exact and is
treated like any other stage. This was measured, not assumed, and the claim is
pinned by ``test_lbfgs_resume_is_bit_exact``: if a future torch moves that state
out of ``state_dict``, that test fails rather than the failure showing up as a
silently different experiment.

An earlier version of this module asserted the opposite and had the trainer
rewind an interrupted L-BFGS stage to its boundary. That threw away a whole
stage per session death, on the platform whose sessions die unpredictably.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch

import pinnslab
from pinnslab.registry.config import CheckpointSpec
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

LAST_NAME = "last.pt"
BEST_NAME = "best.pt"
#: 2 (2026-08-17): the payload gained ``points`` and ``sampler_state``. Version 1
#: checkpoints are refused rather than loaded with an empty cloud, because a
#: resampling run resumed without its cloud is the exact silent-wrong-experiment
#: this field exists to prevent.
_FORMAT_VERSION = 2


@dataclass
class CheckpointPayload:
    """Everything needed to continue a run as if it had never stopped."""

    step: int
    stage_index: int
    steps_in_stage: int
    nets: dict[str, dict[str, Any]]
    extra_params: dict[str, torch.Tensor]
    optimizers: list[dict[str, Any]]
    rng: dict[str, Any]
    elapsed: float
    config_hash: str
    seed: int
    #: The collocation cloud in force at ``step``, by group name. Empty for a
    #: run whose points never move — drawing them once from a checkpointed RNG
    #: stream reproduces them exactly — and populated for every run that
    #: resamples, where nothing else can reproduce them (see
    #: :class:`~pinnslab.training.trainer.TrainState.points`).
    points: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Whatever the resample hook reports through its own ``state_dict``:
    #: counters, residual EMAs, a growing pool. Empty for stateless samplers.
    sampler_state: dict[str, Any] = field(default_factory=dict)
    best_value: float | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_step: int | None = None
    timings: dict[str, float] = field(default_factory=dict)
    pinnslab_version: str = pinnslab.__version__
    format_version: int = _FORMAT_VERSION

    def without_optimizer_state(self) -> CheckpointPayload:
        """The same payload, minus the part only a *resume* needs.

        Weights, RNG, points, sampler state and provenance all stay: what is
        dropped is the optimizer's moments and L-BFGS's curvature history, which
        are meaningless without the step they were about to take.
        """
        return replace(self, optimizers=[])

    def to_dict(self) -> dict[str, Any]:
        """Field-by-field, deliberately **not** ``dataclasses.asdict``.

        ``asdict`` deep-copies everything it walks, so for a payload built from
        ``state_dict``s it duplicates the model and the optimizer state in memory
        on every periodic save — for nothing. :func:`save_checkpoint` serialises
        synchronously and no training step can run in between, so references are
        as good as a snapshot here.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointPayload:
        got = payload.get("format_version")
        if got != _FORMAT_VERSION:
            raise ValueError(
                f"checkpoint format version {got!r} != {_FORMAT_VERSION}; this "
                "checkpoint was written by a different pinnslab. Papers pin tags "
                "for exactly this reason (DESIGN.md §2)."
            )
        return cls(**payload)


def save_checkpoint(path: str | Path, payload: CheckpointPayload) -> None:
    """Atomically write a checkpoint. Never leaves a partial file at ``path``.

    The temporary file has a fixed name, so a ``SIGKILL`` between the write and
    the rename leaves one stale ``.tmp`` that the next save to this path simply
    overwrites. Only a run that never saves again can leave it behind for good,
    and that run's directory has bigger problems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            torch.save(payload.to_dict(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # An exception (OOM, disk full, KeyboardInterrupt) would otherwise leave
        # a half-written .tmp sitting next to a perfectly good checkpoint.
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable, not just the bytes it points at.

    ``os.replace`` is atomic, but on POSIX the directory entry is not on disk
    until the directory is synced — a power loss can therefore lose the rename
    while keeping both files. Windows has no directory file descriptor and needs
    no equivalent, so it is skipped rather than worked around.
    """
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> CheckpointPayload:
    """Load a checkpoint with ``weights_only=True``.

    The payload is built from tensors and plain Python containers precisely so
    that this stays true — see :func:`pinnslab.utils.seeding.capture_rng_state`,
    which converts numpy's state out of ``ndarray`` for the same reason.
    """
    raw = torch.load(path, map_location=map_location, weights_only=True)
    return CheckpointPayload.from_dict(raw)


class CheckpointManager:
    """Owns ``best.pt`` / ``last.pt`` for one run.

    Best + last only (DESIGN.md §11): every-N checkpoints are how storage
    explodes, and nothing downstream reads them.
    """

    def __init__(
        self,
        directory: str | Path,
        spec: CheckpointSpec,
        *,
        config_hash: str,
        seed: int,
        best_mode: str = "min",
    ) -> None:
        self.dir = Path(directory)
        self.spec = spec
        self.config_hash = config_hash
        self.seed = seed
        if best_mode not in ("min", "max"):
            raise ValueError(f"best_mode must be 'min' or 'max', got {best_mode!r}")
        self.best_mode = best_mode
        self._last_save_time = time.perf_counter()
        self._last_save_step = 0

    @property
    def last_path(self) -> Path:
        return self.dir / LAST_NAME

    @property
    def best_path(self) -> Path:
        return self.dir / BEST_NAME

    def is_improvement(self, value: float, best: float | None) -> bool:
        """Is ``value`` a new best? Non-finite values never are.

        Without the finiteness guard the first tracked value being NaN would set
        ``best`` to NaN permanently: every later comparison against NaN is
        ``False``, so nothing could ever displace it and ``best.pt`` would hold
        whatever parameters happened to blow up. Runs normally stop on a
        non-finite loss, so this is reachable only with
        ``stop_on_nonfinite=False`` — which is exactly the setting used to study
        runs that recover from a spike.
        """
        if not math.isfinite(value):
            return False
        if best is None or not math.isfinite(best):
            return True
        return value < best if self.best_mode == "min" else value > best

    def due(self, step: int) -> bool:
        """Has the periodic cadence elapsed? Boundary saves bypass this."""
        by_steps = self.spec.every_steps is not None and (
            step - self._last_save_step >= self.spec.every_steps
        )
        by_time = self.spec.every_seconds is not None and (
            time.perf_counter() - self._last_save_time >= self.spec.every_seconds
        )
        return by_steps or by_time

    def save_last(self, payload: CheckpointPayload) -> None:
        if not self.spec.save_last:
            return
        save_checkpoint(self.last_path, payload)
        self._last_save_time = time.perf_counter()
        self._last_save_step = payload.step

    def save_best(self, payload: CheckpointPayload) -> None:
        """``best.pt``, without the optimizer state.

        Nothing resumes from ``best.pt`` — resume is ``last.pt``, by definition,
        because best-so-far is not a point the run ever continued from. So the
        optimizer state in it is pure storage: Adam carries two moments per
        parameter, i.e. roughly two thirds of the file, written every time the
        metric improves. On a capped Kaggle working directory across a
        thousand-cell sweep that is the difference between finishing and running
        out of disk (DESIGN.md §11).
        """
        if not self.spec.save_best:
            return
        save_checkpoint(self.best_path, payload.without_optimizer_state())

    def load_last(
        self, *, allow_config_change: bool = False
    ) -> CheckpointPayload | None:
        """The resume entry point. ``None`` when there is nothing to resume."""
        if not self.last_path.exists():
            return None
        payload = load_checkpoint(self.last_path)
        self._verify(payload, allow_config_change=allow_config_change)
        return payload

    def _verify(self, payload: CheckpointPayload, *, allow_config_change: bool) -> None:
        mismatches = []
        if payload.config_hash != self.config_hash:
            mismatches.append(f"config {payload.config_hash} != {self.config_hash}")
        if payload.seed != self.seed:
            mismatches.append(f"seed {payload.seed} != {self.seed}")
        if not mismatches:
            return
        message = (
            f"refusing to resume {self.last_path}: {'; '.join(mismatches)}. "
            "A run is identified by (config_hash, seed); resuming across either "
            "would produce a row that describes neither condition."
        )
        if not allow_config_change:
            raise ValueError(message)
        log.warning("%s — proceeding because allow_config_change=True", message)


__all__ = [
    "BEST_NAME",
    "LAST_NAME",
    "CheckpointManager",
    "CheckpointPayload",
    "load_checkpoint",
    "save_checkpoint",
]
