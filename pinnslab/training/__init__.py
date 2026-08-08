"""The training loop we own (DESIGN.md §1): trainer, checkpoint/resume.

``build`` and ``queue`` are deliberately **not** re-exported here. Both reach
``benchmarks`` -> ``geometry`` -> deepxde, and ``import pinnslab.training``
should not drag that in: the trainer takes plain callables and must stay
importable — and testable — with no geometry, models or physics in the picture
at all (DESIGN.md §4's escape hatch). Import them by module:
``from pinnslab.training.queue import run_queue``.
"""

from pinnslab.training.checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from pinnslab.training.trainer import Trainer, TrainState

__all__ = [
    "CheckpointManager",
    "CheckpointPayload",
    "TrainState",
    "Trainer",
    "load_checkpoint",
    "save_checkpoint",
]
