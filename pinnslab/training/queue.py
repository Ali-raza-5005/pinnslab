"""The run queue (DESIGN.md §7, §9 step 3) — many cells across dying sessions.

A Kaggle session is ephemeral and unannounced: it can be killed at any point,
including three seconds into a two-hour cell. The queue's job is to make that a
non-event, so a sweep is a list of work rather than a person watching a
notebook.

Status is derived, never written
--------------------------------
``run_matrix.csv`` is an **immutable declarative input**: a list of
``(config, seed)`` pairs and nothing else. A cell's ``run_id`` is a pure
function of that pair (:func:`run_id_for`), so the run directory *is* the
claim — "claimed" is "the directory exists", "done" is "it has
``result.json``". Nothing writes a status back.

DESIGN.md §7 originally described a mutable status column that the notebook
marked done. Deriving it instead is strictly better here, for three reasons:

* **CLAUDE.md rule 6 holds by construction.** There is no mutable experimental
  bookkeeping to overwrite, and no way for the queue's idea of what ran to
  disagree with what is actually on disk.
* **A killed session leaves no lie.** A status column written before the work
  strands rows in ``claimed`` forever; a column written after it loses the
  record of anything interrupted. The directory is correct either way, because
  it was created by the run itself.
* **Two GPUs never contend.** Combined with the static partitioning in
  :func:`select`, no two workers can look at the same cell, so claiming needs
  no lock, no lease and no heartbeat.

Why the matrix holds only ``(config, seed)``
--------------------------------------------
Every other axis gets its own YAML file. ``seed`` is the one field that may
live outside a config without breaking CLAUDE.md rule 4, because it is already
excluded from the config hash (a run is the pair ``(config_hash, seed)``,
DESIGN.md §4) and five seeds of one condition must share a hash for the
median-and-IQR groupby to have anything to group on. Allowing arbitrary
hyperparameter overrides in the CSV would reintroduce exactly the
number-in-a-script problem rule 4 exists to prevent, and those numbers would
not be hashed.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pinnslab.registry.config import RunConfig, load_config
from pinnslab.registry.run import (
    FAILURES_JSONL,
    RESULT_JSON,
    Run,
    _read_jsonl,
)
from pinnslab.registry.schema import ResultRow
from pinnslab.training.build import build_trainer
from pinnslab.utils.device import configure_runtime
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

#: Columns :func:`load_matrix` requires. ``notes`` is optional and ignored.
REQUIRED_COLUMNS = ("config", "seed")


class CellStatus(StrEnum):
    """What the results directory says about a cell. Derived, never stored."""

    #: No run directory. Nothing has been started.
    PENDING = "pending"
    #: A directory with no ``result.json``: started, not finished, resumable.
    #: This is what a killed session leaves behind.
    RESUMABLE = "resumable"
    #: Unfinished *and* it recorded at least one crash. Still resumable — a
    #: crash does not finish a run (``Run.log_failure``) — but the distinction
    #: is worth keeping, because "the session died" and "the run raised" are
    #: different problems.
    FAILED = "failed"
    #: ``result.json`` exists. Immutable; never claimed again.
    DONE = "done"


@dataclass(frozen=True)
class Cell:
    """One row of the run matrix: which config, which seed."""

    config: Path
    seed: int
    notes: str = ""

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed} for {self.config}")


@dataclass
class QueueReport:
    """What one session actually did. Printed by the notebook; not persisted —
    the run directories are the record (CLAUDE.md rule 6)."""

    completed: list[ResultRow] = field(default_factory=list)
    failed: list[tuple[Cell, str]] = field(default_factory=list)
    deferred: list[Cell] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{len(self.completed)} completed, {len(self.failed)} failed, "
            f"{len(self.deferred)} left for the next session"
        )


def load_matrix(path: str | Path) -> list[Cell]:
    """Read ``run_matrix.csv``.

    ``config`` paths are resolved relative to the CSV, so a matrix and the
    configs it names move between machines as one directory.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}; a cell is a "
            f"(config, seed) pair. Columns present: {sorted(rows[0])}"
        )

    cells = []
    for number, row in enumerate(rows, start=2):  # start=2: line 1 is the header
        try:
            cells.append(
                Cell(
                    config=(path.parent / row["config"].strip()).resolve(),
                    seed=int(row["seed"]),
                    notes=(row.get("notes") or "").strip(),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} line {number}: {exc}") from None
    log.info("loaded %d cells from %s", len(cells), path)
    return cells


def config_for(cell: Cell) -> RunConfig:
    """The cell's validated config, with the matrix's seed applied.

    The matrix's ``seed`` wins over whatever the YAML says: seeding is the
    axis the matrix exists to sweep, and a config file is a *condition*, which
    by DESIGN.md §4 does not include its seed.
    """
    return load_config(cell.config).model_copy(update={"seed": cell.seed})


def run_id_for(cfg: RunConfig, seed: int | None = None) -> str:
    """The deterministic run id for a ``(config, seed)`` pair.

    Deliberately not :func:`pinnslab.registry.run.make_run_id`, which is
    timestamped and uuid-suffixed and therefore cannot be recomputed by a later
    session — which is the entire mechanism here.

    Because the id is the identity, two matrix rows naming the same config and
    seed collapse onto one run instead of silently training it twice. It also
    means ``Run.resume``'s config-mismatch check can never fire from this path:
    the hash is in the directory name, so a changed YAML gets a new directory
    rather than corrupting the old one.
    """
    seed = cfg.seed if seed is None else seed
    return f"{cfg.identity_hash()[:12]}_s{seed}"


def status_of(root: str | Path, run_id: str) -> CellStatus:
    """Read a cell's status off the filesystem."""
    directory = Path(root) / run_id
    if not directory.is_dir():
        return CellStatus.PENDING
    if (directory / RESULT_JSON).exists():
        return CellStatus.DONE
    # _read_jsonl rather than a size check: a session killed mid-write leaves a
    # torn line, and that file is exactly the kind this function is asked about.
    if _read_jsonl(directory / FAILURES_JSONL):
        return CellStatus.FAILED
    return CellStatus.RESUMABLE


def select(
    cells: Sequence[Cell],
    root: str | Path,
    *,
    worker: int = 0,
    workers: int = 1,
) -> list[Cell]:
    """This worker's outstanding cells, most-urgent first.

    Workers partition the matrix statically by row index. That is DESIGN.md
    §7's "two independent configs concurrently, one per GPU via
    ``CUDA_VISIBLE_DEVICES``" — and it is what makes claiming lock-free: two
    workers never consider the same cell, so a stale claim cannot exist and no
    lease or heartbeat is needed to detect one.

    Unfinished cells come before untouched ones. Only started work has compute
    at risk, and on a platform where the session may end at any moment,
    finishing something is worth more than starting something.
    """
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError(f"need 0 <= worker < workers, got {worker}/{workers}")

    mine = [cell for index, cell in enumerate(cells) if index % workers == worker]
    order = {
        CellStatus.RESUMABLE: 0,
        CellStatus.FAILED: 1,
        CellStatus.PENDING: 2,
    }
    outstanding = []
    for cell in mine:
        status = status_of(root, run_id_for(config_for(cell)))
        if status is not CellStatus.DONE:
            outstanding.append((order[status], cell))
    return [cell for _, cell in sorted(outstanding, key=lambda pair: pair[0])]


def run_cell(
    cell: Cell, root: str | Path, *, allow_resampling: bool = False
) -> ResultRow:
    """Train one cell to completion, starting it or resuming it as required.

    Idempotent by construction: the run id is derived, so a second call on a
    finished cell would be refused by ``Run.resume``, and a second call on an
    interrupted one continues it from ``last.pt``.
    """
    cfg = config_for(cell)
    _check_resampling(cfg, cell, allow_resampling=allow_resampling)

    run_id = run_id_for(cfg)
    # configure_runtime first, always: it sets the seed and the default dtype,
    # and both must be in force before build_trainer allocates a parameter.
    ctx = configure_runtime(cfg)
    run = Run.create_or_resume(cfg, root, run_id)
    log.info(
        "cell %s seed=%d -> run %s (%s)",
        cell.config.name,
        cell.seed,
        run_id,
        "resuming" if run.resumed else "starting",
    )

    # Assembly runs after the directory exists but before ``fit`` can record
    # anything, so a config that cannot be built — an unregistered problem, a
    # bad selector, an OOM allocating the nets — would otherwise leave a
    # directory indistinguishable from one a session was killed in. That is a
    # failure *of that configuration* and belongs in the failure rate
    # (DESIGN.md §11). Only the build is wrapped: ``Trainer.fit`` already logs
    # its own crashes, and logging twice would report one failure as two.
    try:
        trainer = build_trainer(cfg, ctx, run)
    except Exception as exc:
        run.log_failure(exc, step=0)
        raise
    return trainer.fit()


def run_queue(
    cells: Sequence[Cell],
    root: str | Path,
    *,
    worker: int = 0,
    workers: int = 1,
    deadline_seconds: float | None = None,
    allow_resampling: bool = False,
) -> QueueReport:
    """Work through this worker's cells until they run out or time does.

    Every config is loaded and checked **before** the first cell trains. A
    typo'd YAML key that surfaces two hours into a session has cost two hours;
    the same typo caught at the top of the notebook costs nothing.

    A cell that raises is logged and the queue moves on. ``Trainer.fit`` has
    already written the crash to ``failures.jsonl`` and the failure is a
    reported metric (DESIGN.md §11), so one bad configuration must not take the
    rest of the sweep down with it.

    ``deadline_seconds`` stops the queue *claiming new* cells once the time
    left is less than the longest cell seen this session. It never interrupts a
    running one: checkpoint/resume already handles a hard kill exactly, so a
    soft stop would add a second mechanism for no gain.
    """
    started = time.monotonic()
    report = QueueReport()

    outstanding = select(cells, root, worker=worker, workers=workers)
    for cell in outstanding:
        _check_resampling(
            config_for(cell), cell, allow_resampling=allow_resampling
        )
    log.info(
        "worker %d/%d: %d outstanding cell(s) of %d in the matrix",
        worker,
        workers,
        len(outstanding),
        len(cells),
    )

    longest = 0.0
    for index, cell in enumerate(outstanding):
        if deadline_seconds is not None:
            remaining = deadline_seconds - (time.monotonic() - started)
            # Claim the first cell regardless: with nothing measured yet there
            # is no basis for declining, and never starting is its own failure.
            if index and remaining < longest:
                log.info(
                    "%.0fs left and the longest cell took %.0fs; leaving %d "
                    "cell(s) for the next session",
                    remaining,
                    longest,
                    len(outstanding) - index,
                )
                report.deferred.extend(outstanding[index:])
                break

        cell_started = time.monotonic()
        try:
            report.completed.append(
                run_cell(cell, root, allow_resampling=allow_resampling)
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the sweep
            log.exception("cell %s seed=%d failed", cell.config.name, cell.seed)
            report.failed.append((cell, f"{type(exc).__name__}: {exc}"))
        longest = max(longest, time.monotonic() - cell_started)

    log.info("session done: %s", report)
    return report


def statuses(
    cells: Iterable[Cell], root: str | Path
) -> list[tuple[Cell, CellStatus]]:
    """The whole matrix's derived status. For humans and for `analysis/`."""
    return [(cell, status_of(root, run_id_for(config_for(cell)))) for cell in cells]


def _check_resampling(
    cfg: RunConfig, cell: Cell, *, allow_resampling: bool
) -> None:
    """Refuse the one combination known to corrupt a resumed run silently.

    ``TrainState.scratch`` is not checkpointed, so a run killed at a step that
    is not a multiple of ``resample_every`` resumes holding the *initial* point
    cloud rather than the one it was training on. Nothing in the metrics shows
    it. The queue is precisely the machinery that makes runs interruptible, so
    this is where that would first bite — and it would bite paper 1, whose
    subject is sampling. Loud beats silent until the sampler-state design
    question in TESTS_TODO.md is settled.
    """
    if allow_resampling or not cfg.checkpoint.save_last:
        return
    stages = [s.name for s in cfg.stages if s.resample_every is not None]
    if stages:
        raise ValueError(
            f"{cell.config.name} sets resample_every on stage(s) {stages} and is "
            "checkpointed, but collocation points are not checkpointed: a "
            "resumed run would silently continue on the initial point cloud "
            "(see TESTS_TODO.md, training/trainer). Pass allow_resampling=True "
            "only if this run will not be interrupted."
        )


__all__ = [
    "Cell",
    "CellStatus",
    "QueueReport",
    "config_for",
    "load_matrix",
    "run_cell",
    "run_id_for",
    "run_queue",
    "select",
    "status_of",
    "statuses",
]
