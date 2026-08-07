"""The Run object — one append-only directory per training run.

CLAUDE.md rule 6 (``results/`` is append-only) is enforced in code, not by
discipline: creating a run refuses an existing directory, the trace is only ever
opened for append, and ``result.json`` is written with exclusive-create so a
second ``finish()`` raises instead of quietly rewriting history.

Directory layout::

    <root>/<run_id>/
        config.yaml       the validated config, as loaded
        config.json       the same, JSON, for machine reads
        provenance.json   first session's provenance
        sessions.jsonl    one line per session (Kaggle runs span several)
        failures.jsonl    one line per crash; a crash need not end the run
        trace.jsonl       downsampled convergence trace, append-only
        result.json       the ResultRow; written exactly once
        checkpoints/      best.pt / last.pt

Aggregation reads these raw files and writes derived tables to ``analysis/``
(DESIGN.md §8). Parquet compaction deliberately does not live here: parquet is
not appendable, so it belongs on the derived side of that line.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from pinnslab.registry.config import RunConfig, dump_config
from pinnslab.registry.provenance import collect_provenance, utc_now
from pinnslab.registry.schema import (
    Provenance,
    ResultRow,
    RunStatus,
    TracePoint,
    json_safe,
)
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_YAML = "config.yaml"
CONFIG_JSON = "config.json"
PROVENANCE_JSON = "provenance.json"
SESSIONS_JSONL = "sessions.jsonl"
FAILURES_JSONL = "failures.jsonl"
TRACE_JSONL = "trace.jsonl"
RESULT_JSON = "result.json"
CHECKPOINT_DIR = "checkpoints"


def make_run_id(cfg: RunConfig) -> str:
    """Sortable, self-describing, collision-free run identifier."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{stamp}Z_{cfg.identity_hash()[:8]}_s{cfg.seed}_{uuid.uuid4().hex[:6]}"


class Run:
    """Owns one run directory. Construct via :meth:`create` or :meth:`resume`."""

    def __init__(
        self,
        *,
        run_id: str,
        path: Path,
        cfg: RunConfig,
        provenance: Provenance,
        resumed: bool,
    ) -> None:
        self.run_id = run_id
        self.path = path
        self.cfg = cfg
        self.config_hash = cfg.identity_hash()
        self.provenance = provenance
        self.resumed = resumed
        self._t0 = time.perf_counter()

    # -- construction ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        cfg: RunConfig,
        root: str | Path,
        *,
        run_id: str | None = None,
    ) -> Run:
        """Start a fresh run. Raises if the directory already exists."""
        root = Path(root)
        run_id = run_id or make_run_id(cfg)
        path = root / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                f"run directory {path} already exists; results/ is append-only "
                "(CLAUDE.md rule 6). Use Run.resume() to continue it, or pick a "
                "new run_id."
            ) from None

        prov = collect_provenance(seed=cfg.seed, dtype=cfg.dtype)
        dump_config(cfg, path / CONFIG_YAML)
        _write_json(path / CONFIG_JSON, cfg.to_dict(), exclusive=True)
        _write_json(
            path / PROVENANCE_JSON, prov.model_dump(mode="json"), exclusive=True
        )
        (path / CHECKPOINT_DIR).mkdir(exist_ok=True)

        run = cls(run_id=run_id, path=path, cfg=cfg, provenance=prov, resumed=False)
        run._append_session("create")
        log.info("run %s created at %s (config %s)", run_id, path, run.config_hash)
        return run

    @classmethod
    def resume(cls, root: str | Path, run_id: str, cfg: RunConfig) -> Run:
        """Reattach to an existing, unfinished run (the Kaggle session-death path)."""
        path = Path(root) / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"no run directory at {path}")
        if (path / RESULT_JSON).exists():
            raise FileExistsError(
                f"run {run_id} already has {RESULT_JSON}; a finished run is "
                "immutable. Start a new run instead."
            )

        stored = json.loads((path / CONFIG_JSON).read_text(encoding="utf-8"))
        stored_hash = RunConfig(**stored).identity_hash()
        if stored_hash != cfg.identity_hash():
            raise ValueError(
                f"config mismatch resuming {run_id}: on disk {stored_hash}, "
                f"supplied {cfg.identity_hash()}. Resuming a run under a "
                "different condition would silently corrupt the result."
            )

        prov = collect_provenance(seed=cfg.seed, dtype=cfg.dtype)
        original = Provenance(
            **json.loads((path / PROVENANCE_JSON).read_text(encoding="utf-8"))
        )
        if prov.gpu_name != original.gpu_name:
            log.warning(
                "run %s started on %r but is resuming on %r; a comparison group "
                "must not span hardware (DESIGN.md §5) — this row will not be "
                "usable alongside its siblings",
                run_id,
                original.gpu_name,
                prov.gpu_name,
            )

        run = cls(run_id=run_id, path=path, cfg=cfg, provenance=original, resumed=True)
        run._append_session("resume", current=prov)
        log.info("run %s resumed at %s", run_id, path)
        return run

    @classmethod
    def create_or_resume(cls, cfg: RunConfig, root: str | Path, run_id: str) -> Run:
        """Idempotent entry point for a queue-driven notebook (DESIGN.md §7)."""
        if (Path(root) / run_id).is_dir():
            return cls.resume(root, run_id, cfg)
        return cls.create(cfg, root, run_id=run_id)

    # -- paths ----------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        return self.path / CHECKPOINT_DIR

    @property
    def is_finished(self) -> bool:
        return (self.path / RESULT_JSON).exists()

    # -- writing --------------------------------------------------------------

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        stage: int = 0,
        wall_time: float | None = None,
    ) -> None:
        """Append one downsampled trace point. The caller owns the schedule."""
        point = TracePoint(
            step=step,
            stage=stage,
            wall_time=(
                time.perf_counter() - self._t0 if wall_time is None else wall_time
            ),
            metrics={k: float(v) for k, v in metrics.items()},
        )
        _append_jsonl(self.path / TRACE_JSONL, point.model_dump(mode="json"))

    def log_failure(self, exc: BaseException, *, step: int) -> None:
        """Append a crash record. Deliberately does **not** finish the run.

        A run that raised is not necessarily over: if it has a checkpoint it can
        be resumed next session, which is the entire point of that machinery
        (DESIGN.md §7). Recording the crash in ``result.json`` would finalise the
        run and throw that compute away, so it goes to an append-only side file
        instead — a run may legitimately crash several times before it finishes.

        ``load_runs(include_unfinished=True)`` reads the last of these back as
        ``status=failed``, which is how a crash reaches the failure rate.
        """
        _append_jsonl(
            self.path / FAILURES_JSONL,
            {"at": utc_now(), "step": step, "error": f"{type(exc).__name__}: {exc}"},
        )

    def finish(
        self,
        *,
        status: RunStatus,
        steps_completed: int,
        final_metrics: dict[str, float] | None = None,
        best_metrics: dict[str, float] | None = None,
        timings: dict[str, float] | None = None,
        error: str | None = None,
        include_config: bool = True,
    ) -> ResultRow:
        """Write ``result.json`` exactly once and return the row.

        A diverged or failed run gets a row too — failure rate is a reported
        metric (DESIGN.md §11), so this is never skipped on the error path.
        """
        row = ResultRow.from_provenance(
            run_id=self.run_id,
            config_hash=self.config_hash,
            status=status,
            prov=self.provenance,
            steps_completed=steps_completed,
            final_metrics=final_metrics or {},
            best_metrics=best_metrics or {},
            timings=timings or {},
            tags=dict(self.cfg.tags),
            config=self.cfg.to_dict() if include_config else None,
            error=error,
        )
        _write_json(
            self.path / RESULT_JSON, row.model_dump(mode="json"), exclusive=True
        )
        log.info(
            "run %s finished: status=%s steps=%d %s",
            self.run_id,
            status.value,
            steps_completed,
            final_metrics or {},
        )
        return row

    def read_trace(self) -> list[TracePoint]:
        """The trace so far, tolerating the truncated final line a SIGKILL leaves.

        Goes through :func:`_read_jsonl` for the same reason :func:`load_runs`
        does: this is most often called on a resumed run, i.e. on precisely the
        file some earlier session was killed in the middle of writing.
        """
        return [TracePoint(**record) for record in _read_jsonl(self.path / TRACE_JSONL)]

    def _append_session(self, event: str, current: Provenance | None = None) -> None:
        prov = current or self.provenance
        _append_jsonl(
            self.path / SESSIONS_JSONL,
            {"event": event, "at": utc_now(), **prov.model_dump(mode="json")},
        )


def load_runs(root: str | Path, *, include_unfinished: bool = False) -> list[ResultRow]:
    """Every finished row under ``root``. Unfinished runs are skipped.

    ``include_unfinished=True`` additionally synthesises a row for each
    directory that has no ``result.json``: ``status=failed`` if it recorded a
    crash, ``status=running`` otherwise. A killed Kaggle session gets no chance
    to write anything on its way out, so those runs are otherwise invisible —
    and a failure rate computed only over the runs that survived long enough to
    report one is not the failure rate (DESIGN.md §11). **Pass this whenever the
    number you are computing is a rate.**
    """
    root = Path(root)
    if not root.is_dir():
        return []
    rows: list[ResultRow] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        result = directory / RESULT_JSON
        if result.exists():
            rows.append(ResultRow(**json.loads(result.read_text(encoding="utf-8"))))
        elif include_unfinished:
            row = _unfinished_row(directory)
            if row is not None:
                rows.append(row)
    return rows


def _unfinished_row(directory: Path) -> ResultRow | None:
    """A row for a run that never reached :meth:`Run.finish`.

    ``failed`` if it left a crash record, ``running`` if it simply stopped
    existing — the difference between a run that broke and a session that was
    killed is worth keeping, and only the first leaves evidence.
    """
    try:
        prov = Provenance(
            **json.loads((directory / PROVENANCE_JSON).read_text(encoding="utf-8"))
        )
        cfg = RunConfig(
            **json.loads((directory / CONFIG_JSON).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        # A session killed inside create() leaves a directory with neither file,
        # and it is not a run — it never got as far as having a condition.
        log.warning("skipping unreadable run directory %s: %s", directory, exc)
        return None

    step, metrics = _last_trace_point(directory)
    failure = _last_failure(directory)
    return ResultRow.from_provenance(
        run_id=directory.name,
        config_hash=cfg.identity_hash(),
        status=RunStatus.FAILED if failure else RunStatus.RUNNING,
        prov=prov,
        steps_completed=step,
        final_metrics=metrics,
        tags=dict(cfg.tags),
        error=failure or f"no {RESULT_JSON}; the session ended before finish() ran",
    )


def _last_failure(directory: Path) -> str | None:
    """The most recent crash record, if the run ever crashed."""
    records = _read_jsonl(directory / FAILURES_JSONL)
    if not records:
        return None
    last = records[-1]
    return f"{last.get('error', 'unknown error')} (at step {last.get('step', '?')})"


def _last_trace_point(directory: Path) -> tuple[int, dict[str, float]]:
    """How far an unfinished run actually got, from its append-only trace."""
    records = _read_jsonl(directory / TRACE_JSONL)
    if not records:
        return 0, {}
    try:
        point = TracePoint(**records[-1])
    except ValueError:
        return 0, {}
    return point.step, point.metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse an append-only file, skipping any record that does not parse.

    A process killed mid-``write`` is the expected way these files end, so a
    partial record is data loss to report, not a reason to fail the whole load.

    The partial record is **not** always the last one. :func:`_append_jsonl`
    opens with ``"a"``, so the next session's first line is appended straight
    onto the stump the killed session left, fusing the two into one unparseable
    line with perfectly good records after it. Skipping costs that fused line;
    stopping at it would silently discard everything the resumed session wrote,
    which for a run killed early and then trained to completion is nearly all of
    it — and it would reach :func:`_last_trace_point` as a plausible-looking but
    badly understated ``steps_completed``.

    Skipping is safe because corruption here is bounded: writes are one line at a
    time, fsynced, append-only, so a torn record can only ever be the tail of the
    file at the moment some session died. It cannot cascade.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        log.warning(
            "skipped %d unparseable line(s) in %s and kept %d record(s); "
            "a session was killed mid-write",
            skipped,
            path,
            len(records),
        )
    return records


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    try:
        with path.open(mode, encoding="utf-8") as fh:
            # allow_nan=False is the assertion behind json_safe: if a non-finite
            # float ever escapes it, fail loudly rather than write a file that
            # only Python can parse.
            json.dump(json_safe(payload), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
    except FileExistsError:
        raise FileExistsError(
            f"{path} already exists and results are append-only (CLAUDE.md rule 6)"
        ) from None


def _append_jsonl(path: Path, payload: Any) -> None:
    line = json.dumps(json_safe(payload), sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


__all__ = ["Run", "load_runs", "make_run_id"]
