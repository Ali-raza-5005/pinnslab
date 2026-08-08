"""The run queue (DESIGN.md §9 step 3).

These are the fast tests: matrix parsing, derived status, worker partitioning,
and the two behaviours a sweep depends on — a finished cell is never claimed
again, and one bad cell does not end the session. The property that actually
justifies the design (a killed session loses nothing and resumes bit-exactly)
needs a real SIGKILL and lives in
``test_queue_survives_a_killed_session.py``.

They run the real Burgers path on a shrunken config rather than a stub. A
stubbed run would pass on a queue that could not drive a run at all, which is
the only thing the queue does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pinnslab.registry.run import FAILURES_JSONL, RESULT_JSON, Run
from pinnslab.training.queue import (
    Cell,
    CellStatus,
    config_for,
    load_matrix,
    run_cell,
    run_id_for,
    run_queue,
    select,
    status_of,
    statuses,
)

pytestmark = pytest.mark.unit

CONFIGS = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
TINY = CONFIGS / "burgers_tiny.yaml"
RESAMPLED = CONFIGS / "burgers_tiny_resampled.yaml"
UNBUILDABLE = CONFIGS / "unbuildable.yaml"


def write_matrix(path: Path, rows: list[tuple[str, int]]) -> Path:
    """A run_matrix.csv naming configs by path relative to it."""
    lines = ["config,seed,notes"]
    lines += [f"{Path(config).as_posix()},{seed}," for config, seed in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def matrix(tmp_path) -> list[Cell]:
    """Three seeds of one condition — the DESIGN.md §8 shape of a real sweep."""
    csv = write_matrix(
        tmp_path / "run_matrix.csv", [(str(TINY), 0), (str(TINY), 1), (str(TINY), 2)]
    )
    return load_matrix(csv)


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """One completed sweep, trained once and read by several tests.

    Module-scoped on purpose: every training in this file is real, and three of
    them are most of the file's runtime against a 60s suite budget. Nothing
    below mutates it — a finished run directory is immutable (rule 6), which is
    what makes sharing it safe.
    """
    base = tmp_path_factory.mktemp("swept")
    cells = load_matrix(
        write_matrix(
            base / "run_matrix.csv", [(str(TINY), seed) for seed in (0, 1, 2)]
        )
    )
    root = base / "results"
    return cells, root, run_queue(cells, root)


# -- the matrix ----------------------------------------------------------------


def test_config_paths_resolve_relative_to_the_matrix(tmp_path):
    """A matrix and its configs must move between machines as one directory."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tiny.yaml").write_text(
        TINY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    csv = write_matrix(tmp_path / "run_matrix.csv", [("configs/tiny.yaml", 0)])

    (cell,) = load_matrix(csv)
    assert cell.config == (tmp_path / "configs" / "tiny.yaml").resolve()
    assert cell.config.is_file()


def test_a_missing_required_column_is_a_load_error(tmp_path):
    csv = tmp_path / "run_matrix.csv"
    csv.write_text("config,notes\nx.yaml,no seed here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_matrix(csv)


def test_a_bad_seed_names_the_line_it_is_on(tmp_path):
    """A 200-row matrix needs the line number, not just 'invalid literal'."""
    csv = tmp_path / "run_matrix.csv"
    csv.write_text(
        f"config,seed\n{TINY.as_posix()},0\n{TINY.as_posix()},oops\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        load_matrix(csv)


def test_a_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        Cell(config=TINY, seed=-1)


def test_the_matrix_seed_wins_over_the_config_file(matrix):
    """Seed is the axis the matrix exists to sweep; the YAML's value is a
    default, not a constraint."""
    assert [config_for(cell).seed for cell in matrix] == [0, 1, 2]


# -- identity ------------------------------------------------------------------


def test_the_run_id_is_the_config_hash_and_the_seed(matrix):
    """A run is the pair (config_hash, seed) — DESIGN.md §4 — so the id that
    identifies it on disk must be exactly that and nothing else."""
    cfg = config_for(matrix[0])
    assert run_id_for(cfg) == f"{cfg.identity_hash()[:12]}_s0"


def test_the_same_config_and_seed_collapse_onto_one_run(tmp_path):
    """Two identical matrix rows are one experiment, not two. Without this a
    duplicated row silently burns a second run's worth of GPU."""
    csv = write_matrix(
        tmp_path / "run_matrix.csv", [(str(TINY), 4), (str(TINY), 4)]
    )
    first, second = load_matrix(csv)
    assert run_id_for(config_for(first)) == run_id_for(config_for(second))


def test_different_seeds_get_different_runs_but_the_same_condition(matrix):
    ids = {run_id_for(config_for(cell)) for cell in matrix}
    hashes = {config_for(cell).identity_hash() for cell in matrix}
    assert len(ids) == 3, "seeds must not share a run directory"
    assert len(hashes) == 1, "seeds of one condition must share a config hash"


def test_the_run_id_survives_a_reparsed_matrix(tmp_path, matrix):
    """The whole claim mechanism is that a *later session* recomputes the same
    id from the same CSV. If the id drifted, every resume would start over."""
    csv = write_matrix(
        tmp_path / "again.csv", [(str(TINY), 0), (str(TINY), 1), (str(TINY), 2)]
    )
    reparsed = load_matrix(csv)
    assert [run_id_for(config_for(c)) for c in reparsed] == [
        run_id_for(config_for(c)) for c in matrix
    ]


# -- derived status ------------------------------------------------------------


def test_no_directory_is_pending(tmp_path, matrix):
    assert status_of(tmp_path, run_id_for(config_for(matrix[0]))) is CellStatus.PENDING


def test_a_directory_without_a_result_is_resumable(tmp_path, matrix):
    """What a killed session leaves behind, and the case the queue exists for."""
    cfg = config_for(matrix[0])
    run_id = run_id_for(cfg)
    Run.create(cfg, tmp_path, run_id=run_id)
    assert status_of(tmp_path, run_id) is CellStatus.RESUMABLE


def test_a_recorded_crash_reads_as_failed(tmp_path, matrix):
    cfg = config_for(matrix[0])
    run_id = run_id_for(cfg)
    run = Run.create(cfg, tmp_path, run_id=run_id)
    run.log_failure(RuntimeError("boom"), step=3)
    assert status_of(tmp_path, run_id) is CellStatus.FAILED


def test_a_torn_failure_line_does_not_read_as_a_crash(tmp_path, matrix):
    """A session killed mid-write leaves an unparseable stump. That is not
    evidence of a crash, and must not be reported as one."""
    cfg = config_for(matrix[0])
    run_id = run_id_for(cfg)
    Run.create(cfg, tmp_path, run_id=run_id)
    (tmp_path / run_id / FAILURES_JSONL).write_text('{"at": "20', encoding="utf-8")
    assert status_of(tmp_path, run_id) is CellStatus.RESUMABLE


def test_a_result_file_is_done_even_after_a_crash(tmp_path, matrix):
    """result.json is the terminal state: a run that crashed, resumed and then
    finished is done, not failed."""
    cfg = config_for(matrix[0])
    run_id = run_id_for(cfg)
    run = Run.create(cfg, tmp_path, run_id=run_id)
    run.log_failure(RuntimeError("boom"), step=3)
    (tmp_path / run_id / RESULT_JSON).write_text("{}", encoding="utf-8")
    assert status_of(tmp_path, run_id) is CellStatus.DONE


# -- selection -----------------------------------------------------------------


def test_workers_partition_the_matrix_without_overlap_or_gaps(tmp_path, matrix):
    """Static partitioning is what makes claiming lock-free: if two workers
    could ever see one cell, the whole no-lease design collapses."""
    zero = select(matrix, tmp_path, worker=0, workers=2)
    one = select(matrix, tmp_path, worker=1, workers=2)

    assert not set(zero) & set(one)
    assert set(zero) | set(one) == set(matrix)


def test_a_bad_worker_index_is_rejected(tmp_path, matrix):
    with pytest.raises(ValueError):
        select(matrix, tmp_path, worker=2, workers=2)


def test_started_work_is_claimed_before_untouched_work(tmp_path, matrix):
    """Only started work has compute at risk, and the session may end at any
    moment — so finishing beats starting."""
    started = matrix[2]
    Run.create(config_for(started), tmp_path, run_id=run_id_for(config_for(started)))

    assert select(matrix, tmp_path)[0] == started


def test_a_finished_cell_is_never_claimed_again(tmp_path, matrix):
    """Re-running a completed matrix must be a no-op — that is what makes it
    safe to just re-run the notebook after every session death."""
    for cell in matrix:
        cfg = config_for(cell)
        run_id = run_id_for(cfg)
        Run.create(cfg, tmp_path, run_id=run_id)
        (tmp_path / run_id / RESULT_JSON).write_text("{}", encoding="utf-8")

    assert select(matrix, tmp_path) == []
    assert [s for _, s in statuses(matrix, tmp_path)] == [CellStatus.DONE] * 3


# -- running -------------------------------------------------------------------


def test_run_queue_works_through_the_whole_matrix(swept):
    cells, root, report = swept

    assert len(report.completed) == 3
    assert [s for _, s in statuses(cells, root)] == [CellStatus.DONE] * 3
    assert {row.seed for row in report.completed} == {0, 1, 2}
    assert {row.run_id for row in report.completed} == {
        run_id_for(config_for(cell)) for cell in cells
    }


def test_the_rows_carry_full_provenance(swept):
    """CLAUDE.md rule 7, on rows the queue produced rather than hand-built ones."""
    _, _, report = swept
    for row in report.completed:
        for field_name in (
            "pinnslab_version",
            "git_sha",
            "gpu_name",
            "dtype",
            "device_profile",
            "config_hash",
        ):
            assert getattr(row, field_name) != "", f"{field_name} empty on {row.run_id}"


def test_re_running_a_finished_matrix_does_nothing(swept):
    """The safety net behind "just re-run the notebook": a completed matrix must
    be a no-op, byte for byte."""
    cells, root, _ = swept
    ids = [run_id_for(config_for(cell)) for cell in cells]
    before = {
        run_id: (root / run_id / RESULT_JSON).read_text(encoding="utf-8")
        for run_id in ids
    }

    again = run_queue(cells, root)

    assert again.completed == [] and again.failed == []
    for run_id, text in before.items():
        after = (root / run_id / RESULT_JSON).read_text(encoding="utf-8")
        assert after == text, f"{run_id} was rewritten; results are append-only"


def test_one_bad_cell_does_not_end_the_sweep(tmp_path):
    """A config is only fully checkable by building it, so some cells can only
    fail at run time. The rest of the session is worth more than the failure."""
    csv = write_matrix(
        tmp_path / "run_matrix.csv",
        [(str(UNBUILDABLE), 0), (str(TINY), 0)],
    )
    report = run_queue(load_matrix(csv), tmp_path / "results")

    assert len(report.failed) == 1
    assert len(report.completed) == 1
    assert report.completed[0].seed == 0


def test_a_failed_cell_is_still_claimable_next_session(tmp_path):
    """status is failed, not done — a crash does not finish a run.

    Also pins that a crash during *assembly* is recorded at all. It happens
    after the run directory exists but before ``Trainer.fit`` can log anything,
    so without ``run_cell`` catching it the directory would be
    indistinguishable from one a session was killed in, and the config's
    failure would never reach the failure rate.
    """
    csv = write_matrix(tmp_path / "run_matrix.csv", [(str(UNBUILDABLE), 0)])
    cells = load_matrix(csv)
    root = tmp_path / "results"

    run_queue(cells, root)

    assert statuses(cells, root)[0][1] is CellStatus.FAILED
    assert select(cells, root) == cells


# -- the resampling guard ------------------------------------------------------


def test_resampling_plus_checkpointing_is_refused(tmp_path):
    """Collocation points are not checkpointed, so a resumed run of such a
    config silently trains on the initial cloud. The queue is the machinery
    that makes runs interruptible, so it is where this must be loud."""
    cell = Cell(config=RESAMPLED, seed=0)
    with pytest.raises(ValueError, match="resample_every"):
        run_cell(cell, tmp_path)


def test_the_guard_fires_before_any_cell_trains(tmp_path):
    """Checked up front, not when the bad cell is reached: discovering it two
    hours into a Kaggle session has already cost the session."""
    csv = write_matrix(
        tmp_path / "run_matrix.csv", [(str(TINY), 0), (str(RESAMPLED), 0)]
    )
    root = tmp_path / "results"

    with pytest.raises(ValueError, match="resample_every"):
        run_queue(load_matrix(csv), root)

    assert not root.exists() or list(root.iterdir()) == []


def test_the_guard_can_be_waived_explicitly(tmp_path):
    """An uninterruptible run is a legitimate case; it just has to be said."""
    report = run_queue(
        [Cell(config=RESAMPLED, seed=0)], tmp_path, allow_resampling=True
    )
    assert len(report.completed) == 1


# -- the deadline --------------------------------------------------------------


def test_an_exhausted_deadline_still_claims_one_cell_and_leaves_the_rest(
    tmp_path, matrix
):
    """The first cell is claimed regardless: with nothing measured there is no
    basis for declining, and a queue that never starts anything is its own kind
    of failure. Everything it declines must stay claimable next session."""
    report = run_queue(matrix, tmp_path, deadline_seconds=0.0)

    assert len(report.completed) == 1
    assert len(report.deferred) == 2
    assert set(report.deferred) == set(select(matrix, tmp_path))
