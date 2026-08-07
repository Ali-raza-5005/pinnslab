"""Provenance (CLAUDE.md rule 7) and append-only enforcement (rule 6)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pinnslab.registry.run import Run, load_runs
from pinnslab.registry.schema import RESULT_SCHEMA_VERSION, RunStatus
from tests.conftest import toy_config

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

#: Exactly the fields CLAUDE.md rule 7 calls non-negotiable.
RULE_7_FIELDS = (
    "pinnslab_version",
    "git_sha",
    "config_hash",
    "gpu_name",
    "dtype",
    "device_profile",
    "seed",
)


def test_create_writes_config_and_provenance(results_root):
    cfg = toy_config()
    run = Run.create(cfg, results_root)

    assert (run.path / "config.yaml").exists()
    assert (run.path / "provenance.json").exists()
    assert run.checkpoint_dir.is_dir()
    stored = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
    assert stored["seed"] == cfg.seed


def test_result_row_carries_every_rule_7_field(results_root):
    run = Run.create(toy_config(), results_root)
    row = run.finish(status=RunStatus.COMPLETED, steps_completed=10)
    dumped = row.model_dump()
    for name in RULE_7_FIELDS:
        assert name in dumped, name
        assert dumped[name] not in (None, ""), name


def test_run_id_encodes_config_and_seed(results_root):
    cfg = toy_config(seed=13)
    run = Run.create(cfg, results_root)
    assert cfg.identity_hash()[:8] in run.run_id
    assert "_s13_" in run.run_id


def test_creating_an_existing_run_directory_raises(results_root):
    cfg = toy_config()
    run = Run.create(cfg, results_root, run_id="fixed")
    with pytest.raises(FileExistsError, match="append-only"):
        Run.create(cfg, results_root, run_id=run.run_id)


def test_finishing_twice_raises(results_root):
    run = Run.create(toy_config(), results_root)
    run.finish(status=RunStatus.COMPLETED, steps_completed=1)
    with pytest.raises(FileExistsError):
        run.finish(status=RunStatus.COMPLETED, steps_completed=2)


def test_resume_appends_to_the_same_trace(results_root):
    cfg = toy_config()
    run = Run.create(cfg, results_root, run_id="r0")
    run.log_metrics(10, {"loss": 1.0})

    resumed = Run.resume(results_root, "r0", cfg)
    resumed.log_metrics(20, {"loss": 0.5})

    steps = [p.step for p in resumed.read_trace()]
    assert steps == [10, 20]
    assert resumed.resumed is True


def test_resume_refuses_a_finished_run(results_root):
    cfg = toy_config()
    run = Run.create(cfg, results_root, run_id="r1")
    run.finish(status=RunStatus.COMPLETED, steps_completed=1)
    with pytest.raises(FileExistsError, match="immutable"):
        Run.resume(results_root, "r1", cfg)


def test_resume_refuses_a_different_condition(results_root):
    Run.create(toy_config(), results_root, run_id="r2")
    with pytest.raises(ValueError, match="config mismatch"):
        Run.resume(results_root, "r2", toy_config(dtype="float32"))


def test_resume_allows_a_different_seed_of_the_same_condition(results_root):
    """Seed is not part of the config identity; the row records it separately."""
    Run.create(toy_config(seed=1), results_root, run_id="r3")
    resumed = Run.resume(results_root, "r3", toy_config(seed=2))
    assert resumed.config_hash == toy_config(seed=1).identity_hash()


def test_create_or_resume_is_idempotent(results_root):
    cfg = toy_config()
    first = Run.create_or_resume(cfg, results_root, "queue-row-7")
    second = Run.create_or_resume(cfg, results_root, "queue-row-7")
    assert first.path == second.path
    assert second.resumed is True


def test_diverged_runs_are_recorded_not_discarded(results_root):
    run = Run.create(toy_config(), results_root)
    row = run.finish(
        status=RunStatus.DIVERGED, steps_completed=42, error="non-finite loss"
    )
    assert row.status is RunStatus.DIVERGED
    assert load_runs(results_root)[0].error == "non-finite loss"


def test_load_runs_skips_unfinished(results_root):
    Run.create(toy_config(), results_root, run_id="done").finish(
        status=RunStatus.COMPLETED, steps_completed=1
    )
    Run.create(toy_config(), results_root, run_id="running")
    rows = load_runs(results_root)
    assert [r.run_id for r in rows] == ["done"]


def test_nonfinite_metrics_stay_valid_json(results_root):
    """RFC 8259 has no NaN, and ``results/`` cannot be repaired after the fact."""
    run = Run.create(toy_config(), results_root)
    run.log_metrics(3, {"loss": float("nan")})
    run.finish(
        status=RunStatus.DIVERGED,
        steps_completed=3,
        final_metrics={"loss": float("nan"), "hi": float("inf"), "lo": float("-inf")},
    )

    def reject_constants(token):
        raise AssertionError(f"{token} is not valid JSON")

    result = (run.path / "result.json").read_text(encoding="utf-8")
    json.loads(result, parse_constant=reject_constants)
    for line in (run.path / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line, parse_constant=reject_constants)

    # ...and the values survive the round-trip, sign of the infinity included.
    metrics = load_runs(results_root)[0].final_metrics
    assert math.isnan(metrics["loss"])
    assert metrics["hi"] == float("inf")
    assert metrics["lo"] == float("-inf")
    assert math.isnan(run.read_trace()[0].metrics["loss"])


def test_a_crash_is_recorded_without_finishing_the_run(results_root):
    """A crash with a checkpoint behind it must stay resumable (DESIGN.md §7)."""
    cfg = toy_config()
    run = Run.create(cfg, results_root, run_id="crashed")
    run.log_failure(RuntimeError("CUDA out of memory"), step=17)

    assert not run.is_finished
    Run.resume(results_root, "crashed", cfg)  # must not raise


def test_unfinished_runs_reach_the_failure_rate_when_asked(results_root):
    """A rate computed only over runs that survived to report is not a rate."""
    Run.create(toy_config(), results_root, run_id="a-done").finish(
        status=RunStatus.COMPLETED, steps_completed=1
    )
    Run.create(toy_config(), results_root, run_id="b-crashed").log_failure(
        RuntimeError("CUDA out of memory"), step=17
    )
    Run.create(toy_config(), results_root, run_id="c-killed").log_metrics(9, {"l": 1.0})

    assert [r.run_id for r in load_runs(results_root)] == ["a-done"]

    rows = load_runs(results_root, include_unfinished=True)
    assert [(r.run_id, r.status) for r in rows] == [
        ("a-done", RunStatus.COMPLETED),
        ("b-crashed", RunStatus.FAILED),
        ("c-killed", RunStatus.RUNNING),
    ]
    crashed = rows[1]
    assert "CUDA out of memory" in crashed.error
    assert crashed.config_hash == toy_config().identity_hash()
    assert all(getattr(crashed, f) for f in RULE_7_FIELDS)
    # The killed run's progress is recovered from the append-only trace.
    assert rows[2].steps_completed == 9


def test_load_runs_survives_a_directory_killed_before_it_was_a_run(results_root):
    (results_root / "half-born").mkdir()
    Run.create(toy_config(), results_root, run_id="real").log_metrics(1, {"l": 1.0})
    rows = load_runs(results_root, include_unfinished=True)
    assert [r.run_id for r in rows] == ["real"]


def test_load_runs_tolerates_a_truncated_final_line(results_root):
    """SIGKILL lands mid-write; the rest of the file is still evidence."""
    run = Run.create(toy_config(), results_root, run_id="killed")
    run.log_metrics(10, {"loss": 1.0})
    with (run.path / "trace.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"step": 20, "wall_ti')
    assert load_runs(results_root, include_unfinished=True)[0].steps_completed == 10


def test_read_trace_tolerates_a_truncated_final_line(results_root):
    """The same SIGKILL ``load_runs`` survives must not break ``read_trace``.

    This is the resume path, so the half-written line is the *expected* state of
    the file, not a corruption: the caller is reading a trace precisely because
    an earlier session was killed part-way through writing it.
    """
    run = Run.create(toy_config(), results_root, run_id="killed-mid-write")
    run.log_metrics(10, {"loss": 1.0})
    with (run.path / "trace.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"step": 20, "wall_ti')

    assert [p.step for p in run.read_trace()] == [10]


def test_records_written_after_a_torn_line_are_recovered(results_root):
    """The torn line is not always the last one, because the run resumes.

    ``_append_jsonl`` opens with ``"a"``, so the next session's first record
    lands directly on the stump the killed one left and the two fuse into a
    single bad line — with good records after it. Stopping there would report a
    run that trained to step 40 as having reached step 10, which is a wrong
    number that looks entirely reasonable.
    """
    cfg = toy_config()
    run = Run.create(cfg, results_root, run_id="killed-then-resumed")
    run.log_metrics(10, {"loss": 1.0})
    with (run.path / "trace.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"step": 20, "wall_ti')  # SIGKILL landed inside this write

    resumed = Run.resume(results_root, "killed-then-resumed", cfg)
    resumed.log_metrics(30, {"loss": 0.5})  # fuses onto the stump above
    resumed.log_metrics(40, {"loss": 0.25})

    # Step 30 is lost with the stump it fused onto; everything after survives.
    assert [p.step for p in resumed.read_trace()] == [10, 40]
    assert load_runs(results_root, include_unfinished=True)[0].steps_completed == 40


# -- schema evolution ---------------------------------------------------------


def test_finish_stamps_the_schema_version(results_root):
    run = Run.create(toy_config(), results_root)
    row = run.finish(status=RunStatus.COMPLETED, steps_completed=1)

    assert row.schema_version == RESULT_SCHEMA_VERSION
    stored = json.loads((run.path / "result.json").read_text(encoding="utf-8"))
    assert stored["schema_version"] == RESULT_SCHEMA_VERSION


def test_a_row_written_before_schema_version_existed_still_loads(results_root):
    """``results/`` is append-only, so every row ever written is permanent.

    The fixture is frozen on disk and must never be regenerated — it is a row in
    the shape today's code can no longer produce, which is the only thing that
    makes this test mean anything.
    """
    directory = results_root / "20260801T101500Z_3f2a1c9d_s7_ab12cd"
    directory.mkdir()
    (directory / "result.json").write_text(
        (FIXTURES / "result_row_v1.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    (row,) = load_runs(results_root)

    # Absent on disk: v1 is by definition the schema that had no such field.
    assert row.schema_version == 1
    assert row.status is RunStatus.DIVERGED
    assert row.steps_completed == 18321
    assert row.tags["sweep"] == "sampling"
    assert math.isnan(row.final_metrics["loss"])
    assert row.final_metrics["l2_relative"] == float("inf")
    for name in RULE_7_FIELDS:
        assert getattr(row, name) not in (None, ""), name


def test_a_row_from_a_newer_schema_loads_without_its_unknown_fields(results_root):
    """A Kaggle session pinned to an older tag must not choke on newer rows.

    Forward compatibility only reaches as far as *added* fields; a rename or a
    removal still needs the reader to branch on ``schema_version``, which is the
    reason the field exists.
    """
    stored = json.loads((FIXTURES / "result_row_v1.json").read_text(encoding="utf-8"))
    stored["schema_version"] = RESULT_SCHEMA_VERSION + 1
    stored["a_field_invented_later"] = {"x": 1}
    directory = results_root / "from-the-future"
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(stored), encoding="utf-8")

    (row,) = load_runs(results_root)

    assert row.schema_version == RESULT_SCHEMA_VERSION + 1
    assert not hasattr(row, "a_field_invented_later")
