"""The DESIGN.md §9 step 3 deliverable: prove the runner survives a killed session.

Everything else about the queue is bookkeeping. This is the property the design
is *for*: a Kaggle session ends without warning, and the sweep must lose no
completed work, no in-progress work, and — the part that is easy to get wrong
and impossible to notice — no reproducibility. A queue that quietly restarted
the interrupted cell from scratch would pass every weaker test in
``test_queue.py`` and would still be correct-looking, just slower. A queue that
resumed it from a stale checkpoint would be neither.

So the assertion that matters is the last one: **every cell's final parameters
are bit-identical to an uninterrupted sweep**. Killed and unkilled must not be
distinguishable in the results.

Out-of-process and marked ``slow`` because the kill has to be a real
``os._exit``. An exception unwinds cleanly, flushes buffers and runs ``finally``
blocks; a killed session does none of that, and the difference is precisely
what the append-only files have to tolerate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pinnslab.registry.run import RESULT_JSON, SESSIONS_JSONL
from pinnslab.training.checkpoint import load_checkpoint
from pinnslab.training.queue import config_for, load_matrix, run_id_for

pytestmark = [pytest.mark.unit, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
WORKER = FIXTURES / "queue_worker.py"
TINY = FIXTURES / "configs" / "burgers_tiny.yaml"
RESAMPLED = FIXTURES / "configs" / "burgers_tiny_resampled.yaml"

#: Both configs run 60 Adam steps and checkpoint every 10, so a kill here lands
#: mid-stage with a real checkpoint (step 30) behind it — not on a boundary,
#: which is the case that would pass by accident.
#:
#: 35 is also, deliberately, *between* resamples for the resampling cell
#: (``resample_every: 20``, so the cloud in force was drawn at step 20 and the
#: next draw is at 40). A resumed run that redrew its points instead of
#: restoring them would diverge from the reference here and nowhere else.
KILL_STEP = 35

#: The cell that gets killed is the resampling one. Sampling is paper 1's
#: subject and the point cloud is the newest thing in the checkpoint, so it is
#: the cell whose bit-exact resume is worth spending the SIGKILL on.
MATRIX = ((TINY, 0), (RESAMPLED, 1), (TINY, 2))


def write_matrix(path: Path) -> Path:
    """A small sweep of the shape a real one has: several cells, mixed configs."""
    rows = "\n".join(f"{config.as_posix()},{seed}," for config, seed in MATRIX)
    path.write_text(f"config,seed,notes\n{rows}\n", encoding="utf-8")
    return path


def run_worker(matrix: Path, root: Path, kill: tuple[str, int] | None = None):
    args = [sys.executable, str(WORKER), str(matrix), str(root)]
    if kill is not None:
        args += [kill[0], str(kill[1])]
    # A script's own directory goes on sys.path, not the cwd, and the package is
    # used from the checkout rather than installed (pytest's `pythonpath` does
    # not reach a subprocess).
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        args, capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=600
    )


def sessions(root: Path, run_id: str) -> list[dict]:
    lines = (root / run_id / SESSIONS_JSONL).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def final_params(root: Path, run_id: str) -> dict[str, torch.Tensor]:
    return load_checkpoint(root / run_id / "checkpoints" / "last.pt").nets["u"]


@pytest.fixture(scope="module")
def killed_and_reference(tmp_path_factory):
    """One uninterrupted sweep, and one killed sweep that was then re-run.

    Module-scoped: this is four subprocesses and three trainings each, and
    every assertion below reads the same two directories.
    """
    base = tmp_path_factory.mktemp("queue")
    matrix = write_matrix(base / "run_matrix.csv")
    cells = load_matrix(matrix)
    ids = [run_id_for(config_for(cell)) for cell in cells]

    reference_root = base / "reference"
    assert run_worker(matrix, reference_root).returncode == 0

    killed_root = base / "killed"
    first = run_worker(matrix, killed_root, kill=(ids[1], KILL_STEP))
    assert first.returncode != 0, "the worker was supposed to be killed"
    assert "KILLING" in first.stderr

    # The session comes back and the notebook is simply re-run — unchanged.
    second = run_worker(matrix, killed_root)
    assert second.returncode == 0, second.stderr[-3000:]

    return ids, reference_root, killed_root


def test_the_whole_matrix_finishes(killed_and_reference):
    ids, _, killed = killed_and_reference
    for run_id in ids:
        assert (killed / run_id / RESULT_JSON).exists(), f"{run_id} never finished"


def test_completed_work_is_not_repeated(killed_and_reference):
    """The cell that finished before the kill must be left alone. Re-running it
    would be invisible in the results and would waste a whole cell of the next
    session's quota — on a sweep that is 90% done, most of the session."""
    ids, _, killed = killed_and_reference
    events = [entry["event"] for entry in sessions(killed, ids[0])]
    assert events == ["create"], f"cell 0 was touched again: {events}"


def test_interrupted_work_is_resumed_not_restarted(killed_and_reference):
    ids, _, killed = killed_and_reference
    events = [entry["event"] for entry in sessions(killed, ids[1])]
    assert events == ["create", "resume"]


def test_work_never_started_is_picked_up(killed_and_reference):
    ids, _, killed = killed_and_reference
    assert [entry["event"] for entry in sessions(killed, ids[2])] == ["create"]


def test_the_resumed_cell_is_claimed_before_the_untouched_one(killed_and_reference):
    """Finishing beats starting: only started work has compute at risk.

    Read off the run directories rather than asserted on ``select`` directly,
    so this is about what the session actually did.
    """
    ids, _, killed = killed_and_reference
    resumed_at = [e for e in sessions(killed, ids[1]) if e["event"] == "resume"][0]
    started_at = sessions(killed, ids[2])[0]
    assert resumed_at["at"] <= started_at["at"]


def test_every_cell_is_bit_identical_to_an_uninterrupted_sweep(killed_and_reference):
    """The assertion the whole design exists to make true.

    Note this also pins that a cell's result does not depend on what ran before
    it in the same process: the killed sweep trains its cells in the order
    (0), (1, 2) across two sessions, the reference in one session as (0, 1, 2).
    If any global state leaked between cells — a seed set once, a default dtype,
    an RNG stream shared across runs — these numbers would differ.
    """
    ids, reference, killed = killed_and_reference
    for run_id in ids:
        expected = final_params(reference, run_id)
        actual = final_params(killed, run_id)
        assert expected.keys() == actual.keys()
        for name, tensor in expected.items():
            assert torch.equal(actual[name], tensor), (
                f"{run_id} parameter {name} drifted; the killed sweep is a "
                "different experiment from the uninterrupted one"
            )


def test_the_recorded_metrics_match_too(killed_and_reference):
    """Bit-identical parameters with a different reported number would mean the
    metric, not the training, moved across the resume."""
    ids, reference, killed = killed_and_reference
    for run_id in ids:
        expected = json.loads(
            (reference / run_id / RESULT_JSON).read_text(encoding="utf-8")
        )
        actual = json.loads(
            (killed / run_id / RESULT_JSON).read_text(encoding="utf-8")
        )
        assert actual["final_metrics"] == expected["final_metrics"], run_id
        assert actual["steps_completed"] == expected["steps_completed"]
