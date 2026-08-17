"""The command-line entry points, exercised as commands.

``scripts/`` is where a user meets this repo — README's first three commands are
all in here — so "it imports" is not the claim worth testing. Each test calls a
script's ``main`` with an argv the README actually documents, and checks the
artefact it promises appears on disk.

They are in-process rather than ``subprocess`` on purpose: a subprocess pays a
fresh ``import torch`` (~5s) and would push this file into the ``slow`` marker,
where the pre-commit loop would stop running it. What a subprocess *would* add —
that ``python scripts/x.py`` works from a checkout with no install — is covered
by ``test_the_scripts_run_from_a_bare_checkout`` below, which is marked slow and
runs one of them for real.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pinnslab.registry.config import load_config
from pinnslab.registry.run import RESULT_JSON

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
TINY = FIXTURES / "burgers_tiny.yaml"
EXAMPLES = REPO_ROOT / "examples"

sys.path.insert(0, str(SCRIPTS))

import run as run_script  # noqa: E402
import run_search as run_search_script  # noqa: E402
import run_sweep as run_sweep_script  # noqa: E402


def test_a_single_run_writes_a_complete_run_directory(tmp_path):
    results = tmp_path / "results"
    assert run_script.main([str(TINY), "--results", str(results)]) == 0

    (directory,) = list(results.iterdir())
    assert (directory / RESULT_JSON).exists()
    assert (directory / "config.yaml").exists()
    assert (directory / "trace.jsonl").exists()


def test_the_seed_override_names_the_same_condition(tmp_path):
    """A run is ``(config_hash, seed)``: overriding the seed must produce a
    second run of the *same* condition, not a second condition."""
    results = tmp_path / "results"
    run_script.main([str(TINY), "--results", str(results), "--seed", "0"])
    run_script.main([str(TINY), "--results", str(results), "--seed", "1"])

    directories = sorted(p.name for p in results.iterdir())
    condition = load_config(TINY).identity_hash()[:12]
    assert directories == [f"{condition}_s0", f"{condition}_s1"]


def test_re_running_a_finished_run_does_not_redo_it(tmp_path, capsys):
    """Same command, same directory: "resume" and "already done" are one gesture.

    A finished run is immutable, so the second invocation must report that and
    exit 0 rather than raise ``FileExistsError`` at the user — which is what it
    did before this test existed.
    """
    results = tmp_path / "results"
    run_script.main([str(TINY), "--results", str(results)])
    (directory,) = list(results.iterdir())
    written = (directory / RESULT_JSON).read_text(encoding="utf-8")
    capsys.readouterr()

    assert run_script.main([str(TINY), "--results", str(results)]) == 0

    assert "already done" in capsys.readouterr().out
    assert (directory / RESULT_JSON).read_text(encoding="utf-8") == written


def test_a_sweep_runs_every_cell_of_a_matrix(tmp_path):
    matrix = tmp_path / "run_matrix.csv"
    matrix.write_text(
        f"config,seed\n{TINY.as_posix()},0\n{TINY.as_posix()},1\n", encoding="utf-8"
    )
    results = tmp_path / "results"

    assert run_sweep_script.main([str(matrix), "--results", str(results)]) == 0
    assert len(list(results.iterdir())) == 2


def test_a_sweep_partitions_between_workers(tmp_path):
    """Two GPUs are two invocations, and they must not do the same work."""
    matrix = tmp_path / "run_matrix.csv"
    rows = "\n".join(f"{TINY.as_posix()},{seed}" for seed in range(4))
    matrix.write_text(f"config,seed\n{rows}\n", encoding="utf-8")
    results = tmp_path / "results"

    argv = [str(matrix), "--results", str(results), "--workers", "2"]
    run_sweep_script.main([*argv, "--worker", "0"])
    first = {p.name for p in results.iterdir()}
    run_sweep_script.main([*argv, "--worker", "1"])
    second = {p.name for p in results.iterdir()} - first

    assert len(first) == 2 and len(second) == 2


def _tiny_search(path: Path, **overrides) -> Path:
    spec = {
        "name": "smoke",
        "space": {
            "stages.0.optimizers.0.lr": {
                "kind": "continuous",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            }
        },
        "algorithm": "random",
        "pop_size": 2,
        "generations": 1,
        "budget": {"rungs": [5]},
        "fitness": {"metric": "rel_l2", "direction": "min"},
        "batched": False,
    }
    spec.update(overrides)
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_a_search_runs_and_reports_its_best_candidate(tmp_path, capsys):
    spec = _tiny_search(tmp_path / "search.yaml")

    code = run_search_script.main(
        [str(spec), "--base", str(TINY), "--root", str(tmp_path / "search")]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "best fitness" in out
    assert "stages.0.optimizers.0.lr" in out, "the winning config is not reported"


def test_re_running_a_finished_search_does_not_redo_its_generations(tmp_path):
    """The outer loop checkpoints at generation granularity, so the second
    invocation of the same command has nothing left to do — and must not spend
    a second population's worth of GPU discovering that."""
    from pinnslab.search.state import SearchState

    root = tmp_path / "search"
    spec = _tiny_search(tmp_path / "search.yaml")
    run_search_script.main([str(spec), "--base", str(TINY), "--root", str(root)])
    after_first = SearchState.load(root)

    run_search_script.main([str(spec), "--base", str(TINY), "--root", str(root)])
    after_second = SearchState.load(root)

    assert after_first is not None and after_second is not None
    assert after_second.generation == after_first.generation == 1
    assert len(after_second.archive) == len(after_first.archive) == 2


def test_a_search_root_written_by_a_different_spec_is_refused(tmp_path):
    """Resuming under a changed space, algorithm or budget would silently mix
    two experiments in one archive."""
    root = tmp_path / "search"
    spec = _tiny_search(tmp_path / "search.yaml")
    run_search_script.main([str(spec), "--base", str(TINY), "--root", str(root)])

    changed = _tiny_search(tmp_path / "changed.yaml", generations=2)
    with pytest.raises(ValueError, match="different search"):
        run_search_script.main(
            [str(changed), "--base", str(TINY), "--root", str(root)]
        )


def test_a_batched_search_on_ground_truth_is_refused_with_the_reason(tmp_path):
    """The batched evaluator never touches a reference solution, so a spec
    asking it for rel_l2 would silently score something else."""
    spec = _tiny_search(tmp_path / "search.yaml", batched=True)

    with pytest.raises(SystemExit, match="batched"):
        run_search_script.main(
            [str(spec), "--base", str(TINY), "--root", str(tmp_path / "s")]
        )


def test_register_imports_a_paper_s_own_components(tmp_path):
    """``--register`` is how a paper's sampler reaches a run that the library
    knows nothing about: the config names it, and the flag is what makes the
    name resolvable.

    The plugin is written here rather than reusing ``examples/rad_sampler.py``
    so that this test does not depend on whether another test imported that
    module first — a registry is process-global, and a test that passes only in
    file order is worse than no test.
    """
    plugin = tmp_path / "paper_sampler.py"
    plugin.write_text(
        "from pinnslab.geometry.samplers import GeometrySampler, register_sampler\n"
        "@register_sampler('test.script_plugin')\n"
        "class Mine(GeometrySampler):\n"
        "    def __init__(self, spec, problem):\n"
        "        super().__init__(spec.model_copy(update={'strategy': 'pseudo'}),"
        " problem)\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    config["sampling"]["points"]["interior"]["strategy"] = "test.script_plugin"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    code = run_script.main(
        [
            str(path),
            "--results",
            str(tmp_path / "results"),
            "--register",
            str(plugin),
        ]
    )

    assert code == 0
    from pinnslab.components import SAMPLERS

    assert "test.script_plugin" in SAMPLERS


def test_the_ci_workflow_parses_and_runs_both_documented_commands():
    """A malformed workflow does not fail loudly — GitHub just declines to run
    it, and the repo looks green because nothing ran. This file was in fact
    invalid when first written (an unquoted ``budget: <60s`` in a step name).

    It also pins that CI runs the two commands CLAUDE.md requires of a session,
    rather than drifting into some third thing nobody runs by hand.
    """
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    )
    # `on:` is YAML 1.1's boolean true; PyYAML parses the key that way.
    triggers = workflow.get("on") or workflow.get(True)
    runs = [step.get("run", "").strip() for step in workflow["jobs"]["test"]["steps"]]

    assert {"push", "pull_request"} <= set(triggers)
    assert any("ruff check ." in run for run in runs)
    assert any('pytest -m "unit and not slow"' in run for run in runs)
    assert any(
        run.startswith("pytest") and "-m" not in run for run in runs
    ), "CI never runs the full suite, so slow and golden tests are ungated"


@pytest.mark.slow
def test_the_scripts_run_from_a_bare_checkout(tmp_path):
    """No install, no PYTHONPATH: ``python scripts/run.py`` must just work.

    Python puts the *script's* directory on ``sys.path``, not the working
    directory, so this failed until ``scripts/_bootstrap.py`` existed — and it
    failed for exactly the user who follows the README's first command.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run.py"),
            str(TINY),
            "--results",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # not the repo root: nothing is on the path by accident
        env=env,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-3000:]
    assert "completed" in result.stdout
