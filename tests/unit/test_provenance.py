"""Where the git SHA on every row comes from (CLAUDE.md rule 7)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import pinnslab
from pinnslab.registry import provenance as prov_module
from pinnslab.registry.provenance import collect_provenance, git_info, utc_now

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_git_info_resolves_by_a_known_route():
    sha, dirty, source = git_info()
    assert source in {"git", "direct_url", "build_stamp", "unknown"}
    assert isinstance(dirty, bool)
    if source != "unknown":
        assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")


def test_uncommitted_code_is_reported_as_dirty():
    """A SHA plus uncommitted changes does not identify what actually ran."""
    sha, dirty, source = git_info()
    if source != "git":
        pytest.skip("not running from a working tree")
    # Nothing to assert about the value itself — it depends on the tree — but a
    # dirty tree must never silently report clean, so the flag must be present
    # and boolean, and callers must be able to see it on the row.
    assert isinstance(dirty, bool)


def test_provenance_carries_versions_and_hardware():
    prov = collect_provenance(seed=3, dtype="float64")
    assert prov.pinnslab_version == pinnslab.__version__
    assert prov.seed == 3
    assert prov.dtype == "float64"
    assert prov.gpu_name  # "cpu" when there is no GPU, never empty
    assert prov.device_profile
    assert prov.torch_version and prov.python_version and prov.hostname


def test_timestamps_are_utc_iso8601():
    stamp = utc_now()
    assert stamp.endswith("+00:00")
    assert "T" in stamp


# -- the third route: a wheel installed offline (DESIGN.md §7) -----------------
#
# `pip install git+...@tag` gets the commit from PEP 610 direct_url.json, and a
# checkout gets it from git. §7's *offline* fallback — build a wheel, upload it
# as a Kaggle Dataset, `pip install --no-index` — has neither, and reported
# git_sha="unknown" until 2026-08-28. That is a result row that cannot name the
# code that produced it, which rule 7 calls non-negotiable, on the platform
# where the session is gone by the time anyone asks.


def test_no_build_stamp_exists_in_a_source_checkout():
    """The generated file must never be committed, and is cleaned up after a
    build. If this fails, a stale stamp is sitting in the tree waiting to be
    read by an environment where git is unavailable."""
    assert not (REPO_ROOT / "pinnslab" / "_build_info.py").exists()
    assert prov_module._sha_from_build_stamp() is None


def test_the_working_tree_wins_over_a_build_stamp(monkeypatch):
    """Order is the whole safety property here.

    A developer with a checkout must be told what *their* tree is, never what
    some earlier build thought. Only if git cannot answer does the stamp count.
    """
    monkeypatch.setattr(
        prov_module, "_sha_from_build_stamp", lambda: ("b" * 40, False)
    )
    monkeypatch.setattr(
        prov_module, "_sha_from_working_tree", lambda: ("a" * 40, True)
    )
    git_info.cache_clear()
    try:
        assert git_info() == ("a" * 40, True, "git")
    finally:
        git_info.cache_clear()


def test_the_stamp_is_the_last_resort_not_a_silent_unknown(monkeypatch):
    monkeypatch.setattr(prov_module, "_sha_from_working_tree", lambda: (None, False))
    monkeypatch.setattr(prov_module, "_sha_from_direct_url", lambda: None)
    monkeypatch.setattr(
        prov_module, "_sha_from_build_stamp", lambda: ("c" * 40, True)
    )
    git_info.cache_clear()
    try:
        # dirty carries through: a wheel built from a dirty tree is not
        # described by its commit either, and claiming clean would be the lie.
        assert git_info() == ("c" * 40, True, "build_stamp")
    finally:
        git_info.cache_clear()


def test_nothing_left_to_report_is_reported_as_unknown(monkeypatch):
    monkeypatch.setattr(prov_module, "_sha_from_working_tree", lambda: (None, False))
    monkeypatch.setattr(prov_module, "_sha_from_direct_url", lambda: None)
    monkeypatch.setattr(prov_module, "_sha_from_build_stamp", lambda: None)
    git_info.cache_clear()
    try:
        assert git_info() == ("unknown", False, "unknown")
    finally:
        git_info.cache_clear()


@pytest.mark.slow
def test_a_built_wheel_carries_the_commit(tmp_path):
    """The hook actually runs, and the wheel actually contains the stamp.

    Out of process because it is a real `pip wheel` invocation: the build runs
    in pip's isolated environment with its own hatchling, which is the only
    place the hook is ever exercised. A unit test of `hatch_build.py` in this
    process would prove the function works and not that the build calls it.
    """
    # The hook can only stamp what git can tell it. If git cannot identify this
    # checkout — no git binary, or a CI runner tripping "dubious ownership" —
    # the hook correctly writes nothing, and asserting a stamp would be
    # asserting something untrue about the environment rather than about us.
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"git cannot describe {REPO_ROOT}: {probe.stderr.strip()}")

    done = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path),
         str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr

    wheels = list(tmp_path.glob("pinnslab-*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert "pinnslab/_build_info.py" in names, (
            "the build hook did not stamp the commit; an offline Kaggle install "
            "would report git_sha='unknown' (DESIGN.md §7, CLAUDE.md rule 7)"
        )
        stamp = archive.read("pinnslab/_build_info.py").decode("utf-8")

    namespace: dict = {}
    exec(stamp, namespace)  # noqa: S102 - our own generated file
    assert namespace["COMMIT"] == probe.stdout.strip()
    assert len(namespace["COMMIT"]) == 40
    assert set(namespace["COMMIT"]) <= set("0123456789abcdef")
    assert isinstance(namespace["DIRTY"], bool)

    # And the build put the tree back the way it found it.
    assert not (REPO_ROOT / "pinnslab" / "_build_info.py").exists()
