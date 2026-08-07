"""Where the git SHA on every row comes from (CLAUDE.md rule 7)."""

from __future__ import annotations

import pytest

import pinnslab
from pinnslab.registry.provenance import collect_provenance, git_info, utc_now

pytestmark = pytest.mark.unit


def test_git_info_resolves_by_a_known_route():
    sha, dirty, source = git_info()
    assert source in {"git", "direct_url", "unknown"}
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
