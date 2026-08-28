"""Provenance capture (CLAUDE.md rule 7, DESIGN.md §2).

Every result row records the library version, the git SHA it was produced by, the
config hash, the GPU, the dtype, the device profile and the seed. Non-negotiable:
a number in a paper that cannot be traced back to the code that produced it is
not defensible, and by the time a reviewer asks, the session is long gone.

Resolving the git SHA is the fiddly part, because the two situations differ:

* **local development** — there is a working tree; ask git, and record whether it
  was dirty (a dirty tree means the SHA alone does not identify the code).
* **Kaggle, online** — the package was installed with
  ``pip install git+...@tag`` and there is no ``.git`` anywhere. PEP 610
  ``direct_url.json`` records the commit pip resolved the tag to, which is
  exactly what we need.
* **Kaggle, offline** — DESIGN.md §7's fallback builds a wheel, uploads it as a
  Dataset and installs with ``--no-index``. A plain wheel carries no VCS
  metadata at all, so this reported ``"unknown"`` until 2026-08-28 — a result
  row that could not name the code that produced it, on the platform where the
  session is gone by the time anyone asks. ``hatch_build.py`` now stamps the
  commit into the wheel and this module reads it back.

The order matters and is deliberate: **working tree first, stamp last.** A
developer with a checkout must never be told what some earlier build thought,
and the stamp is gitignored so it cannot exist in a checkout anyway.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch

import pinnslab
from pinnslab.registry.schema import Provenance
from pinnslab.utils.device import device_profile, gpu_name

_GIT_TIMEOUT_S = 5.0


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def git_info() -> tuple[str, bool, str]:
    """``(sha, dirty, source)`` where source is ``git``/``direct_url``/``unknown``."""
    sha, dirty = _sha_from_working_tree()
    if sha is not None:
        return sha, dirty, "git"
    sha = _sha_from_direct_url()
    if sha is not None:
        # An installed artifact cannot be dirty.
        return sha, False, "direct_url"
    stamped = _sha_from_build_stamp()
    if stamped is not None:
        return stamped[0], stamped[1], "build_stamp"
    return "unknown", False, "unknown"


def _sha_from_working_tree() -> tuple[str | None, bool]:
    pkg_dir = Path(pinnslab.__file__).resolve().parent
    top = _git(pkg_dir, "rev-parse", "--show-toplevel")
    if top is None:
        return None, False
    # Guard against picking up an unrelated enclosing repository, e.g. a
    # site-packages tree that happens to live inside someone else's checkout.
    if not (Path(top) / "pinnslab" / "__init__.py").exists():
        return None, False
    sha = _git(pkg_dir, "rev-parse", "HEAD")
    if sha is None:
        return None, False
    # Untracked files count as dirty: code that is not in the commit is exactly
    # what makes the SHA an incomplete description of what ran. `--porcelain`
    # already honours .gitignore, so results/ and __pycache__ do not trip this.
    status = _git(pkg_dir, "status", "--porcelain")
    return sha, bool(status)


def _sha_from_direct_url() -> str | None:
    try:
        raw = metadata.distribution("pinnslab").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        return None
    return commit or None


def _sha_from_build_stamp() -> tuple[str, bool] | None:
    """The commit ``hatch_build.py`` wrote into the wheel, if this is one.

    Carries the build's dirty flag through rather than forcing it False: a
    wheel built from a dirty tree is not described by its commit either, and
    silently claiming otherwise is the failure this whole module exists to
    avoid.
    """
    try:
        from pinnslab import _build_info  # type: ignore[attr-defined]
    except ImportError:
        return None
    commit = getattr(_build_info, "COMMIT", None)
    if not commit:
        return None
    return str(commit), bool(getattr(_build_info, "DIRTY", False))


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def collect_provenance(*, seed: int, dtype: str) -> Provenance:
    """Snapshot everything rule 7 demands, plus enough to debug a bad run."""
    sha, dirty, source = git_info()
    return Provenance(
        pinnslab_version=pinnslab.__version__,
        git_sha=sha,
        git_dirty=dirty,
        git_source=source,
        gpu_name=gpu_name(),
        device_profile=device_profile(),
        dtype=dtype,
        seed=seed,
        timestamp_utc=utc_now(),
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.platform(),
    )


__all__ = ["collect_provenance", "git_info", "utc_now"]
