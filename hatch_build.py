"""Stamp the commit into the wheel, so an offline install can still say what ran.

CLAUDE.md rule 7 makes the git SHA on every result row non-negotiable, and
:func:`pinnslab.registry.provenance.git_info` has two ways to find it:

* a **working tree** — local development, ``git rev-parse`` answers;
* **PEP 610 ``direct_url.json``** — ``pip install git+...@tag``, which is
  DESIGN.md §7's primary Kaggle path; pip records the commit it resolved.

That leaves one documented path with no answer: §7's offline fallback, *"build
wheel, upload as Kaggle Dataset, pip install --no-index --find-links=..."*. A
plain wheel carries no VCS metadata at all, so provenance came out as
``("unknown", False, "unknown")`` — a result row that cannot name the code that
produced it, which is precisely what rule 7 exists to prevent, on the platform
where the session is gone by the time anyone asks.

So the build writes ``pinnslab/_build_info.py`` with the commit it was built
from. The file is gitignored, is never present in a source checkout, and is
consulted *last* — a working tree always wins, so a developer can never be
misled by a stale stamp left over from a local build.

If the build happens somewhere without git or without a repository (an sdist
unpacked in a container, say), the hook writes nothing and provenance degrades
to ``"unknown"`` exactly as before. Being unable to determine the commit is a
fact to report, not a reason to fail a build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

TARGET = Path("pinnslab") / "_build_info.py"

TEMPLATE = '''"""Generated at build time by hatch_build.py. Not in version control.

Present only in a built artifact. A source checkout has no such file, and
``registry.provenance`` prefers the working tree anyway.
"""

COMMIT = {commit!r}
DIRTY = {dirty!r}
'''


class BuildInfoHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    _written: Path | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        commit, dirty = _describe(Path(self.root))
        if commit is None:
            return
        path = Path(self.root) / TARGET
        path.write_text(
            TEMPLATE.format(commit=commit, dirty=dirty), encoding="utf-8"
        )
        # force_include rather than relying on the package glob, so the file is
        # picked up even when hatchling has already resolved the file list.
        build_data.setdefault("force_include", {})[str(path)] = str(TARGET)
        self._written = path

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        """Take the stamp back out of the working tree.

        It is gitignored, and ``git_info`` prefers the working tree anyway, so
        leaving it would be harmless today. Removing it keeps that from being
        something anyone has to know: a generated file that outlives its build
        is a stale file waiting for the one environment where it gets read.
        """
        written = getattr(self, "_written", None)
        if written is not None:
            Path(written).unlink(missing_ok=True)
            self._written = None


def _describe(root: Path) -> tuple[str | None, bool]:
    """``(sha, dirty)`` for ``root``, or ``(None, False)`` if git cannot say."""
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return None, False
    # Untracked files count as dirty for the same reason they do in
    # provenance.py: code that is not in the commit is what makes the SHA an
    # incomplete description of what ran.
    return sha, bool(_git(root, "status", "--porcelain"))


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
