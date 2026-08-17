"""Importing a paper's method code so its ``@register_*`` decorators run.

A registration only exists once the module holding it has been imported. In a
notebook that is a plain ``import src.method``; from a command line it needs a
flag, because the method lives in the *paper* repo and ``pinnslab`` must not
know its name (CLAUDE.md rule 2).

Both forms are accepted, and which one is right depends on where the code sits:

* ``pinnslab.utils.plugins.load_plugins(["paper01.method"])`` — an installed or
  importable module, the normal case inside a paper repo;
* ``load_plugins(["src/method/rad.py"])`` — a path, for a file that is not on
  ``sys.path`` yet (a Kaggle input directory, this repo's ``examples/``).

Failures are loud and name the thing that failed. A sweep whose sampler silently
failed to register would fall back to... nothing, actually — the config would
raise on the unknown name — but it would raise deep in the first cell rather
than at the top of the session, which on Kaggle is an hour later.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from pinnslab.utils.logging import get_logger

log = get_logger(__name__)


def load_plugin(target: str) -> ModuleType:
    """Import one module, by dotted name or by file path."""
    if target.endswith(".py") or "/" in target or "\\" in target:
        return _load_path(Path(target))
    module = importlib.import_module(target)
    log.info("registered components from module %s", target)
    return module


def load_plugins(targets: Iterable[str] | None) -> list[ModuleType]:
    return [load_plugin(t) for t in (targets or [])]


def _load_path(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"no plugin at {path}; --register takes a module name "
            "('paper01.method') or a path to a .py file"
        )
    name = f"pinnslab_plugin_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot import {path} as a module")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a plugin that imports itself does not loop.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    log.info("registered components from %s", path)
    return module


__all__ = ["load_plugin", "load_plugins"]
