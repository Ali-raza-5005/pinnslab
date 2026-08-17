"""Importing a paper's method code, which is the only way its registrations run.

A registration exists once the module holding it has been imported, and the
module holding it lives in the *paper* repo (CLAUDE.md rule 2). So the loader is
the seam between a config that names ``strategy: rad`` and the file that defines
it, and its failure modes are worth pinning: a sweep whose sampler never
registered should say so at the top of the session, not an hour in.
"""

from __future__ import annotations

import pytest

from pinnslab.components import SAMPLERS
from pinnslab.utils.plugins import load_plugin, load_plugins

pytestmark = pytest.mark.unit

PLUGIN = """
from pinnslab.geometry.samplers import GeometrySampler, register_sampler


@register_sampler({name!r})
class Mine(GeometrySampler):
    pass
"""


def write_plugin(tmp_path, name: str):
    path = tmp_path / f"{name.replace('.', '_')}.py"
    path.write_text(PLUGIN.format(name=name), encoding="utf-8")
    return path


def test_a_plugin_file_registers_its_components(tmp_path):
    path = write_plugin(tmp_path, "test.plugin_by_path")

    load_plugin(str(path))

    assert "test.plugin_by_path" in SAMPLERS


def test_an_importable_module_is_loaded_by_name():
    """The normal case inside a paper repo, where ``src/`` is importable."""
    module = load_plugin("pinnslab.benchmarks.burgers")
    assert module.NAME == "burgers1d"


def test_loading_the_same_plugin_twice_is_not_a_duplicate_registration(tmp_path):
    """A registry refuses a duplicate key, so a loader that re-executed a module
    would turn "pass --register twice" into a crash — and passing it twice is
    exactly what a script and a notebook cell will do."""
    path = write_plugin(tmp_path, "test.plugin_twice")

    first = load_plugin(str(path))
    second = load_plugin(str(path))

    assert first is second


def test_a_missing_plugin_says_what_the_flag_accepts(tmp_path):
    with pytest.raises(FileNotFoundError, match="module name"):
        load_plugin(str(tmp_path / "not_here.py"))


def test_a_plugin_that_raises_does_not_stay_half_imported(tmp_path):
    """Otherwise the second attempt finds a broken module in ``sys.modules``
    and silently succeeds with nothing registered."""
    import sys

    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        load_plugin(str(path))

    assert "pinnslab_plugin_broken" not in sys.modules


def test_no_plugins_is_not_an_error():
    assert load_plugins(None) == []
    assert load_plugins([]) == []
