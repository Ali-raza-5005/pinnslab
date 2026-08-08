"""The config hash is the join key for rows, checkpoints, caches and figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from pinnslab.registry.config import RunConfig, load_config
from pinnslab.registry.hashing import canonical_json, config_hash, to_jsonable
from tests.conftest import toy_config

pytestmark = pytest.mark.unit

#: Pinned output of ``config_hash`` for a fixed payload. Not arbitrary: every
#: checkpoint, cache entry and result row on disk is keyed by this recipe
#: (json.dumps options -> sha256 -> truncate). The relative tests below (same
#: input -> same hash, different input -> different hash) would pass just as
#: happily if the recipe silently changed underneath them — this is the only
#: thing that would catch that, the same way DERIVE_SEED_GOLDEN catches a
#: drift in derive_seed.
CONFIG_HASH_GOLDEN_PAYLOAD = {
    "dtype": "float64",
    "seed": 7,
    "stages": [{"name": "adam", "steps": 20}],
}
CONFIG_HASH_GOLDEN_VALUE = "65c7cf2ef92a19c1"


def test_config_hash_matches_its_pinned_value():
    assert config_hash(CONFIG_HASH_GOLDEN_PAYLOAD) == CONFIG_HASH_GOLDEN_VALUE


def test_to_jsonable_reduces_a_basemodel_nested_inside_a_plain_list():
    """The only real caller path (``identity()``/``to_dict()``) hands to_jsonable
    a top-level BaseModel or an already-flattened dict, since pydantic's own
    ``model_dump`` recurses on its own from there. Nothing currently exercises
    to_jsonable's own recursion into a dict/list containing a raw, un-dumped
    BaseModel — this is the case the dict/list branches exist for.
    """

    class _Nested(BaseModel):
        weight: float

    payload = {"items": [_Nested(weight=1.5), _Nested(weight=2.5)]}
    assert to_jsonable(payload) == {"items": [{"weight": 1.5}, {"weight": 2.5}]}
    assert config_hash(payload)  # must not raise


def test_key_order_does_not_change_the_hash():
    assert config_hash({"a": 1, "b": [2, 3]}) == config_hash({"b": [2, 3], "a": 1})


def test_value_change_changes_the_hash():
    assert config_hash({"lr": 1e-3}) != config_hash({"lr": 1e-4})


def test_nan_is_rejected_rather_than_silently_hashed():
    with pytest.raises(ValueError):
        canonical_json({"lr": float("nan")})


@pytest.mark.slow
def test_hash_is_stable_across_processes():
    """`hash()` and `pickle` are not; this must be, or resumes break silently."""
    root = Path(__file__).resolve().parents[2]
    here = config_hash(toy_config().identity())
    code = (
        f"import sys; sys.path.insert(0, r'{root}');"
        "from tests.conftest import toy_config;"
        "print(toy_config().identity_hash())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == here


def test_identity_ignores_seed_device_and_operational_fields():
    """Five seeds of one condition must share a hash (DESIGN.md §8 groupby)."""
    base = toy_config()
    assert toy_config(seed=1).identity_hash() == toy_config(seed=2).identity_hash()
    assert toy_config(device="cpu").identity_hash() == base.identity_hash()
    assert toy_config(name="other").identity_hash() == base.identity_hash()


def test_identity_covers_dtype_and_stages():
    """float32 and float64 results are not comparable (DESIGN.md §5)."""
    base = toy_config()
    assert toy_config(dtype="float32").identity_hash() != base.identity_hash()

    stages = [s.model_copy(update={"steps": s.steps + 1}) for s in base.stages]
    assert toy_config(stages=stages).identity_hash() != base.identity_hash()


def test_yaml_round_trip_preserves_the_hash(tmp_path):
    from pinnslab.registry.config import dump_config

    cfg = toy_config()
    path = tmp_path / "cfg.yaml"
    dump_config(cfg, path)
    assert load_config(path).identity_hash() == cfg.identity_hash()


def test_unknown_yaml_key_is_an_error_not_a_silent_ignore(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "stages:\n  - name: a\n    steps: 1\n    optimizers: [{name: adam}]\n"
        "lr: 0.001\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="lr"):
        load_config(path)


def test_stage_names_must_be_unique():
    from pinnslab.registry.config import OptimizerSpec, StageSpec

    stage = StageSpec(name="dup", steps=1, optimizers=[OptimizerSpec()])
    with pytest.raises(ValueError, match="unique"):
        RunConfig(stages=[stage, stage])
