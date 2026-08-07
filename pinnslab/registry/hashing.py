"""Config hashing — the identity of an experimental condition.

The hash is the join key between a result row, a checkpoint, a search-layer
cache entry, and a figure. It must therefore be stable across processes, machines
and Python versions: ``hash()`` and ``pickle`` are both unusable for this.

Recipe: pydantic -> JSON-native dict -> canonical JSON (sorted keys, no
whitespace, no NaN) -> sha256 -> first 16 hex chars.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

HASH_LENGTH = 16


def to_jsonable(obj: Any) -> Any:
    """Reduce a config object to JSON-native types."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Byte-stable JSON for hashing.

    ``allow_nan=False`` is deliberate: a NaN in a config is a bug, and letting it
    through would produce non-standard JSON that other tools decode differently.
    """
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(obj: Any, *, length: int = HASH_LENGTH) -> str:
    """Stable short hash of a validated config."""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


__all__ = ["HASH_LENGTH", "canonical_json", "config_hash", "to_jsonable"]
