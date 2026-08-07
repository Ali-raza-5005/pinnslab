"""Run provenance: the Run object, config hashing, and the results schema.

This package is about *recording experiments*. Component registration
(``@register_weighting`` and friends) lives in :mod:`pinnslab.components` — a
different sense of the word "registry" entirely.
"""

from pinnslab.registry.config import RunConfig, load_config
from pinnslab.registry.hashing import config_hash
from pinnslab.registry.run import Run, load_runs
from pinnslab.registry.schema import MetricSchedule, ResultRow, RunStatus

__all__ = [
    "MetricSchedule",
    "ResultRow",
    "Run",
    "RunConfig",
    "RunStatus",
    "config_hash",
    "load_config",
    "load_runs",
]
