"""One logging format for the whole library. Deliberately tiny."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``pinnslab`` root, configured once per process.

    Level comes from ``PINNSLAB_LOG_LEVEL`` (default ``INFO``) so a Kaggle
    notebook can turn up verbosity without editing code.
    """
    global _configured
    root = logging.getLogger("pinnslab")
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("PINNSLAB_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _configured = True
    qualified = name if name.startswith("pinnslab") else f"pinnslab.{name}"
    return logging.getLogger(qualified)


__all__ = ["get_logger"]
