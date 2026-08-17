"""Make ``python scripts/<name>.py`` work from a plain checkout.

Python puts the *script's* directory on ``sys.path``, not the working
directory, so a checkout that has not been ``pip install``ed cannot import
``pinnslab`` from here — which is a confusing first five minutes, given that
``pytest`` works in the same tree (``pythonpath = ["."]`` in pyproject.toml).

Importing this module first puts the repository root on the path. It is a no-op
when the package is installed and the checkout is the install, which is the
normal case (``pip install -e .``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
