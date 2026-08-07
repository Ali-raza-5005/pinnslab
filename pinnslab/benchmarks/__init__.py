"""Frozen canonical PDEs.

Only 1-D Burgers so far. Allen-Cahn, KdV, Helmholtz, NS-lid and wave arrive
when a paper needs them — each as its own ``@register_problem`` file.

Importing this package registers every built-in benchmark, which is what makes
``ProblemSpec(name="burgers1d")`` resolvable from a config.
"""

from pinnslab.benchmarks import burgers  # noqa: F401  (registers burgers1d)
from pinnslab.benchmarks.problem import Problem, build_problem

__all__ = ["Problem", "build_problem"]
