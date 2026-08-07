"""Loss weighting: the object that reduces per-point residuals to a scalar.

Only the trivial fixed-coefficient reducer exists so far — NTK, GradNorm,
self-adaptive, causal and min-max weighting arrive with the papers that need
them (DESIGN.md §9).
"""

from pinnslab.losses.weighting import MeanWeighting, build_weighting

__all__ = ["MeanWeighting", "build_weighting"]
