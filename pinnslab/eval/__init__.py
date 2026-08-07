"""Metrics: rel-L2, absolute and max error, evaluation grids.

Defined once here so that two papers reporting "rel-L2" are reporting the same
quantity. Conservation error and time-to-accuracy arrive with the problems that
need them.
"""

from pinnslab.eval.metrics import l2_error, max_error, relative_l2, uniform_grid

__all__ = ["l2_error", "max_error", "relative_l2", "uniform_grid"]
