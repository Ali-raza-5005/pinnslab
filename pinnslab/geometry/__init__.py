"""Geometry adapters.

The ONLY place deepxde may be imported; no deepxde object escapes it
(DESIGN.md §1). Import from here rather than from ``adapters`` directly — the
module path is an implementation detail of which library generates the points.
"""

from pinnslab.geometry.adapters import Domain, interval, with_time

__all__ = ["Domain", "interval", "with_time"]
