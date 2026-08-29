"""pinnslab — personal PINN methods research infrastructure.

Importing this package has NO side effects on global torch state. In particular
it does not call ``torch.set_default_dtype``: precision is a validated config
field (``RunConfig.dtype``, default ``float64``) applied by
:func:`pinnslab.utils.device.configure_runtime` at run start, so that it enters
the config hash and is recorded on every result row (DESIGN.md §5).
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
