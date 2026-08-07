"""Network architectures.

Only the plain MLP so far. Modified-MLP, Fourier features and SIREN are born in
the paper repo that first needs them and are promoted here on the second
(DESIGN.md §2) — each as its own ``@register_model`` file, editing nothing.
"""

from pinnslab.models.mlp import MLP, Sin, build_net, count_parameters

__all__ = ["MLP", "Sin", "build_net", "count_parameters"]
