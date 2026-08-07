"""Built-in optimizer factories.

Registered, not hard-coded: a paper repo adds ``@register_optimizer("soap")`` in
its own ``src/method/`` and the config accepts it with zero edits here
(CLAUDE.md rule 9).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pinnslab.components import register_optimizer
from pinnslab.registry.config import OptimizerSpec

Params = Iterable[torch.Tensor]


@register_optimizer("adam")
def build_adam(params: Params, spec: OptimizerSpec) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=spec.lr, **spec.options)


@register_optimizer("lbfgs")
def build_lbfgs(params: Params, spec: OptimizerSpec) -> torch.optim.Optimizer:
    """Full-batch L-BFGS with strong-Wolfe line search (DESIGN.md §5).

    Without a line search L-BFGS on PINN losses stalls or diverges; ``max_iter``
    is deliberately modest so that one ``.step()`` remains a meaningful unit of
    progress for the trace and the stage step count.
    """
    options = {"max_iter": 20, "line_search_fn": "strong_wolfe", **spec.options}
    return torch.optim.LBFGS(params, lr=spec.lr, **options)


def build_optimizer(params: Params, spec: OptimizerSpec) -> torch.optim.Optimizer:
    from pinnslab.components import OPTIMIZERS

    return OPTIMIZERS.get(spec.name)(params, spec)


__all__ = ["build_adam", "build_lbfgs", "build_optimizer"]
