"""Built-in optimizer factories, and the capability protocol the loop reads.

Registered, not hard-coded: a paper repo adds ``@register_optimizer("soap")`` in
its own ``src/method/`` and the config accepts it with zero edits here
(CLAUDE.md rule 9).

The capability protocol (added 2026-08-29, DESIGN.md §4)
--------------------------------------------------------
``Trainer`` used to decide how to drive an optimizer with
``isinstance(opt, torch.optim.LBFGS)``. That is a hole in the registration seam:
an optimizer that is not gradient-based *can* be registered, and would then be
driven down a path that never shows it the objective — it would silently do
nothing useful. Population methods (CSO, PSO, DE over network weights) are
exactly that shape, and optimizer schedules are one of the four research
directions of DESIGN.md §6, so the gap is on the main line, not in a corner.

An optimizer now declares what it needs with two optional class attributes, and
the loop reads them through the predicates below:

``requires_closure: bool = False``
    The loop calls ``opt.step(closure)`` instead of ``opt.step()``. Declare this
    for anything that must evaluate the objective itself — a line search, a
    trust region, a population.

``uses_gradients: bool = True``
    Set ``False`` for a derivative-free method. The loop then skips
    ``backward()`` entirely, and *refuses* the config fields that only mean
    something for a gradient (``max_grad_norm``, ``direction: max``) rather than
    ignoring them. Refuse, never approximate — DESIGN.md §6 CORRECTION 2.

There is deliberately no second "bind the objective" hook. The optimizer already
owns its parameter tensors, so a population method writes candidate *p* into
``param.data`` and calls the closure to score it; one mechanism covers both the
line-search and the population case.

Two contracts a closure-based optimizer must honour
---------------------------------------------------
1. ``step(closure)`` **must return the objective value**, not ``None``. The loop
   has no other source for the step's loss, and a silent ``None`` would put
   ``nan`` in the trace.
2. **The last closure call must be at the parameters the optimizer leaves
   installed.** The loop takes the step's ``loss`` *and* its per-residual trace
   from that final evaluation, so a population method must re-evaluate its
   winner after installing it. This is what keeps a derivative-free step free of
   the off-by-one that the first-order path carries (see ``trainer._step_first_
   order``): the number reported for step *k* describes θ(k), not θ(k−1).
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


# -- the capability predicates the training loop dispatches on ----------------


def requires_closure(optimizer: torch.optim.Optimizer) -> bool:
    """Whether the loop must call ``step(closure)`` rather than ``step()``.

    ``torch.optim.LBFGS`` is named explicitly because it predates this protocol
    and we do not monkey-patch torch. It is the *only* isinstance check left,
    and it is a compatibility shim rather than a decision about what may exist.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        return True
    return bool(getattr(optimizer, "requires_closure", False))


def uses_gradients(optimizer: torch.optim.Optimizer) -> bool:
    """Whether ``.grad`` means anything for this optimizer. Default: yes."""
    return bool(getattr(optimizer, "uses_gradients", True))


__all__ = [
    "build_adam",
    "build_lbfgs",
    "build_optimizer",
    "requires_closure",
    "uses_gradients",
]
