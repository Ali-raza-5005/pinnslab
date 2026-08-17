"""Collocation samplers — the registry the resampler actually consults.

DESIGN.md §6 makes *where the points go* the first research direction, so this
is the seam a sampling paper extends. CLAUDE.md rule 9: a new sampler is one new
file in the paper repo with one ``@register_sampler`` line, and **zero edits
here**.

The contract, deliberately the same shape as ``benchmarks.problem.ResidualTerm``
so there is one convention to learn:

* a **factory** ``(spec: PointSetSpec, problem: Problem) -> Sampler`` is what
  gets registered, and it is handed the whole declared spec, so ``n``,
  ``region`` and the free-form ``options`` are all available at construction;
* the **sampler** is called ``sampler(state, current) -> (n, dim)`` once per
  resample, where ``current`` is the cloud this group is holding right now
  (``None`` on the first draw).

``state`` is the ordinary :class:`~pinnslab.training.trainer.TrainState`, which
is what makes *adaptive* sampling expressible without a second abstraction: it
carries the networks (so a residual can be evaluated at candidate points), the
current step and stage, the run's config, and — the one that matters for
reproducibility — the trainer's own ``generator``. **Draw from that generator
and from nothing else.** A sampler that reaches for ``torch.rand`` or
``np.random`` puts the point cloud outside the checkpoint and quietly breaks
DESIGN.md §5's bit-exact resume.

Why samplers carry ``state_dict``/``load_state_dict``
-----------------------------------------------------
A stateless sampler needs neither; the base class supplies empty ones. But a
sampler that accumulates anything — a residual EMA, a pool it grows, a counter —
holds part of the experiment, and a resumed run that dropped it would continue a
*different* experiment under the same ``run_id``. The trainer checkpoints
whatever the resample hook reports here, alongside the point cloud itself
(``training/checkpoint.py``).

Determinism note for the quasirandom strategies
-----------------------------------------------
``halton``, ``hammersley`` and ``sobol`` return the same points for a given
``n`` no matter the RNG, so resampling with one is a no-op that still costs the
draw — see :data:`pinnslab.geometry.adapters.DETERMINISTIC_STRATEGIES`. That is
a property of the strategy, not a bug here, and a sampling paper should know it
before using one as a baseline for a resampling experiment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from pinnslab.components import SAMPLERS, register_sampler
from pinnslab.geometry.adapters import GEOMETRY_STRATEGIES

if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from pinnslab.benchmarks.problem import Problem
    from pinnslab.registry.config import PointSetSpec
    from pinnslab.training.trainer import TrainState


class Sampler:
    """Draws one point group. Subclass, or just register any callable factory.

    The default state methods are empty because most samplers are a pure
    function of ``(state, current)``; override them the moment yours is not.
    """

    def __call__(
        self, state: TrainState, current: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class GeometrySampler(Sampler):
    """The built-in draws: uniform pseudorandom and the quasirandom sequences.

    Every one of them is DeepXDE's, reached through
    :class:`~pinnslab.geometry.adapters.Domain` so that this module never sees a
    DeepXDE object (CLAUDE.md rule 1). It ignores ``current``: a fresh draw does
    not depend on the cloud it replaces.
    """

    def __init__(self, spec: PointSetSpec, problem: Problem) -> None:
        self.domain = problem.domain
        self.region = spec.region
        self.n = spec.n
        self.strategy = spec.strategy
        if spec.options:
            raise TypeError(
                f"sampler {spec.strategy!r} takes no options, got "
                f"{sorted(spec.options)}. Options are forwarded verbatim to the "
                "registered factory, so an unread one is a typo or a wrong "
                "sampler name."
            )

    def __call__(
        self, state: TrainState, current: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.domain.sample(
            self.region,
            self.n,
            generator=state.generator,
            strategy=self.strategy,
            dtype=state.dtype,
            device=state.device,
        )


# One registration per geometric strategy, rather than one registration that
# switches internally: the registry is the list of legal `strategy:` values, and
# a config naming an unregistered sampler should fail with that list in the
# message rather than reach DeepXDE and fail with a different one.
for _name in sorted(GEOMETRY_STRATEGIES):
    register_sampler(_name)(GeometrySampler)


def build_sampler(spec: PointSetSpec, problem: Problem) -> Sampler:
    """``PointSetSpec`` -> the sampler it names.

    The single lookup point, so "which samplers exist" has one answer and an
    unknown name reports the registered ones.
    """
    return SAMPLERS.get(spec.strategy)(spec, problem)


__all__ = ["GeometrySampler", "Sampler", "build_sampler", "register_sampler"]
