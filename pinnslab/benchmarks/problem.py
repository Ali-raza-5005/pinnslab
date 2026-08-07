"""What a frozen benchmark is.

DESIGN.md §3: benchmarks are *frozen*. The PDE, its domain, its boundary and
initial conditions and its reference solution are fixed so that every paper
compares against the same problem — a benchmark that drifts between papers
makes the author's own results incomparable, which is a worse failure than any
single wrong number.

What a config may vary is which benchmark, plus the handful of physical
constants the benchmark itself declares varyable (``ProblemSpec.options`` ->
:attr:`Problem.params`). Resolution, sampling and architecture are *not*
properties of the problem and live elsewhere in the config.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from pinnslab.components import PROBLEMS
from pinnslab.geometry import Domain
from pinnslab.registry.config import ProblemSpec, ResidualSpec

if TYPE_CHECKING:
    from pinnslab.training.trainer import TrainState

#: A residual term: ``(state, points) -> (N,)``. Per-point, never reduced
#: (CLAUDE.md rule 5). The points arrive as an argument rather than being drawn
#: inside, because L-BFGS's line search may invoke the closure several times per
#: step and a residual that resampled itself would break it.
ResidualTerm = Callable[["TrainState", torch.Tensor], torch.Tensor]

#: A registered residual *factory*: ``(spec, problem) -> ResidualTerm``. It
#: takes the problem so that physical constants have exactly one home — a
#: residual reading ``nu`` from its own options could disagree with the
#: reference solution that was computed with a different one, and the run would
#: look perfectly healthy while solving a different equation than it reports.
ResidualFactory = Callable[[ResidualSpec, "Problem"], ResidualTerm]


@dataclass(frozen=True)
class Problem:
    """A frozen benchmark: where it lives, what solves it, what is true."""

    name: str
    domain: Domain
    #: Physical constants, resolved from ``ProblemSpec.options`` and defaults.
    #: Recorded so a residual and the reference solution cannot disagree.
    params: Mapping[str, float] = field(default_factory=dict)
    #: ``(N, d) -> (N, 1)`` exact or high-accuracy solution, or ``None`` when
    #: the problem has no reference (then rel-L2 is simply not reportable).
    reference: Callable[[torch.Tensor], torch.Tensor] | None = None
    #: Default evaluation grid resolution, per coordinate.
    eval_resolution: tuple[int, ...] = ()
    #: Which network in ``state.nets`` the reference solution describes. Named
    #: rather than assumed, so a multi-net run (per-field, per-subdomain) is not
    #: silently scored against whichever network happens to be called "u".
    solution_net: str = "u"

    def reference_at(self, points: torch.Tensor) -> torch.Tensor:
        if self.reference is None:
            raise ValueError(
                f"benchmark {self.name!r} has no reference solution, so accuracy "
                "against ground truth cannot be reported for it"
            )
        return self.reference(points)


def build_problem(spec: ProblemSpec) -> Problem:
    """``ProblemSpec`` -> the registered benchmark it names."""
    return PROBLEMS.get(spec.name)(spec)


def resolve_params(
    spec: ProblemSpec, defaults: Mapping[str, float], *, name: str
) -> dict[str, float]:
    """Merge ``spec.options`` over ``defaults``, rejecting unknown constants.

    A typo'd physical constant is the worst kind of silent failure available
    here: ``{"mu": 0.01}`` where ``nu`` was meant leaves the run solving the
    default equation while the recorded config claims otherwise, and every
    number downstream is then wrong and self-consistent.
    """
    unknown = sorted(set(spec.options) - set(defaults))
    if unknown:
        raise ValueError(
            f"benchmark {name!r} has no parameter(s) {unknown}; it declares "
            f"{sorted(defaults)}"
        )
    return {key: float(spec.options.get(key, value)) for key, value in defaults.items()}


__all__ = [
    "Problem",
    "ResidualFactory",
    "ResidualTerm",
    "build_problem",
    "resolve_params",
]
