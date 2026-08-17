"""RAD — residual-based adaptive sampling, as a paper repo would write it.

This file is the worked answer to "how do I add a sampler?". It is deliberately
**not** in ``pinnslab/``: CLAUDE.md rule 2 says method code is born in a paper's
own ``src/method/`` and is promoted into the library only when a second paper
needs it. What the library owns is the seam
(:mod:`pinnslab.geometry.samplers`); what a paper owns is the method. Copy this
file into ``paper-01/src/method/``, change the algorithm, change nothing else.

The method (Wu, Zhu, Tan, Kartha & Lu, *CMAME* 2023): draw a large uniform pool,
score it with the PDE residual, and keep points with probability

    p(x) ∝ eps(x)^k / E[eps^k] + c

``k`` sharpens the concentration, ``c`` keeps a uniform floor so the cloud never
collapses onto the front and forgets the rest of the domain. ``k=0`` or a large
``c`` reduces to uniform sampling, which is what makes this a *one-knob* ablation
against its own baseline rather than a different method.

Three things this file demonstrates that any sampler must get right
------------------------------------------------------------------
1. **Draw from ``state.generator``, never from ``torch.rand``.** That stream is
   checkpointed; anything else puts the point cloud outside the resume and
   silently breaks DESIGN.md §5.
2. **Score with a registered residual**, built from the *problem*, so the
   sampler cannot disagree with the term the loss uses about what the equation
   is (a sampler carrying its own ``nu`` is a run solving two different PDEs).
3. **Report state through ``state_dict``** if it accumulates any. Here it is
   only a generation counter, but the trainer checkpoints it, and a resumed run
   therefore continues the sampler's schedule instead of restarting it.

Usage — from a config, with no edit to pinnslab::

    sampling:
      points:
        interior:
          region: interior
          n: 2000
          strategy: rad
          options: {k: 1.0, c: 1.0, pool: 20000}

and make sure this module is imported before the config is built, which is what
``--register`` does in ``scripts/run.py`` / ``scripts/run_sweep.py``.
"""

from __future__ import annotations

import torch

from pinnslab.benchmarks.problem import Problem
from pinnslab.components import RESIDUALS
from pinnslab.geometry.samplers import Sampler, register_sampler
from pinnslab.registry.config import PointSetSpec, ResidualSpec


@register_sampler("rad")
class RADSampler(Sampler):
    """Residual-adaptive distribution over a uniform pool."""

    def __init__(self, spec: PointSetSpec, problem: Problem) -> None:
        options = dict(spec.options)
        self.k = float(options.pop("k", 1.0))
        self.c = float(options.pop("c", 1.0))
        # 10x the point count by default: big enough that the empirical
        # residual distribution is not itself noise, small enough that scoring
        # it costs less than the training step it feeds.
        self.pool = int(options.pop("pool", 10 * spec.n))
        residual_kind = str(options.pop("residual", f"{problem.name}.pde"))
        if options:
            raise TypeError(
                f"rad got unknown option(s) {sorted(options)}; it takes "
                "k, c, pool and residual"
            )
        if self.pool < spec.n:
            raise ValueError(
                f"rad needs a pool at least as large as n, got pool={self.pool} "
                f"for n={spec.n}: it selects a subset of the pool"
            )

        self.spec = spec
        self.problem = problem
        # Built through the registry, so the sampler scores exactly the term the
        # loss minimises — including its physical constants, which live on the
        # problem and nowhere else.
        self.term = RESIDUALS.get(residual_kind)(
            ResidualSpec(kind=residual_kind, points=(spec.region,)), problem
        )
        self.generations = 0

    def __call__(
        self, state, current: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.generations += 1
        if current is None:
            # Generation 0 has no residual to score yet, so it is the same
            # uniform draw the baseline makes: RAD is a *modification* of the
            # control arm rather than a different procedure, which is what makes
            # the mechanism ablation (k=0, or a large c) meaningful.
            #
            # Note it does not produce the *identical* cloud to the baseline
            # run. The trainer's stream is derived from `(seed, "trainer",
            # config_hash)`, so two conditions never share an RNG stream by
            # construction — deliberate (an accidental correlation between arms
            # is harder to notice than an obvious one), and the reason a
            # comparison here is unpaired and needs its >=5 seeds.
            return self._uniform(state, self.spec.n)

        pool = self._uniform(state, self.pool)
        weights = self._probabilities(state, pool)
        chosen = torch.multinomial(
            weights, self.spec.n, replacement=False, generator=state.generator
        )
        return pool[chosen]

    def _uniform(self, state, n: int) -> torch.Tensor:
        return self.problem.domain.sample(
            self.spec.region,
            n,
            generator=state.generator,
            strategy="pseudo",
            dtype=state.dtype,
            device=state.device,
        )

    def _probabilities(self, state, pool: torch.Tensor) -> torch.Tensor:
        """``eps^k / mean(eps^k) + c``, normalised.

        Not under ``no_grad``: the residual differentiates the network with
        respect to its *inputs*, so the graph is needed to compute it at all.
        The result is detached immediately — a point cloud is data, not
        something the optimizer may pull on.
        """
        with torch.enable_grad():
            residual = self.term(state, pool).detach().abs()
        weights = residual**self.k
        mean = weights.mean()
        # A network that fits the pool exactly gives an all-zero residual; the
        # uniform floor is then the whole distribution, which is the right
        # answer rather than a division by zero.
        weights = weights / mean if mean > 0 else torch.zeros_like(weights)
        return weights + self.c

    def state_dict(self) -> dict:
        return {"generations": self.generations}

    def load_state_dict(self, payload: dict) -> None:
        self.generations = int(payload["generations"])


__all__ = ["RADSampler"]
