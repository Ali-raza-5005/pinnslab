"""The two evaluators a :class:`~pinnslab.search.loop.Search` can be driven by.

Both turn a list of :class:`RunConfig` into one fitness each, and they are
interchangeable — which is the point. :class:`SequentialEvaluator` is the
oracle: it runs each candidate through the *ordinary* ``build_trainer`` path,
so what it reports is by construction what a single reproduction run reports.
:class:`BatchedEvaluator` is what makes a real search affordable.

Which to use
------------
**Sequential** for: a fitness measured against ground truth (``rel_l2`` needs
the benchmark's eval grid per candidate); architecture search (candidates
differ in shape and do not batch); any space that changes stage structure; and
any number going into a paper where you want the plain path to have produced it.

**Batched** for: sampling, loss weighting, activations — everything where every
candidate's network has the same shape and the fitness is the training
objective. See :mod:`pinnslab.search.population` for the measured speedup and
its limits.

The batched path fixes the point count across the population
------------------------------------------------------------
A batched matmul needs one shape, so every candidate must draw the same number
of points. That reads like a limitation on the flagship search axis — "how many
collocation points?" — but it is not the one it appears to be: DESIGN.md §8
already requires an **identical collocation point count** across compared
methods for the comparison to be fair. A sampling search at fixed budget is
therefore the scientifically correct default, and *where* the points go is the
question. Varying the count is a separate, compute-parity-relevant axis; run it
through :class:`SequentialEvaluator`, or group candidates by count and batch
within groups (DESIGN.md §6's shape-grouping note).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

from pinnslab.benchmarks.problem import build_problem
from pinnslab.components import RESIDUALS
from pinnslab.registry.config import RunConfig
from pinnslab.registry.run import Run
from pinnslab.registry.schema import RunStatus
from pinnslab.search.population import Ensemble, train_population
from pinnslab.search.spec import FitnessSpec
from pinnslab.training.build import assemble, build_trainer
from pinnslab.utils.device import RuntimeContext, configure_runtime
from pinnslab.utils.logging import get_logger

log = get_logger(__name__)


class SequentialEvaluator:
    """One candidate at a time, through the ordinary single-run path.

    With ``root`` set, every candidate becomes a real run directory — as
    inspectable, resumable and provenance-stamped as any other run (CLAUDE.md
    rule 7), which is what lets a search's winner be audited afterwards.
    Without one the runs go to scratch, which is what a quick local probe wants.
    """

    def __init__(
        self,
        fitness: FitnessSpec | None = None,
        *,
        root: str | Path | None = None,
    ) -> None:
        self.fitness = fitness or FitnessSpec()
        self.root = Path(root) if root else None

    def __call__(self, configs: Sequence[RunConfig], steps: int) -> list[float]:
        return [self._one(cfg, steps) for cfg in configs]

    def _one(self, cfg: RunConfig, steps: int) -> float:
        cfg = with_step_budget(cfg, steps)
        run_id = f"{cfg.identity_hash()[:12]}_s{cfg.seed}_n{steps}"
        try:
            ctx = configure_runtime(cfg)
            root = self.root or _scratch()
            run = Run.create_or_resume(cfg, root, run_id)
            row = build_trainer(cfg, ctx, run).fit()
        except Exception as exc:  # noqa: BLE001 - a bad candidate is data
            # A configuration that cannot train is a legitimate search result,
            # not a crash of the search: DESIGN.md §11 says failures are data.
            # The loop turns non-finite into the generation's penalty.
            log.warning("candidate %s failed to evaluate: %s", run_id, exc)
            return math.nan
        if row.status is not RunStatus.COMPLETED:
            return math.nan
        return float(row.final_metrics.get(self.fitness.metric, math.nan))


class BatchedEvaluator:
    """Every candidate trained simultaneously in one graph.

    The fitness is the **training objective**, not a held-out metric: this path
    never touches a reference solution. That is deliberate and is the dividing
    line between the two evaluators — a search whose fitness is ``rel_l2``
    against ground truth belongs in :class:`SequentialEvaluator`.

    Residual terms are reused unchanged from the config's registered
    ``ResidualSpec``s. They work here because :mod:`pinnslab.physics.diffops`
    indexes with ``...``, so a residual written for ``(N, m)`` outputs serves
    ``(P, N, m)`` without knowing a population exists.
    """

    def __init__(self, *, lr: float | None = None) -> None:
        self.lr = lr

    def __call__(self, configs: Sequence[RunConfig], steps: int) -> list[float]:
        if not configs:
            return []
        # Before assembling anything: an unsupported config should be told so,
        # not discovered through whatever error assemble happens to raise first.
        _reject_unsupported(configs)
        # configure_runtime per candidate, not once for the population. It
        # reseeds the global RNG, and `assemble` draws the network's initial
        # weights from it — so building candidates back to back would give
        # candidate k an initialisation that depends on how many candidates
        # preceded it. That makes a config's fitness a function of its position
        # in the batch, which silently breaks the candidate cache: the cache is
        # keyed on (config_hash, steps), and the same key would name different
        # experiments. Caught by the batched-vs-sequential equivalence test.
        parts, ctx = [], None
        for cfg in configs:
            ctx = configure_runtime(cfg)
            parts.append(assemble(cfg, ctx))
        assert ctx is not None

        net_name = parts[0].problem.solution_net
        ensemble = Ensemble([p.nets[net_name] for p in parts])
        points, offsets = _stack_points(parts, configs, ctx)
        residual = _population_residual(configs[0], net_name, offsets)

        lr = self.lr if self.lr is not None else configs[0].stages[0].optimizers[0].lr
        return train_population(
            ensemble, points, residual, steps=steps, lr=lr
        ).tolist()


# -- building the batched residual ---------------------------------------------


class _EnsembleState:
    """Just enough of a ``TrainState`` for a registered residual term.

    Terms read ``state.nets[name]`` and call it. Handing them the ensemble
    means one term evaluates the whole population at once, and the term itself
    is untouched.
    """

    def __init__(self, nets: dict, extra_params: dict, generator, dtype, device):
        self.nets = nets
        self.extra_params = extra_params
        self.generator = generator
        self.dtype = dtype
        self.device = device
        self.points: dict = {}
        self.scratch: dict = {}
        self.step = 0


def _stack_points(
    parts, configs, ctx: RuntimeContext
) -> tuple[torch.Tensor, dict[str, slice]]:
    """Each candidate's own cloud, stacked, plus where each group sits in it.

    The groups are concatenated in sorted name order so the slice map is stable
    across candidates and across processes — a resumed search must lay the
    batch out identically or the residual terms silently swap point groups.

    The candidate's own networks are handed to the sampler, because an adaptive
    one may score candidate points with them even on its first draw. Note the
    cloud is drawn **once**: :func:`train_population` does not resample, so a
    search whose subject is *resampling* (rather than the initial distribution)
    belongs in :class:`SequentialEvaluator`.
    """
    clouds, layouts = [], []
    for part, cfg in zip(parts, configs, strict=True):
        state = _EnsembleState(
            part.nets,
            part.extra_params,
            torch.Generator().manual_seed(cfg.seed),
            ctx.dtype,
            ctx.device,
        )
        part.on_resample(state)
        groups = state.points
        names = sorted(groups)
        layouts.append(tuple((name, len(groups[name])) for name in names))
        clouds.append(torch.cat([groups[name] for name in names], dim=0))

    if len(set(layouts)) != 1:
        raise ValueError(
            "batched evaluation needs one point-group layout across the "
            f"population, got {len(set(layouts))} distinct ones: "
            f"{sorted(set(layouts))}. Hold the point counts fixed in the search "
            "space — DESIGN.md §8 requires an identical collocation count "
            "across compared methods anyway — or use SequentialEvaluator."
        )

    offsets, start = {}, 0
    for name, count in layouts[0]:
        offsets[name] = slice(start, start + count)
        start += count
    return torch.stack(clouds), offsets


def _population_residual(cfg: RunConfig, net_name: str, offsets):
    """Every declared term, on its own point groups, concatenated to ``(P, N)``.

    Concatenated rather than weighted because :func:`train_population` takes
    the mean of squares, which *is* the default ``mean`` weighting. A config
    asking for NTK, causal or self-adaptive weighting is refused up front by
    :func:`_reject_unsupported` rather than silently trained under a different
    objective than it declared.
    """
    problem = build_problem(cfg.problem)
    built = [
        (RESIDUALS.get(spec.kind)(spec, problem), spec.points)
        for spec in cfg.residuals.values()
    ]

    def residual(ensemble: Ensemble, points: torch.Tensor) -> torch.Tensor:
        state = _EnsembleState(
            {net_name: ensemble}, {}, None, points.dtype, points.device
        )
        rows = []
        for term, groups in built:
            selected = torch.cat(
                [points[:, offsets[g], :] for g in sorted(groups)], dim=1
            )
            rows.append(term(state, selected))
        return torch.cat(rows, dim=1)

    return residual


def _reject_unsupported(configs: Sequence[RunConfig]) -> None:
    """Say what this path cannot do, instead of quietly doing something else."""
    first = configs[0]
    if first.weighting.kind != "mean":
        raise ValueError(
            f"the batched evaluator applies the plain mean weighting, but this "
            f"config declares {first.weighting.kind!r}. Training under a "
            "different objective than the config declares would make the "
            "search's fitness incomparable with its own reproduction runs. Use "
            "SequentialEvaluator, or pass a custom population residual to "
            "train_population."
        )
    if any(cfg.extra_params for cfg in configs):
        raise ValueError(
            "the batched evaluator does not carry inverse-problem parameters "
            "(extra_params); those are per-candidate trainable tensors that "
            "would need stacking alongside the network. Use SequentialEvaluator."
        )
    structures = {
        tuple(sorted((n, s.kind, s.points) for n, s in cfg.residuals.items()))
        for cfg in configs
    }
    if len(structures) != 1:
        raise ValueError(
            "every candidate must declare the same residual terms to be batched "
            f"together, got {len(structures)} distinct structures. A search over "
            "residual structure belongs in SequentialEvaluator."
        )


def with_step_budget(cfg: RunConfig, steps: int) -> RunConfig:
    """Retarget a config's stages onto a fidelity rung's total step budget.

    Proportional across stages, so an Adam->L-BFGS schedule keeps its shape at
    every rung rather than becoming pure Adam at the cheap rungs and something
    structurally different at the expensive ones — which would make the rungs
    measure different things and successive halving meaningless. A stage that
    rounds to zero is dropped; a stage of no steps is not a stage.
    """
    total = cfg.total_steps
    if total == steps:
        return cfg
    scaled = []
    for stage in cfg.stages:
        allotted = round(stage.steps * steps / total)
        if allotted:
            scaled.append(stage.model_copy(update={"steps": allotted}))
    if not scaled:
        scaled = [cfg.stages[0].model_copy(update={"steps": max(1, steps)})]
    return cfg.model_copy(update={"stages": scaled})


def _scratch() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="pinnslab_search_"))


__all__ = ["BatchedEvaluator", "SequentialEvaluator", "with_step_budget"]
