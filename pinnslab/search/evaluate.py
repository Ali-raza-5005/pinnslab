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
from pinnslab.utils.seeding import derive_seed, make_generator

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
        residual = _population_residual(configs, net_name, offsets, ctx)

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

    The generator is derived exactly as ``Trainer`` derives its own —
    ``derive_seed(seed, "trainer", identity_hash())`` — and not from ``cfg.seed``
    directly (fixed 2026-08-29). It used to be
    ``torch.Generator().manual_seed(cfg.seed)``, which meant the two evaluators
    drew *different* clouds for one config: harmless for ranking, but it made
    "rerun the winner sequentially and you get the search's number" false, and
    the whole reason :class:`SequentialEvaluator` is called the oracle is that
    the batched path is supposed to be reproducible through it.
    """
    clouds, layouts = [], []
    for part, cfg in zip(parts, configs, strict=True):
        state = _EnsembleState(
            part.nets,
            part.extra_params,
            make_generator(
                derive_seed(cfg.seed, "trainer", cfg.identity_hash()), device="cpu"
            ),
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


def _population_residual(
    configs: Sequence[RunConfig], net_name: str, offsets, ctx: RuntimeContext
):
    """Every declared term, on its own point groups, rescaled and concatenated.

    Why the rescaling, and why it is not cosmetic
    ---------------------------------------------
    :func:`train_population` reduces with **one pooled mean of squares** over
    everything the residual returns. ``MeanWeighting`` — the objective the
    config declares and the single-run path minimises — is
    ``sum_k coeff_k * mean(r_k ** 2)``: a mean *per term*, then a sum. Those
    two are equal only when every term has the same point count, which is
    essentially never. This module previously concatenated the terms raw and
    asserted in a comment that the pooled mean "*is* the mean weighting"; on
    ``examples/configs/burgers_uniform.yaml`` (pde on 1150 points, ic on 100,
    bc on 50) the two objectives differed by 6.3x, and the boundary term was
    weighted 26x lower in the batched objective than in the one the config
    asked for. A search run that way optimises something no reproduction run
    will reproduce.

    The fix keeps the single pooled reduction (so ``train_population``'s
    contract, and every proof of population independence built on it, is
    untouched) and puts the weighting into the rows instead. Scaling term ``k``
    by ``sqrt(coeff_k * N_total / N_k)`` gives

        pooled_mean = (1 / N_total) * sum_k sum_i coeff_k * (N_total / N_k) r_ki^2
                    = sum_k coeff_k * mean(r_k ** 2)

    which is ``MeanWeighting`` exactly. The scale is per **candidate** as well
    as per term, so a search over ``weighting.coefficients`` — one of
    DESIGN.md §6's four directions — batches correctly instead of silently
    training every candidate under ``configs[0]``'s coefficients.

    A config asking for NTK, causal or self-adaptive weighting is refused up
    front by :func:`_reject_unsupported` rather than approximated here.
    """
    problem = build_problem(configs[0].problem)
    built = [
        (name, RESIDUALS.get(spec.kind)(spec, problem), spec.points)
        for name, spec in configs[0].residuals.items()
    ]
    counts = {
        name: sum(offsets[g].stop - offsets[g].start for g in groups)
        for name, _, groups in built
    }
    total = sum(counts.values())

    def _scale(name: str, cfg: RunConfig) -> float:
        coefficient = cfg.weighting.coefficients.get(name, 1.0)
        if coefficient < 0.0:
            # The scale is a square root, so a negative coefficient cannot be
            # folded into the residual rows at all. MeanWeighting accepts one
            # (it is a plain multiply), so this is a limit of the batched
            # reduction rather than of the objective — say so instead of
            # raising a domain error from inside math.sqrt.
            raise ValueError(
                f"weighting coefficient for {name!r} is {coefficient}; the "
                "batched evaluator folds coefficients into the residual as a "
                "square root and cannot represent a negative one. Use "
                "SequentialEvaluator."
            )
        return math.sqrt(coefficient * total / counts[name])

    # (P, 1) per term, so the multiply broadcasts across that term's points.
    scales = {
        name: torch.tensor(
            [[_scale(name, cfg)] for cfg in configs],
            dtype=ctx.dtype,
            device=ctx.device,
        )
        for name, _, _ in built
    }

    population = len(configs)

    def residual(ensemble: Ensemble, points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] != population:
            # The per-candidate scales are (P, 1) and would broadcast silently
            # against a (P', N) row, turning a one-candidate call into a P-row
            # result nobody asked for. Callers evaluating a subset (the
            # "separate" arm of a speedup benchmark) must build a residual for
            # exactly the configs they are passing.
            raise ValueError(
                f"this residual was built for {population} candidate(s) but "
                f"got points for {points.shape[0]}. Build one per population "
                "you actually evaluate: _population_residual(configs[i:i+1], "
                "...) for a single candidate."
            )
        state = _EnsembleState(
            {net_name: ensemble}, {}, None, points.dtype, points.device
        )
        rows = []
        for name, term, groups in built:
            selected = torch.cat(
                [points[:, offsets[g], :] for g in sorted(groups)], dim=1
            )
            rows.append(term(state, selected) * scales[name])
        return torch.cat(rows, dim=1)

    return residual


def _reject_unsupported(configs: Sequence[RunConfig]) -> None:
    """Say what this path cannot do, instead of quietly doing something else.

    Everything checked here is a field the batched path reads from
    ``configs[0]`` and then applies to the whole population. Left unchecked, a
    search space touching any of them produces the worst failure available: a
    full set of plausible fitnesses for configurations that were never
    evaluated, each cached under a ``config_hash`` naming a run that did not
    happen.
    """
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
    if any(cfg.weighting.options for cfg in configs):
        raise ValueError(
            "the batched evaluator reads only weighting.coefficients; this "
            f"config also sets options {sorted(first.weighting.options)}, which "
            "would be recorded and never applied. Use SequentialEvaluator."
        )
    structures = {
        tuple(sorted((n, s.kind, s.points, s.net) for n, s in cfg.residuals.items()))
        for cfg in configs
    }
    if len(structures) != 1:
        raise ValueError(
            "every candidate must declare the same residual terms to be batched "
            f"together, got {len(structures)} distinct structures. A search over "
            "residual structure belongs in SequentialEvaluator."
        )

    # The residual terms are built once, from configs[0].problem, and handed to
    # the whole population. A space varying a physical constant would train
    # everyone on candidate 0's equation while recording each candidate's own
    # value in its config.
    problems = {c.problem.model_dump_json() if c.problem else "" for c in configs}
    if len(problems) != 1:
        raise ValueError(
            "every candidate must name the same problem and the same physical "
            f"constants to be batched together, got {len(problems)} distinct "
            "ones. The residual terms are built once for the whole population, "
            "so a varying constant would be recorded but never applied. Use "
            "SequentialEvaluator."
        )

    _reject_unsupported_optimisation(configs)


def _reject_unsupported_optimisation(configs: Sequence[RunConfig]) -> None:
    """``train_population`` is a flat loop of plain Adam steps. Say so.

    It takes one scalar ``lr`` for the whole stacked population, so each of
    these would otherwise be read off ``configs[0]`` and applied to everyone:

    * a **multi-stage** schedule. An Adam->L-BFGS config evaluated here gets
      Adam only, so the search would rank candidates under a training procedure
      no reproduction run performs. L-BFGS genuinely does not batch: its
      curvature history and line search are global, so one step over stacked
      parameters couples every candidate.
    * a **different optimizer, its options, or a per-candidate learning rate.**
      The optimizer axis is one of DESIGN.md 6's four research directions, and
      searching it here would score every candidate at ``configs[0]``'s lr.
    * an **ascent** optimizer or ``max_grad_norm``. The first is half of a
      min-max scheme whose other half this path cannot run; the second couples
      the population (see ``train_population``).
    * ``resample_every``. The cloud is drawn once here, so a search whose
      subject is resampling would never see a resample.
    """
    if any(len(cfg.stages) != 1 for cfg in configs):
        raise ValueError(
            "the batched evaluator runs one flat loop of Adam steps, but this "
            "config declares several stages. Evaluating an Adam->L-BFGS "
            "schedule as Adam alone would rank candidates under a training "
            "procedure no reproduction run performs. Use SequentialEvaluator."
        )
    if any(len(cfg.stages[0].optimizers) != 1 for cfg in configs):
        raise ValueError(
            "the batched evaluator drives one optimizer over the stacked "
            "population; a stage with several (a min-max or self-adaptive "
            "scheme) belongs in SequentialEvaluator."
        )

    specs = {cfg.stages[0].optimizers[0].model_dump_json() for cfg in configs}
    if len(specs) != 1:
        raise ValueError(
            f"every candidate must declare the same optimizer to be batched "
            f"together, got {len(specs)} distinct ones. One Adam runs over the "
            "stacked population with a single scalar learning rate, so a space "
            "varying lr (or any other optimizer field) would score every "
            "candidate at the first one's setting. Use SequentialEvaluator."
        )

    spec = configs[0].stages[0].optimizers[0]
    if spec.name != "adam":
        raise ValueError(
            f"the batched evaluator runs Adam; this config asks for "
            f"{spec.name!r}. Use SequentialEvaluator."
        )
    if spec.options:
        raise ValueError(
            f"the batched evaluator passes no options to Adam, but this config "
            f"sets {sorted(spec.options)}. Use SequentialEvaluator."
        )
    if spec.direction != "min":
        raise ValueError(
            "the batched evaluator only descends; direction='max' is half of a "
            "min-max scheme whose other half it cannot run. Use "
            "SequentialEvaluator."
        )
    if spec.max_grad_norm is not None:
        raise ValueError(
            "global gradient-norm clipping couples the population: one norm "
            "over all P candidates means a single diverging candidate damps "
            "every other one's step. Use SequentialEvaluator."
        )
    if any(cfg.stages[0].resample_every for cfg in configs):
        raise ValueError(
            "train_population draws the collocation cloud once and never "
            "resamples, but this config sets resample_every. A search whose "
            "subject is resampling must see the resampling happen: use "
            "SequentialEvaluator."
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
