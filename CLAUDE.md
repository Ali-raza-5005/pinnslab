# CLAUDE.md — pinnslab

Standing rules for every session in this repo. Read `DESIGN.md` for the *why*.

## What this repo is
Personal PINN **methods** research infrastructure. Not a general-purpose library,
not for external adoption. Success metric = the author's papers-per-year, nothing
else. The research program is metaheuristic/population search over PINN
configuration space (sampling → weighting → optimizers → architecture).

## Hard rules (do not violate without updating DESIGN.md first)

1. **DeepXDE is imported in exactly ONE file**: `pinnslab/geometry/adapters.py`.
   No DeepXDE object escapes it — raw `torch.Tensor` only. Never wrap DeepXDE's
   `Model`/training loop; we own the loop.
2. **Promotion rule**: new method code is born in a paper repo's `src/method/`.
   It enters `pinnslab` ONLY when a second paper needs it. Not before.
3. **Deletion rule**: anything unused across two consecutive papers gets deleted.
4. **No hyperparameter is ever a Python literal in a script.** Every number lives
   in a validated, hashed config. **Config system is YAML + pydantic — no Hydra**
   (YAML on disk → pydantic validation on load → hash the validated object). The
   pydantic schema doubles as the metaheuristic search space.
5. **Residuals return per-point tensors `(N,)`, never scalars.** Reduction lives
   in the weighting object.
6. **`results/` is append-only.** Never overwrite or clean in place. Aggregation
   reads raw, writes to `analysis/`.
7. **Every result row records**: `pinnslab.__version__`, git SHA, config hash,
   `gpu_name`, `dtype`, `device_profile`, seed. Non-negotiable.
8. **float64 is the default dtype.** float32 only for throughput-bound search on
   T4 (see DESIGN.md §5), and never mixed within a comparison group.
9. **Extension is by registration, not inheritance** (`@register_*`). New method
   = one new file, zero edits to existing files.
10. **Golden tests must stay green and CPU-runnable in <2 min.** Run them before
    any commit that touches training/physics/losses.

## Every-session logging
- **`FRICTION.md`**: any time you must edit `pinnslab` core to accomplish a
  paper-level task, log one line (date, what was wanted, what had to change).
  This is the evidence base for where abstractions are wrong. Don't skip it.
- **`LOG.md`**: dated ~5-line weekly entry — what ran, what was learned, what's
  next, current phase, days left in timebox.

## Workflow
- Edit locally, commit to `main` (solo repo — no PRs, no branch protection).
- Bump a version tag when a paper needs to pin the current state.
- Never edit this library on Kaggle. Push → tag → `pip install git+...@tag`.
- Keep source files small and single-purpose.

## Reviewer defenses (must exist in any paper's PROTOCOL.md before P2 runs)
Compute parity (incl. search cost) · random-search baseline at matched budget ·
CMA-ES/DE serious-optimizer baseline · mechanism ablation · >=5 seeds,
median+IQR, explicit failure rate · equal tuning budget across methods.

## Tests — two commands
- **Before every commit**: `pytest -m "unit and not slow"`. The DESIGN.md §3 60s
  budget applies to *this* number.
- **Before every push/tag**: `pytest` — adds `slow` (subprocess tests, each
  paying a fresh `import torch`) and `golden`.

## Current status
Bootstrap phase. Build order in DESIGN.md §9. **Steps 1-5 are done — the
bootstrap is complete.** `utils/`, `registry/`, `training/` (checkpoint,
trainer, build, queue), `components.py`, `geometry/adapters.py`,
`models/mlp.py`, `physics/diffops.py`, `eval/metrics.py`,
`benchmarks/burgers.py`, the Burgers golden test, `notebooks/kaggle_runner.py`,
`viz/{style,aggregate,convergence,tables}.py`, `scripts/make_figures.py`, and
`search/{space,spec,algorithms,population,evaluate,cache,state,loop}.py`.

Hardened for use 2026-08-17 (**v0.2.0**, see CHANGELOG.md): the sampler seam
(`geometry/samplers.py`) is wired to the registry, the collocation cloud is
checkpointed, `scripts/{run,run_sweep,run_search,benchmark_population}.py` and
`examples/` exist, and CI runs both test commands. 451 tests.

Four things to read before touching the relevant area, because each records a
*measurement* that overrode an earlier design:
- **DESIGN.md §6** — the population evaluator is a **batched graph, not
  `vmap`**; `vmap` cannot host a PINN residual. Residuals are rank-agnostic
  (`diffops` indexes with `...`), so one residual serves a single run and a
  population. Global grad-norm clipping is refused: it couples candidates.
- **DESIGN.md §7** — the queue derives cell status from the results directory
  instead of writing a status column, and partitions workers statically.
- **DESIGN.md §8** — figure conventions and the palette; SciencePlots' default
  cycle was rejected on a colorblind-safety measurement, not on taste.
- **`viz/aggregate.band`** — `step` is intersected exactly, `wall_time` is
  interpolated onto a common grid. Seeds never share a timestamp, so the
  per-second figure has no other way to exist; the version that intersected them
  crashed on every real results directory while its test passed on a fixture
  where all seeds ran at identical speed.

**Next: P0 on paper 1 (sampling).** Infrastructure work from here is
paper-driven — log every core edit a paper task forces in `FRICTION.md`.

Both items that were open before the first real sweep are **fixed** (2026-08-17):
the `resample_every`-plus-resume gap (the cloud and the sampler's `state_dict`
are checkpointed, the queue's refusal is gone, and `tests/unit/test_resampling.py`
pins bit-exact resume for an adaptive sampler), and checkpoint retention
(`best.pt` no longer carries optimizer state — nothing resumes from it, and
Adam's moments were most of the file).
