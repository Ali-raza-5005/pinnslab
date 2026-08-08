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
Bootstrap phase. Build order in DESIGN.md §9. **Steps 1-4 are done**: `utils/`,
`registry/`, `training/` (checkpoint, trainer, build, queue), `components.py`,
`geometry/adapters.py`, `models/mlp.py`, `physics/diffops.py`,
`eval/metrics.py`, `benchmarks/burgers.py`, the Burgers golden test,
`notebooks/kaggle_runner.py`, `viz/{style,aggregate,convergence,tables}.py` and
`scripts/make_figures.py`. 333 tests.

The queue derives cell status from the results directory instead of writing a
status column, and partitions workers statically; a killed sweep is proven
bit-identical to an uninterrupted one (DESIGN.md §7). Figure conventions and
the *measured* palette decision are in DESIGN.md §8 — read that before touching
`viz/`; the palette is not a matter of taste and SciencePlots' default cycle
was rejected on a colorblind-safety measurement.

**Next: step 5** — `search/`: SearchSpec + vmap population evaluator +
outer-loop checkpoint + candidate cache + multi-fidelity. Then P0 on paper 1.

Before the first real sweep, in TESTS_TODO.md: checkpoint retention, and the
`resample_every`-plus-resume gap (collocation points are not checkpointed, so an
interrupted resampling run resumes on the wrong cloud). The second is now
*guarded* — `run_queue` refuses `resample_every` on a checkpointed run — but not
fixed, and it lands directly on paper 1.
