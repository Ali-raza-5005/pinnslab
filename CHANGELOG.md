# CHANGELOG

Notable changes per tag. A paper pins a tag (DESIGN.md §2), so what matters here
is what would change a *result*: anything that moves numbers, invalidates a
config hash, or changes what a checkpoint can be resumed from.

## v0.3.0 — 2026-08-28

A research-readiness audit before the first real experiments. The bootstrap ran
and every test passed; what the audit found is that several of those tests
pinned the *wrong* behaviour, and the failures were concentrated in one seam —
the batched search evaluator, where a field read off `configs[0]` was applied to
the whole population.

**Every config hash changes** (`EvalSpec` gained `target_mode`), and the batched
evaluator's fitness changes for any config whose point groups differ in size —
which is all of them. No results existed yet, so nothing on disk is invalidated;
this is the last moment that was true.

### Fixed — scientific correctness

- **The batched evaluator optimised a different objective than the config
  declared.** `_population_residual` concatenated every residual term and
  `train_population` took **one pooled mean of squares** over the result.
  `MeanWeighting` — what the single-run path minimises — is `sum_k coeff_k *
  mean(r_k**2)`: a mean *per term*, then a sum. Those agree only when every term
  has the same point count, which never happens. On the shipped
  `examples/configs/burgers_uniform.yaml` (pde on 1150 points, ic on 100, bc on
  50) the two objectives differed by **6.3x**, with the boundary term weighted
  **26x lower** than the config asked for. `weighting.coefficients` were dropped
  entirely. Terms are now rescaled by `sqrt(coeff_k * N_total / N_k)` before the
  pooled reduction, which reproduces `MeanWeighting` exactly, per candidate — so
  a search over loss weights (DESIGN.md §6's second direction) now works instead
  of silently training everyone under `configs[0]`'s coefficients. Verified:
  batched and single-run objectives now agree to `ratio == 1.0`.
- **The test that should have caught it pinned the bug instead.**
  `test_batched_and_sequential_agree_on_the_training_objective`, described in
  its own docstring as "THE test of this module", computed its "oracle" by
  re-implementing the pooled mean rather than calling the config's weighting
  object — so both sides computed the same wrong number. It now calls
  `part.weighting(residuals, state)`. This is the third instance in this repo of
  a test passing on a premise that cannot occur (see v0.2.0's `wall_time`
  fixture); the pattern is worth naming.
- **`Ensemble` ignored the network's activation and always used `tanh`.** A
  config declaring `activation: sin` was batched as a tanh network, so the
  search scored a candidate its own `config_hash` did not describe and no
  reproduction run would match. Measured disagreement on identical inputs for a
  depth-2 width-8 `sin` MLP: **7.9e-2**. The activation is now inferred from the
  members; a population mixing activations, or a member mixing them across
  layers, is refused rather than silently unified.

### Fixed — the metaheuristic layer

- **A diverged candidate could score better than every real one.**
  `Search._penalty` returned `10 * max(finite)`. For a *maximised* fitness,
  `_orient` negates the scores, so the finite values are negative and the
  penalty was an order of magnitude **better** than any candidate that trained:
  DE would have driven the population toward configurations that diverge, while
  the fitness curve looked like convergence. It also tied with the best when
  every score was 0.0. The penalty now steps past the worst finite score by a
  margin taken from the batch's own spread, which is strictly worse for any
  sign and stays on the batch's scale.
- **`FidelitySchedule.cost` under-reported the search's own compute by 21%.**
  It charged a promoted candidate the *increment* `rungs[r] - rungs[r-1]`, as a
  warm-started ladder would. Both evaluators retrain a survivor from scratch, so
  the real cost is `rungs[r]` again. On `rungs=(1000, 5000, 20000), keep=0.5,
  pop=16` it reported 108,000 inner steps against the 136,000 the loop spends —
  understating exactly the number DESIGN.md §8's compute-parity defence rests
  on. A new test asserts the bound and the loop's own measurement agree.

### Fixed — experimental fairness

- **The batched path silently ignored per-candidate differences it could not
  express.** A search space over `stages.0.optimizers.0.lr` ran without
  complaint and scored **every** candidate at the first one's learning rate,
  archiving distinct config hashes for one experiment and caching those
  fitnesses forever. The same held for the optimizer name and options, the
  problem's physical constants, `weighting.options`, multi-stage schedules (an
  Adam→L-BFGS config was evaluated as Adam alone), ascent optimizers,
  `max_grad_norm`, and `resample_every` — the last being the worst, since
  sampling is paper 1's subject and `train_population` never resamples. All are
  now refused with a message naming `SequentialEvaluator`.
- **Time-to-target borrowed `best_mode` for its direction.** A config keeping
  the highest value of one metric while targeting a low value of another
  recorded the target as reached at the first trace point — a reviewer-facing
  compute-parity number, wrong and self-consistent. `EvalSpec.target_mode` is
  now explicit and defaults to `min`.

### Added

- **Search timing and provenance.** DESIGN.md §11 makes search cost a
  first-class result ("a GA that burns 3000x compute winning is not a result")
  and nothing recorded it: the archive held fitnesses and no clock. `SearchState`
  now carries `total_seconds`, `total_inner_steps` and a rule-7 provenance
  block; `GenerationReport` carries `seconds`; `Evaluation` carries `seconds`
  and an honest `cached` flag. Totals accumulate across sessions, so a resumed
  search reports the whole search's cost rather than the last session's.
  `scripts/run_search.py` prints the measured cost beside the declared bound.
- **`tests/unit/test_algorithms_on_benchmarks.py`** (29 tests). The suite drove
  the algorithms through `Search` with stub evaluators, which proved the
  plumbing and said nothing about whether the optimiser optimises — a real gap
  when §8 makes "does it beat random search at matched budget?" the question a
  paper lives on. DE and random search now run on Sphere, Rosenbrock and
  Rastrigin in the unit cube, with box-constraint, non-collapse, knob-sensitivity,
  reproducibility and resume checks. Measured: DE reaches 1.2e-11 on Sphere and
  1.0e-4 on Rosenbrock, beating random search by 4x at 20 generations rising to
  3e11x at 300.
- **`scripts/validate_gpu.py`.** README's honest limit was that no GPU had ever
  run this code, which left DESIGN.md §5's determinism switches, §5's
  precision-by-GPU rule, §6's "20-50x on a T4" and §7's two-GPU strategy all
  unverified. One command now checks environment, bit-exact determinism,
  FP64/FP32 cost ratio, golden accuracy, bit-exact resume across a simulated
  session death, the batched speedup curve, and two-GPU concurrency — and prints
  a report to paste back. Smoke-tested on CPU with `--allow-cpu`.
- A population-size guard in `_population_residual`: a residual built for P
  candidates now refuses points for P′, instead of broadcasting a
  one-candidate call into P rows.

### Changed

- `pinnslab/utils/seeding.py` had a UTF-8 BOM and four double-encoded section
  marks (`Â§`). Repaired.
- 451 → 497 tests.

## v0.2.0 — 2026-08-17

The bootstrap was complete but three things stood between it and a first real
sweep: the per-second figure could not be drawn from real data, the sampler
registry was not wired to anything, and a resampling run could not be resumed.
All three are fixed, and the repo is now usable end to end from a clean clone.

### Fixed

- **The wall-clock convergence figure was impossible to produce.**
  `viz.aggregate.band` intersected exact float x values across seeds — correct
  for `step` (seeds share that grid), impossible for `wall_time` (no two seeds
  finish a step at the same second). The intersection was `{0.0}`, the log axis
  dropped it, and matplotlib raised *"Data has no positive values"* inside
  `savefig`, so `scripts/make_figures.py` crashed on its own documented
  invocation. Continuous axes are now interpolated onto a common grid: the union
  of the seeds' timestamps, clipped to the interval every seed measured (no
  extrapolation), interpolated in log space because that is the geometry the log
  y axis draws. Non-finite points are dropped per seed; a seed left with fewer
  than two points leaves the band and is reported through `n_used`.
- **The test that covered it could not fail.** Its fixture gave every seed
  `wall_time = step/10` — identical timings, which cannot occur. Seeds now carry
  realistic per-seed timings, and the regression test *saves* the figure,
  because the crash was at draw time.
- **`SAMPLERS` was a registry nobody read.** `build.py` passed a config's
  `strategy:` straight to DeepXDE, so the five geometric draws worked and
  adaptive sampling — the subject of paper 1 — could not be added without
  editing core, contradicting rule 9. `geometry/samplers.py` now owns the
  lookup, with one built-in registration per geometric strategy.
- **`resample_every` plus resume was silently wrong**, and the queue refused the
  combination outright rather than fixing it. The collocation cloud and the
  resample hook's own `state_dict` are now checkpointed, so a run resumes onto
  the points it was training on. The guard and its `allow_resampling` waiver are
  gone.
- **`scripts/*.py` did not run from a checkout.** Python puts the script's
  directory on `sys.path`, not the working directory, so every documented
  command failed with `ModuleNotFoundError` unless the package was installed —
  for exactly the user following the README. `scripts/_bootstrap.py` fixes it.
- **`scripts/run.py` crashed on an already-finished run** with `FileExistsError`
  instead of reporting it done. Found by its own test.
- **A flaky test**: `test_initialisation_is_ours_not_torchs_default` asserts a
  sample standard deviation and inherited whatever RNG state preceding tests
  left. It failed on one ordering and passed on another; it is seeded now,
  rather than having its tolerance loosened.

### Added

- `pinnslab/geometry/samplers.py` — the sampler seam: `Sampler`,
  `build_sampler`, `register_sampler`, and `GeometrySampler` for the five
  geometric draws. A sampler is `(state, current) -> points`, the same shape as
  a residual term, and may carry checkpointed state.
- `TrainState.points` — collocation points promoted out of `scratch`, which is
  documented as not surviving a resume. `CheckpointPayload` gains `points` and
  `sampler_state`; format version 1 -> 2.
- `pinnslab/utils/plugins.py` — `--register`, so a paper's own components can be
  imported by module name or file path from the command line.
- `search.spec.load_search_spec` — a `search.yaml` loader, the mirror of
  `load_config`. The spec docstring promised this file existed; nothing loaded
  it.
- `scripts/run.py`, `scripts/run_sweep.py`, `scripts/run_search.py`,
  `scripts/benchmark_population.py`.
- `examples/` — a complete uniform-vs-RAD comparison: two configs differing in
  one field, a 2x5 run matrix, a search spec, and `rad_sampler.py` as a paper
  repo would write it. Measured runtimes and results in `examples/README.md`.
- CI (`.github/workflows/tests.yml`): ruff plus both test commands on push and
  PR, on the declared minimum Python (3.11).
- `time_to_target_resolution_steps` on every row that records a target.

### Changed

- `README.md` rewritten: it claimed only step 1 of the bootstrap existed.
- `run_cell` / `run_queue` lost the `allow_resampling` parameter.
- `pinnslab.training.build.POINTS` is gone; use `state.points`.
- **Time-to-target is measured every step** when the metric is one the step
  already paid for (`loss`, `residual/<name>`), instead of on the trace
  schedule. It was a reviewer-facing compute-parity number that depended on
  `logging.trace` — a field deliberately excluded from the config hash — so two
  runs of one condition could report different times for identical training. An
  `eval_fn`-derived target (`rel_l2`) stays on the schedule and now records the
  resolution it was observed at (DESIGN.md §11).
- **`best.pt` no longer carries optimizer state.** Nothing resumes from it —
  resume is `last.pt` by definition — and Adam's two moments per parameter were
  most of the file, rewritten on every improvement. `last.pt` is unchanged and
  stays resumable.

### Tests

401 -> 451. New: `test_resampling.py` (the sampler seam and bit-exact resume for
a plain and an adaptive sampler), `test_examples.py`, `test_scripts.py`,
`test_plugins.py`, wall-clock band cases in `test_viz.py`, and search-spec
loading in `test_search.py`. The SIGKILL sweep test now kills the *resampling*
cell, between resamples. Both new regression tests were confirmed to fail
against the pre-fix behaviour.

### Not verified

- **No GPU has run any of this.** DESIGN.md §6's "20-50x on a T4" remains
  unmeasured; `scripts/benchmark_population.py` reproduces it on demand (CPU
  here: 1.4x at P=2, 2.1x at P=4, 2.5x at P=8).
- **Kaggle itself**: session limits, dataset publishing, the two-GPU split. The
  local twin of the notebook (`scripts/run_sweep.py`) is tested, including a
  real `os._exit` kill.
- **CI has not run**, having never been pushed. Its workflow file is parsed and
  its commands are asserted by a test, because a malformed workflow does not
  fail loudly — GitHub declines to run it and the repo looks green.

## v0.1.0 — 2026-08-08

Bootstrap complete (DESIGN.md §9 steps 1-5): `utils/`, `registry/`, `training/`,
`geometry/`, `models/`, `physics/`, `losses/`, `eval/`, `benchmarks/burgers`,
`viz/`, `search/`, the Burgers golden test and the Kaggle runner. 401 tests.
Never tagged at the time; the tag exists from v0.2.0 onward.
