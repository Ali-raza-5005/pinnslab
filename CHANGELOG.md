# CHANGELOG

Notable changes per tag. A paper pins a tag (DESIGN.md §2), so what matters here
is what would change a *result*: anything that moves numbers, invalidates a
config hash, or changes what a checkpoint can be resumed from.

## v0.5.2 — 2026-08-31

Stages can be budgeted in **work** rather than in steps.

**This invalidates every config hash.** No numbers move and no behaviour changes
for a config that does not set the new field, but `identity()` hashes
`model_dump(mode="json")`, which includes fields at their defaults — so adding
one optional field re-hashed every config in existence. Consequences:

- Stored results are unaffected: a row carries the hash it was written with and
  nothing recomputes it, so existing runs stay internally consistent.
- **Re-running a pre-v0.5.2 config under v0.5.2 yields a different
  `config_hash`**, and its rows will not group with the originals. Pin per study.
- `sampling_identity_hash` is **not** affected — it covers
  `problem + sampling + dtype`, and `stages` is not in it. Pairing across
  optimizer arms survives, which is the property that matters.
- Checkpoint format is now **3**; version 2 checkpoints are refused rather than
  loaded with a zeroed work origin, for the same reason version 1 was refused.

### Added

- **`StageSpec.max_work`** — stop a stage once it has consumed this much work,
  where work is whatever the caller's new `Trainer(work_fn=...)` counts. The
  library stays ignorant of the unit; a paper counting residual evaluations,
  function evaluations or matrix-vector products passes a reader for its own
  counter.

  **Why.** A step count is not a budget for any optimizer whose per-step cost
  depends on the data. Measured in paper-01: five runs of an identical 1250-step
  L-BFGS stage, differing only in seed, consumed **6,855 to 11,574** residual
  evaluations — a **1.7x spread**. Two arms given "the same 1250 steps" are not
  at equal compute, and choosing per-seed step counts to hit a target means
  reading the budget off the outcome.

  Measured after, on a synthetic L-BFGS stage across five seeds:

  | budget | spread | worst deviation |
  | --- | --- | --- |
  | 200 | 1.059x | 7.000% |
  | 1,000 | 1.008x | 1.000% |
  | 5,000 | 1.002x | 0.220% |
  | 15,001 | **1.000x** | **0.007%** |

  The budget is checked **after** each step, never before: a step's cost is not
  knowable in advance, so the choice is between overshooting by at most one step
  and stopping short by an unknown amount. The overshoot is a fixed absolute
  quantity, so its relative size falls as the budget grows — which is why the
  table above converges.

  `steps` remains required and becomes a **safety bound**: it caps the run if
  the work counter stalls. Budgets are **per stage**, mirroring `steps`, so a
  schedule's total is the sum of its stages.

- **`Trainer(work_fn=...)`** — a reader for the caller's work counter.
- **Reported, not inferred**: run timings gain `stage.<name>.work` (what the
  stage spent) and, when a budget is set, `stage.<name>.hit_work_budget`
  (whether the budget is what stopped it). A stage that ran out of steps first
  had a *non-binding* budget, and that is a different experiment from one that
  spent it — invisible in a results table, so it is recorded.
- **`CheckpointPayload.work_at_stage_start`** — the budget survives a resume. A
  stage resumed with a fresh origin would spend its allowance twice, and the
  runs most likely to be interrupted are the slow ones, so the arms that would
  silently overspend are exactly the expensive ones.

### Fixed, before release

- **`record_last` fired on the step bound only**, so a stage that ended on its
  work budget never forced a final trace point. `is_last` tested
  `step_in_stage == stage.steps`, and a work-bounded stage never reaches that —
  `steps` is deliberately set unreachably high. The run then reported metrics
  from whichever scheduled trace point happened to fire last: on the first
  budgeted L-BFGS run on a real problem the row claimed 354 residual
  evaluations while the accounting sidecar, reading the counter directly, said
  401. **A completed run with a stale final metric and nothing anywhere saying
  so**, in the parity currency itself.

  Found by running the feature end to end on a real problem rather than only in
  unit tests, all eleven of which passed throughout. Pinned by
  `test_the_final_trace_point_is_recorded_when_the_budget_ends_the_stage`,
  confirmed to fail without the fix.

### Refused

- A stage that sets `max_work` while the `Trainer` has no `work_fn` now raises.
  Falling back to the step bound would produce a run that looks budgeted and is
  not, with the config, the log and the results table all agreeing with each
  other — the most expensive shape this feature could fail in.

### Known and unaddressed

Every additive schema change invalidates every config hash, because `identity()`
dumps fields at their defaults. Excluding unset optional fields would fix this
permanently at the cost of one further break. Not done here; it is a change to
identity semantics and deserves its own decision.

## v0.5.1 — 2026-08-30

A GPU-only crash, found by the first GPU sweep this library has ever had.

**This changes no numbers.** On CPU nothing here can execute differently: the
bug it fixes is unreachable without CUDA, and the collocation cloud drawn for a
given seed is unchanged. It is a bug-fix tag, and any v0.5.0 result remains
bit-reproducible under it.

### Fixed

- **DeepXDE's import-time `torch.set_default_device("cuda")` is now undone**,
  in the same place and for the same reason as its float64 side effect
  (`geometry/adapters.py`). That call installs a process-global
  `__torch_function__` mode, after which every tensor factory invoked *without*
  an explicit device allocates on the GPU — including one handed a CPU
  generator, which raises rather than merely being slow:

      RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

  It fired at the first collocation draw of every run in paper 1's first Kaggle
  sweep, from `_numpy_stream`'s seed draw. This library assumes the opposite
  everywhere — `Domain.sample` takes a `device`, `_to_tensor` honours it, and
  `Trainer` builds its sampling generator on CPU deliberately so a cloud is a
  function of the seed and not of the hardware — so the mode silently
  contradicted the design for every device-less call, not only this one.
- `_numpy_stream` draws its numpy seed with `device=generator.device`. Belt and
  braces: this is the call that proved the assumption was load-bearing, and a
  generator's own device is the only correct answer for a draw taken from it.

### Why the suite did not catch it

`torch.cuda.is_available()` is False on CPU, so DeepXDE never installs the mode
and the entire class of bug is invisible there. `tests/unit/test_default_device.py`
now installs a non-CPU default device with `torch.device("meta")`, which hijacks
factories the same way, so three of its four tests fail on CPU without this fix.
The fourth guards the real invariant on a GPU box.

## v0.5.0 — 2026-08-29

The collocation cloud stops depending on how a run is optimised. Driven by
paper 1 again, and by a measurement: an optimizer-schedule ablation was not an
ablation.

**This changes numbers.** Every run's initial cloud differs from v0.4.0, so no
v0.4.0 result is bit-reproducible under this tag. That is the whole point of
pinning a tag (DESIGN.md §2); nothing published is affected, because nothing is
published. Config hashes are unchanged — `identity_hash()` is untouched, so a
v0.4.0 run directory is still found and still resumed by the same id. A
**checkpoint written by v0.4.0 must not be resumed under v0.5.0**: it carries
the old cloud in its payload, so it would continue correctly, but a run half
of which used one draw and half another is not a run anyone should report.
Start those again.

### Changed

- **`Trainer`'s sampling stream is keyed on `cfg.sampling_identity_hash()`**,
  not on `run.config_hash`. The full config identity covers `stages`, so
  appending a stage redrew the cloud that every stage *before* it had trained
  on, and

      stages: [adam 15000]
      stages: [adam 15000, lbfgs 500]

  did not share their Adam phase. They were two different experiments agreeing
  on every field a reader would check — the failure was invisible, and it
  landed squarely on "Adam → L-BFGS", which DESIGN.md §12 calls the canonical
  hybrid.

  Measured on Burgers at nu=0.01/pi, seed 100, identical architecture, point
  counts, optimizer, learning rate and step count, differing only by an
  appended L-BFGS stage — rel-L2 at the end of the *identical* Adam stages:

  | condition | rel-L2 after 15000 Adam steps |
  | --- | --- |
  | `stages: [adam]` | 0.1405 |
  | `stages: [adam, lbfgs]` | 0.5644 |

  A 4x spread from the draw alone. The old behaviour was never *biased* — both
  conditions drew from the same distribution of clouds, so the comparison was
  valid and merely noisy — but it was noise that pairing removes for free,
  where removing it by adding seeds instead costs one to two orders of
  magnitude more compute.

### Added

- **`RunConfig.sampling_identity_hash()`** and **`SAMPLING_IDENTITY =
  ("problem", "sampling", "dtype")`** — the fields that decide what a
  legitimate cloud is: the geometry and constants, the groups and counts and
  strategies, and the precision the points are drawn in. Everything else
  describes what is *done* with the cloud, not what it may be.
- **`tests/unit/test_sampling_identity.py`** — pins both halves, because
  asserting only the first would be satisfied by a trainer that gave every
  config one cloud: changing the schedule, the architecture, the loss weights
  or the name keeps the draw; changing the seed, the point counts, the strategy
  or a physical constant still moves it.

## v0.4.0 — 2026-08-29

The optimizer seam becomes a capability protocol. Driven by paper 1 (CSO over
network weights), which could not be expressed at all: `Trainer` chose how to
drive an optimizer by its concrete type, so a registered derivative-free method
was handed to the first-order path and never saw the objective.

**No config hash changes and no existing behaviour moves** for a gradient-based
run — Adam and L-BFGS take exactly the path they took in v0.3.0. One number does
change: the batched evaluator's collocation cloud (below). Nothing on disk is
invalidated; no results exist yet.

### Added

- **`requires_closure` / `uses_gradients`** — two optional attributes an
  optimizer may declare, read by `training/optimizers.py`'s predicates of the
  same name. `requires_closure=True` makes the loop call `step(closure)`;
  `uses_gradients=False` makes it skip `backward()` entirely. A derivative-free
  optimizer is now a genuine `@register_optimizer` with zero edits to core, and
  `stages: [{cso}, {adam}, {lbfgs}]` is one config, one hash, one `Run`, with
  staging, checkpointing, tracing, timing, divergence handling and
  time-to-target unchanged. DESIGN.md §4.
- **`Trainer._reject_undrivable`** — the refusals, added before the capability
  they guard (DESIGN.md §6 CORRECTION 2's rule). Refused: a closure-based
  optimizer sharing a stage; `direction: max` on one (no gradient exists to
  flip); `max_grad_norm` on a derivative-free one (no gradient exists to clip);
  `uses_gradients=False` without `requires_closure=True` (the optimizer could
  never see the loss); and a closure-based `step()` returning `None` (which
  would put `nan` in the trace and the divergence check).
- **`tests/unit/test_optimizer_seam.py`** — 11 tests over a toy population
  optimizer with the shape of CSO: end-to-end training, composition with a
  gradient stage, bit-exact resume of a population through the ordinary
  checkpoint, all five refusals, and the capability predicates themselves.
- **DESIGN.md §4 conformance item 8** — a derivative-free optimizer over network
  weights. Item 6 ("Adam→L-BFGS") passed while this seam was type-gated, because
  every example on the list was gradient-based.

### Fixed

- **The two search evaluators drew different collocation clouds for one
  config.** `SequentialEvaluator` seeds from `derive_seed(seed, "trainer",
  config_hash)`; `_stack_points` used `torch.Generator().manual_seed(cfg.seed)`.
  Harmless for ranking — which is why the v0.3.0 audit filed it as a limitation
  rather than a bug — but it made "rerun the winner sequentially and you get the
  search's number" **false**, so no batched fitness could be quoted in a paper
  and defended. Both now derive identically. **This changes every batched
  fitness value.**
- **The oracle in `test_batched_and_sequential_agree_on_the_training_objective`
  mirrored the evaluator, not the trainer.** Its state used
  `manual_seed(cfg.seed)`, so it agreed with the batched path on a cloud no
  reproduction run would ever draw. Now derived like `Trainer`. This is the
  *fourth* instance in this repo of a test passing on a premise that cannot
  occur; the previous three are named in v0.2.0 and v0.3.0.

### Documented, not changed

- **A trace point can mix two parameter vectors.** `_step_first_order` reports
  the loss at the parameters it is about to update, so step *k*'s `loss` is
  θ(k−1) while its metrics and checkpoint are θ(k). Fixing it costs a second
  forward pass on every step of every run, so it is kept — but it is now pinned
  by `test_the_first_order_path_still_reports_the_pre_update_loss` rather than
  recorded only in an audit file. The closure path does **not** inherit it: a
  closure-based optimizer must make its last closure call at the parameters it
  leaves installed, so `loss` and `residual/<name>` both describe θ(k).

### Not done, on purpose

- **No batched fitness path for population optimizers.** `Ensemble` would make P
  candidates cheaper, but the measured CPU speedup (1.7-3.4x) does not change a
  conclusion, and the primary compute-parity currency for this comparison should
  be residual evaluations — implementation-independent — with wall-clock
  secondary and its unoptimised status stated. Adding it would also put CSO on
  the narrow path and require a new set of refusals. DESIGN.md §6.
- **No new benchmarks, architectures or weightings.** Allen-Cahn, Helmholtz,
  KdV, NS and wave remain unimplemented; per the promotion rule they are born in
  a paper repo's `src/method/` and enter here only when a second paper needs
  them.

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
- **The offline Kaggle wheel now carries its commit.** DESIGN.md §7's fallback
  (`build wheel → Kaggle Dataset → pip install --no-index`) has no working tree
  and no PEP 610 metadata, so provenance came out `git_sha="unknown"` — rule 7
  failing silently on the platform where the session is gone by the time anyone
  asks. `hatch_build.py` stamps `pinnslab/_build_info.py` at build time and
  `registry.provenance` reads it back as a third route, `git_source =
  "build_stamp"`, consulted **last** so a working tree always wins. The stamp is
  gitignored and removed after the build. Verified end to end: a wheel installed
  with `--no-index` into a clean venv reports the right SHA and its dirty flag.
- Verified the pip-install workflow itself, from outside the checkout: install
  from `git+...@main`, a config and a `@register_sampler` living in a mock paper
  repo, a real Burgers run — provenance resolved via `direct_url`, version
  0.3.0, correct SHA. `README.md` gains a "Starting a paper repo" section for
  that flow.

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
