# LOG.md

Dated ~5-line weekly entry: what ran, what was learned, what's next, current
phase, days left in timebox. Newest first.

---

## 2026-08-28 — research-readiness audit before P0

- **Ran**: a full audit of every module against the question "would this produce
  a wrong number quietly?" Suite 451 → 497 green, ruff clean, the 10-cell
  example sweep and the figure loop end to end, and `scripts/validate_gpu.py`
  smoke-tested on CPU. Tagged v0.3.0.
- **Learned (the expensive one)**: **the batched search evaluator was scoring a
  different objective than the config declared.** It pooled every residual term
  into one mean; `MeanWeighting` means each term separately then sums. On the
  shipped Burgers example that is a 6.3x difference with the boundary term 26x
  under-weighted, and `weighting.coefficients` were dropped outright. Every
  batched search number would have been unreproducible by the single-run path
  the paper reports. Fixed by folding the weighting into the residual rows.
- **Learned (the structural one, and it generalises)**: four separate bugs in
  that one seam were the *same* bug — a field read off `configs[0]` and applied
  to all P. lr, optimizer, physical constants, multi-stage schedules,
  `resample_every`: each ran without complaint and scored every candidate at
  candidate 0's setting, while the archive recorded distinct config hashes. The
  rule now enforced: **a narrow path must refuse what it cannot express, never
  approximate it.** DESIGN.md §6 has the correction.
- **Learned (about the tests)**: the test that should have caught #1 called
  itself "THE test of this module" and built its oracle by re-implementing the
  pooled mean, so both sides computed the same wrong number. Third instance of
  this in the repo, after the `wall_time` fixture and the unread `SAMPLERS`
  registry. **An oracle that reimplements the thing under test is not an
  oracle.** Every regression test added this week was checked to fail against
  the old behaviour first.
- **Also fixed**: a diverged candidate could score *better* than every real one
  under a maximised fitness (the penalty was `10 * max(finite)`, and oriented
  scores are negative), so DE would have chased divergence while the curve
  looked like convergence; `FidelitySchedule.cost` under-reported the search's
  own compute by 21%; time-to-target borrowed `best_mode` for its direction.
- **Added**: `tests/unit/test_algorithms_on_benchmarks.py` — DE and random
  search on Sphere/Rosenbrock/Rastrigin, because nothing had ever checked that
  the optimiser optimises. DE is healthy: 1.2e-11 on Sphere at 300 generations,
  1.0e-4 on Rosenbrock at 400, beating random search 4x at 20 generations rising
  to 3e11x at 300. Also `scripts/validate_gpu.py`, one command to close every
  claim that needs hardware this machine does not have.
- **Next**: **P0 on paper 1 (sampling)**. Still open and unchanged: the GPU
  numbers themselves — §6's "20-50x on a T4" and §5's FP64/FP32 ratio are
  waiting on a Kaggle session, and `validate_gpu.py` is what closes them.
- **Phase**: infrastructure audited and hardened. Paper 1 timebox not yet
  started.

---

## 2026-08-17 — the three things between the bootstrap and a sweep

- **Ran**: full suite 445 tests green (401 before), unit gate ~40-57s against
  the 60s budget, ruff clean. Also ran the whole loop for real, outside the
  tests: a 10-cell sweep (2 arms x 5 seeds, 7m25s), figures and tables from it,
  and a 2-generation DE search (4m08s).
- **Learned (the one that would have cost a paper)**: the per-second figure
  could not be drawn from real data *at all*. `band()` intersected exact float
  timestamps across seeds; seeds never share one, so the band was the single
  point t=0, the log axis dropped it, and matplotlib raised inside `savefig`.
  Its test passed because the fixture gave every seed `wall_time = step/10`. A
  fixture that cannot occur is worse than no fixture: it converts an untested
  path into one that looks tested. Now interpolated onto a common grid, in log
  space, clipped to the interval every seed measured.
- **Learned (the structural one)**: `SAMPLERS` had been declared since day one
  and *nothing read it* — `build.py` passed `strategy:` straight to DeepXDE. So
  the one axis this whole program is about could not be extended without editing
  core, while the config docstring claimed it could. A registry with no lookup
  is not an extension point, and the only way to know is to try to extend it
  from outside. Fixed in `geometry/samplers.py`; `examples/rad_sampler.py` is
  the proof, written the way a paper repo would write it.
- **Decided**: the collocation cloud is **stored in the checkpoint, not
  replayed** from the RNG. Replay works only while sampling is a pure function
  of the stream; an adaptive cloud is a function of the network as it stood at
  the last resample, and that network is gone after a resume. Cost is one
  `(N, d)` tensor per group per checkpoint — small beside Adam's two moments per
  parameter. That closes the `resample_every` gap, so `run_queue`'s refusal and
  its `allow_resampling` waiver are deleted, and the SIGKILL sweep test now
  kills the resampling cell between resamples. Both regression tests were
  checked to fail against the old behaviour before being kept.
- **Also**: `python scripts/x.py` never worked from a checkout (Python puts the
  *script's* directory on `sys.path`, not the cwd) — every documented command
  failed for exactly the person following the README. CI added. Tagged v0.2.0.
- **Next**: **P0 on paper 1 (sampling)**, on the real viscosity and a real
  budget. Also closed checkpoint retention: `best.pt` drops the optimizer
  state, which nothing resumes from and which is most of the file. Still open:
  the GPU speedup measurement (`scripts/benchmark_population.py` is ready and
  waiting for hardware).
- **Phase**: bootstrap complete, infrastructure hardened. Paper 1 timebox not
  yet started.

---

## 2026-08-08 — bootstrap step 5: the search layer (bootstrap complete)

- **Ran**: DESIGN.md §9 step 5 —
  `search/{space,spec,algorithms,population,evaluate,cache,state,loop}.py`.
  333 → 401 tests, full suite ~89s. The bootstrap is finished; from here the
  work is paper-driven.
- **Learned (the expensive one, and it rewrites §6)**: **`vmap` cannot host a
  PINN residual.** §6 prescribed `torch.func.vmap` over `stack_module_state` +
  `functional_call`. A residual differentiates the net w.r.t. its *inputs*, so
  it must call `requires_grad_()` on the collocation points, and vmap refuses
  that outright. Making it work would mean respelling every PDE against
  `jacrev`/`hessian` — the per-paper monkey-patching §1 exists to avoid. Fixed
  by keeping the goal and dropping the mechanism: put the population on a
  **leading batch dimension** and build one graph. `diffops` now indexes with
  `...` instead of a leading colon, so one residual serves both a single run
  and a population and never learns the population exists. Verified before
  building on it, not after: batched vs separate evaluation of a Burgers
  residual agrees to **0.0e+00**, and one Adam over stacked `(P, ...)` params
  drifts **1.1e-16** from P separate Adams over 25 steps (Adam is elementwise,
  so this is exact in principle and the number is just float noise).
- **Also measured, and it deflates a design claim**: batching gives **1.7× at
  P=4, 2.8× at P=8, 3.4× at P=16, ~2.2× at P=50** on this CPU with a real
  Burgers residual — not §6's "20–50×", which is a GPU claim about kernel
  launch overhead and is **untested here**. Recorded as untested in §6 rather
  than quietly inherited; an unverified speedup is a hole in the compute-parity
  defence.
- **Learned (a real bug, caught by the batched-vs-sequential equivalence
  test)**: `BatchedEvaluator` called `configure_runtime` once for the whole
  population, but that reseeds the global RNG and `assemble` draws the initial
  weights from it — so candidate *k*'s initialisation depended on **how many
  candidates preceded it in the batch**. A config's fitness was a function of
  its position, which silently breaks the candidate cache, whose whole premise
  is that `(config_hash, steps)` names one experiment. Now reseeded per
  candidate, pinned by a test that scores a config first, last and alone.
- **Decided**: only `random` and `de` ship. Random search is not a placeholder —
  §8 makes it a mandatory matched-budget baseline, and DE's generation 0 *is* a
  random sample at the same seed, so the two are paired. PSO/GA/GWO are one
  file each under rule 2 when a paper needs them. Also: the cache is keyed on
  `(config_hash, steps)`, because a fitness at 200 steps is not the same number
  as one at 20000; and `SearchState.best()` compares only within a fidelity
  rung, or the search crowns candidates that got lucky cheaply and were never
  tested properly.
- **Next**: **P0 on paper 1 (sampling).** Still open before the first real
  sweep: checkpoint retention, and actually fixing the `resample_every` gap
  rather than guarding it.
- **Phase**: bootstrap complete. Paper 1 timebox not yet started.

---

## 2026-08-08 — bootstrap step 4: figures and tables

- **Ran**: DESIGN.md §9 step 4 — `viz/style.py` (house rcParams, palette,
  colormaps, save), `viz/aggregate.py` (`results/` → median/IQR/failure rate
  with the §5 hardware check), `viz/convergence.py`, `viz/tables.py` (booktabs),
  and `scripts/make_figures.py`, the worked config→figure loop with no manual
  steps. 278 → 333 tests; unit ~17s, full suite ~89s. Also researched what the
  PINN literature actually does and what the plotting ecosystem offers before
  writing any of it.
- **Decided (measured, not chosen)**: **SciencePlots is not adopted.** Two
  independent reasons. Its `science` style sets `text.usetex: True`, so every
  figure depends on a working TeX install — a figure that renders here and dies
  in CI is not reproducible. And its default color cycle *fails* a
  colorblind-safety check: orange `#FF9500` against green `#00B945` is ΔE 2.8
  under protanopia (OKLab ×100, Machado et al. 2009), i.e. indistinguishable —
  on the two colors a "ours vs baseline" plot reaches for first. Paul Tol's
  `bright` fails on lightness band and chroma floor. Settled on Okabe-Ito
  **reordered** so the worst adjacent pair is ΔE 9.6 across six slots on paper
  white, with a linestyle and marker per slot because IEEE wants B&W-readable
  figures and three of the six sit below 3:1 contrast. The palette is pinned by
  a test; it is not a matter of taste.
- **Learned (the expensive one, and it did not come from a test)**: the first
  three defects all survived a green 44-test suite and were caught by
  *rendering the figure and looking at it*. (1) A diverged seed's trace was
  inside the median band, so the legend read `n=5` while the figure's own
  failure note read "1/5 failed" — two numbers from the same five runs
  disagreeing. (2) The log x-axis silently dropped step 0, which is exactly the
  untrained baseline `MetricSchedule.record_first` exists to capture; matplotlib
  discards non-positive x without a word. (3) The table printed
  `9.34 \times 10^{-4}` directly above `0.00111` — one quantity, two notations,
  in the cell a reader compares — and `\caption{... rel_l2 ...}` would not have
  compiled. All three now have tests, written after the render, not before.
  **Add "render it and look at it" to the definition of done for any figure
  code**; the assertions I would have written unprompted all passed.
- **Also decided**: `viz/` is deliberately not re-exported from a package
  `__init__` (matplotlib import cost, same rule as `training/build` and
  `training/queue`); `dev` now pulls `analysis`, because a dev install that
  cannot collect `test_viz.py` is not a dev install; and `registry/run.py`
  grew a plain `read_trace(directory)` so aggregation can read a finished run
  without constructing a `Run` and re-collecting provenance.
- **Next**: step 5 — `search/`: SearchSpec, the vmap population evaluator,
  outer-loop checkpointing, the candidate cache, multi-fidelity. Then P0 on
  paper 1. Still open before the first real sweep: checkpoint retention, and
  actually fixing the `resample_every` gap rather than guarding it.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-08-08 — bootstrap step 3: the Kaggle runner

- **Ran**: DESIGN.md §9 step 3 — `training/queue.py` and
  `notebooks/kaggle_runner.py`, plus the suite-budget decision that was blocking
  it. 244 → 278 tests. Split into two commands: `pytest -m "unit and not slow"`
  for the commit loop (43s, was 56.5s) and `pytest` before a push or tag (205s:
  +105s `slow`, +79s golden). Only the three *subprocess* tests got the `slow`
  marker; `test_resume_is_bit_exact` was on the list but is in-process and is
  the load-bearing test of the checkpoint layer, so it stays in the loop that
  runs before every commit.
- **Decided (a departure from §7, recorded there)**: **status is derived, never
  written.** §7 specified a mutable status column that the notebook marked done.
  Deriving it from the results directory is strictly better: the `run_id` is a
  pure function of `(config_hash, seed)`, so the directory *is* the claim, rule
  6 holds by construction, and — the deciding argument — a status column cannot
  be written correctly by a process that may be killed at any instant. Written
  before the work it strands rows in `claimed` forever; written after, it loses
  every interrupted run. Combined with static worker partitioning (`index % n`),
  claiming needs no lock, no lease and no heartbeat, because two workers never
  consider the same cell.
- **Learned**: a crash during **assembly** left no evidence at all. `run_cell`
  creates the run directory, then builds; `Trainer.fit` logs its own crashes but
  had not been reached yet, so an unregistered problem or an OOM allocating the
  nets left a directory indistinguishable from one a session was killed in — and
  a config that cannot be built would never reach the failure rate (§11) at all.
  Found by a test that asserted `FAILED` and got `RESUMABLE`. Only the build is
  wrapped: `fit` already records its own, and logging both would report one
  failure as two.
- **Also learned**: the killed-session proof pins more than resumability. The
  killed sweep trains its cells as (0) then (1, 2) across two processes while
  the reference trains (0, 1, 2) in one, so asserting the two are bit-identical
  also pins that no global state leaks between cells — a seed set once, a
  default dtype, a shared RNG stream would all show up here. The kill is a real
  `os._exit`, not an exception: an exception unwinds and flushes, which is the
  one thing a dying Kaggle session does not do.
- **Decided**: `run_queue` **refuses** `resample_every` on a checkpointed run
  rather than allowing the known-silent corruption (collocation points are not
  checkpointed, so a resumed run continues on the initial cloud). The queue is
  the machinery that makes runs interruptible, so it is where that would first
  bite — on paper 1, whose subject is sampling. `allow_resampling=True` waives
  it. Also: `queue` is deliberately not re-exported from `training/__init__`,
  for the same reason `build` never was — both reach deepxde, and
  `import pinnslab.training` must not.
- **Next**: step 4 — `viz/style.py` + one figure script reading `results/` into a
  publication-ready convergence plot, config→figure with zero manual steps.
  Still open before the first real sweep: checkpoint retention, and actually
  fixing (not just guarding) the `resample_every` gap.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-08-08 — bootstrap step 2: Burgers end-to-end

- **Ran**: DESIGN.md §9 step 2 in full — config volatile axes (`problem`,
  `nets`, `residuals`, `weighting`, `sampling`) + their registries,
  `geometry/adapters.py`, `models/mlp.py`, `physics/diffops.py`,
  `eval/metrics.py`, `benchmarks/burgers.py`, `training/build.py`, and the
  Burgers golden test. 99 → 244 tests; unit ~53s, golden 87s, both inside
  budget. First code to import DeepXDE, and rule 1 is now enforced by a test
  that scans the package rather than by memory.
- **Learned (the expensive one)**: the assembled run reached rel-L2 0.13–0.27
  where stock DeepXDE reached 0.066 on identical architecture, optimizer and
  budget — and did so with a *lower* loss, which is why nothing looked wrong.
  Ruled out by measurement in this order: residual algebra and coordinate order
  (our diffops output is **bit-identical** to `dde.grad`, 0.0 difference), the
  Cole-Hopf reference (h² convergence of its own Burgers residual; independent
  FD solve agrees to 1.3e-4), the eval path (supervised fit through the same
  grid and metric reaches 0.011), the trainer (identical plateau without it).
  What was left: **DeepXDE evaluates its PDE on `train_x_all`, which includes
  the boundary and initial points** — 2780, not our 2540 interior. Enforcing
  the residual at t=0 ties `u_t` to derivatives the IC already pins, and that
  is what carries the IC forward; interior-only is simply an easier objective
  whose minimiser is not the true solution. Fixed by making
  `ResidualSpec.points` a list, so the choice is declared, hashed and on the
  results row instead of buried in point bookkeeping. Now 0.030–0.044 at 15k
  Adam vs DeepXDE's 0.066. **The oracle of DESIGN.md §1 paid for itself on its
  first use** — no unit test would have found this, because every unit was
  correct.
- **Also learned**: three DeepXDE behaviours the adapter now neutralises, each
  silent — the backend is ambient (bare `import deepxde` dies here on a missing
  `tensorflow_probability`), geometry samples from numpy's *global* RNG, and
  `dde.config.set_default_float` mutates `torch.set_default_dtype`. And a
  concrete instance of DESIGN.md §8's heavy tail: one seed of a correct config
  landed at 0.435 against 0.030–0.044 for its siblings. I briefly mistook that
  outlier for a sampling-strategy effect; a 2×2 over strategy and seed
  falsified it. Report failure rates, not means.
- **Decided**: the golden test runs at `nu = 0.1/pi`, not the standard
  `0.01/pi`. At standard viscosity a CPU-budget run spans 0.064–0.204 across
  seeds, so the only reliably-passing threshold (~0.35) would not have caught
  the bug just fixed (0.127–0.273). Smoothed, the same budget gives ~6e-4 with
  <1.5× spread and a threshold with teeth. The cost is recorded in the test
  docstring rather than hidden: **this golden test is blind to the failure that
  actually happened** — interior-only scores 0.00081 on it, indistinguishable
  from correct. That failure is guarded by the frozen config's hash instead.
- **Next**: step 3 — the Kaggle runner (≤20-line notebook, install-from-tag,
  `run_matrix.csv` claim/run/checkpoint/push/mark-done), and prove it survives a
  killed session. Before the first real sweep: checkpoint retention, and the
  `resample_every` + resume gap now queued in TESTS_TODO.md.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-08-08 — training/trainer review

- **Ran**: module walkthrough of `training/trainer.py` against `checkpoint.py`,
  `registry/config.py`, `registry/run.py`, `registry/schema.py`,
  `training/optimizers.py` and the existing test suite. 99 unit tests green on
  CPU in ~19s (98 → 99), ruff clean. One source bug found and fixed; two
  tradeoffs queued in TESTS_TODO.md.
- **Learned**: `_fit` used `global_step == 0` as a proxy for "this is a fresh
  run," but `_restore` returns that same shape when a checkpoint exists and
  happens to record step 0 — which is exactly what the stage-boundary save
  (before the first training step) writes. `CheckpointSpec.every_seconds`
  defaults to 600.0 with `every_steps` unset, so that step-0 checkpoint is the
  only one on disk for the first 10 minutes of any run. A session killed in
  that window — the entire premise of checkpointing a Kaggle run — resumed
  into `_record_baseline` firing a second time, appending a duplicate `step=0`
  point to the append-only `trace.jsonl`. The existing
  `test_a_resumed_run_does_not_re_record_the_baseline` didn't catch it because
  its simulated crash happens at step 12, after a real periodic save had
  already moved `last.pt` past step 0.
- **Decided**: `_restore` now returns whether it actually loaded a payload,
  and `_fit` gates baseline recording on that instead of on `global_step == 0`.
  Pinned by `test_a_crash_before_the_first_step_does_not_duplicate_the_baseline`
  (first written against the old condition to confirm it reproduces the
  duplicate, then checked green against the fix). Also added one sentence to
  the module docstring's contract section: `residual_fn` must be deterministic
  given `state` within a step (read collocation points from `state.scratch`,
  don't sample inline), because L-BFGS's closure can run more than once per
  `.step()`. Left queued: `time_to_target_*`/best-checkpoint tracking piggybacks
  on the (downsampled) trace schedule rather than checking every step, and
  L-BFGS's repeated closure calls mean `residual/<name>` trace metrics can
  reflect a different iterate than the `loss` reported alongside them —
  both loop restructures, not small patches, so deferred rather than changed
  under review pressure.
- **Next**: step 2 — 1D Burgers + vanilla MLP end-to-end, golden test asserting
  target rel-L2. First code to import DeepXDE, `geometry/adapters.py` only.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-08-08 — registry/run + training/checkpoint review

- **Ran**: module walkthroughs of `registry/run.py` and `training/checkpoint.py`.
  98 unit tests green on CPU in ~15s (87 → 98), ruff clean. Four source fixes in
  each file; storage/retention and search-state checkpointing queued.
- **Learned**: **L-BFGS checkpoints cleanly.** `state_dict` does carry
  `old_dirs`/`old_stps`/`ro`/`H_diag`/`prev_flat_grad`/`d`/`t`; measured
  bit-exact resume through the real save/load path on torch 2.12, with and
  without `strong_wolfe`. The module docstring and DESIGN.md §5 asserted the
  opposite, and the trainer rewound an interrupted L-BFGS stage to its boundary
  — throwing away a whole stage per session death, on the platform whose
  sessions die unpredictably. Killed at step 9 of 12, the old policy resumed
  from step 0; it now resumes from step 8. Both files corrected; the claim is
  pinned by `test_lbfgs_resume_is_bit_exact` so a future torch regression fails
  a test instead of silently producing a different experiment. The 07-31 entry's
  "bit-exact through L-BFGS stages" was true *because* of the rewind and so
  never exposed this.
- **Decided**: `ResultRow` opts out of `extra="forbid"` and gains
  `schema_version` (default 1 — a row without the field *is* v1). Rows are built
  in code, so forbid caught no typos there while guaranteeing that adding a
  field breaks every permanent row on disk. Everything parsed from YAML keeps
  forbid. Frozen fixture at `tests/fixtures/result_row_v1.json`; never
  regenerate it. Also: `_read_jsonl` now skips bad lines instead of `break`ing —
  a killed session's stump fuses with the resumed session's first append, so the
  corruption is mid-file and `break` was silently discarding everything written
  after the resume (a run at step 40 reported as step 10).
- **Next**: step 2 — 1D Burgers + vanilla MLP end-to-end, golden test asserting
  target rel-L2. First code to import DeepXDE, `geometry/adapters.py` only.
  Before the first real sweep: checkpoint retention (last.pt is dead weight once
  `result.json` exists; best.pt needs no optimizer state) — currently ~6
  parameter-sets of storage per run against a capped Kaggle working dir.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-08-03 — registry/schema review

- **Ran**: module walkthrough of `registry/schema.py`; 87 unit tests green on
  CPU in ~17s, ruff clean. Four source bugs found and fixed, five gaps queued in
  TESTS_TODO.md.
- **Learned**: `json.dump` writes bare `NaN` tokens and reads them straight
  back, so a diverged row round-trips fine in Python while being invalid JSON
  (RFC 8259) to everything else — and under rule 6 those rows are permanent.
  Non-finite floats now travel as `"nan"/"inf"/"-inf"` strings (`json_safe` out,
  a `BeforeValidator` back in), with `allow_nan=False` as the assertion behind it.
- **Decided**: a crash does **not** finish a run. `RunStatus.FAILED` was dead
  code; writing it into `result.json` was the obvious fix and the wrong one —
  it finalises the run, so a transient OOM would burn every checkpointed step
  (caught by `test_resume_is_bit_exact`). Crashes now append to `failures.jsonl`
  and surface via `load_runs(include_unfinished=True)`, which also reports
  session-killed runs as `running`. Use that flag whenever the number is a rate.
  Also: `record_first` was unreachable (the trainer increments before checking,
  so step 0 never arrived) and now costs one forward pass to trace the untrained
  baseline; an all-off `MetricSchedule` is now a validation error. Left queued —
  in-loop trace points mix a pre-update loss with post-update `eval_fn` metrics
  and checkpoint, which matters only when `best_metric="loss"`.
- **Next**: step 2 — 1D Burgers + vanilla MLP end-to-end, golden test asserting
  target rel-L2. Still the first code to import DeepXDE, `geometry/adapters.py`
  only.
- **Phase**: bootstrap. No paper timebox running yet.

---

## 2026-07-31 — bootstrap step 1

- **Ran**: scaffolded the package and implemented DESIGN.md §9 step 1 only —
  `utils/seeding`, `utils/device`, `components.py`, `registry/` (hashing,
  config, provenance, schema, Run), `training/checkpoint`, `training/trainer`.
  59 unit tests green on CPU in ~10s; ruff clean.
- **Learned**: bit-exact resume survives a *hard* kill (`os._exit` mid-Adam,
  rerun identical command → identical final params through Adam and L-BFGS
  stages). Two provenance traps found while building: `torch.load(weights_only=
  True)` rejects numpy's ndarray RNG state (so it is stored as plain ints), and
  `git status --untracked-files=no` reported a fully-untracked package as clean.
- **Decided**: config hash excludes seed/device/logging/checkpoint — a run is
  `(config_hash, seed)`. Recorded in DESIGN.md §4. `@register_*` lives in
  `components.py`, not `registry/`, to avoid the name collision.
- **Next**: step 2 — 1D Burgers + vanilla MLP end-to-end, golden test asserting
  target rel-L2. That is the first code to import DeepXDE, and only in
  `geometry/adapters.py`.
- **Phase**: bootstrap. No paper timebox running yet.
