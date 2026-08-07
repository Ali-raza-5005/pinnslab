# LOG.md

Dated ~5-line weekly entry: what ran, what was learned, what's next, current
phase, days left in timebox. Newest first.

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
