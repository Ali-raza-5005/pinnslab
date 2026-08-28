# TESTS_TODO.md

Test gaps found during the module-by-module review, deliberately deferred to a
single hardening pass once the walkthrough of `pinnslab/` is finished. Items are
removed from this file when the test lands — this is a queue, not a log.

Anything found here that is a *source* bug rather than a test gap gets fixed
immediately instead of being queued.

---

## the unit gate's 60s budget (measured 2026-08-28)

- [ ] **`pytest -m "unit and not slow"` now straddles its budget.** Measured
      five times on this laptop after the audit: **59s, 63s, 66s, 75s, 82s** for
      478 tests. DESIGN.md §3 budgets 60s and says why it is load-bearing —
      "this is the command whose latency decides whether tests get run at all".

      Two honest observations before anyone optimises:
      1. **Machine variance dominates the change.** Identical work measured 59s
         and 82s back to back. The audit's new tests account for ~4s of it
         (the whole `test_algorithms_on_benchmarks.py` file is 3.9s, and ~2s of
         that is import cost it shares with the rest of the suite).
      2. **The slowest tests are pre-existing and each trains a real network**:
         `test_a_cheap_target_is_measured_every_step_not_on_the_trace_schedule`
         (4.7s), `test_a_sweep_partitions_between_workers` (4.5s),
         `test_min_and_max_optimizers_coexist_on_disjoint_slices` (2.2s).

      So the options are to shrink those (fewer steps, smaller nets — they are
      testing loop mechanics, not convergence), to move a few to `slow`, or to
      accept a higher budget and say so in §3. Decide deliberately; do **not**
      let it drift, because the failure mode is silent: a gate that takes two
      minutes stops being run before anyone notices it stopped being run.

---

## search/ (built 2026-08-08, audited 2026-08-28)

One source bug was found and fixed while building rather than queued:
`BatchedEvaluator` reseeded the RNG once per population, so a candidate's
network initialisation — and therefore its fitness — depended on its position
in the batch, which silently breaks the `(config_hash, steps)` cache key.

The 2026-08-28 audit found five more, all fixed with regression tests rather
than queued (CHANGELOG v0.3.0, DESIGN.md §6 CORRECTION 2): the batched path
optimised a pooled objective instead of the declared per-term one; `Ensemble`
ignored the declared activation; a diverged candidate could outscore every real
one under a maximised fitness; `FidelitySchedule.cost` under-reported compute by
21%; and the path silently ignored per-candidate `lr`, optimizer, problem
constants, multi-stage schedules and `resample_every`. What is left:

- [ ] **The batched speedup is unmeasured on a GPU**, which is the only
      hardware where DESIGN.md §6's "20-50x" claim could hold. CPU gives
      1.7-3.4x, peaking at P=16. Measure on a T4 and a P100 before any paper
      quotes a search-cost number, and record P alongside it — the curve is not
      monotone, so "pop_size 50" may be slower per candidate than 16.
      **`scripts/validate_gpu.py` is what runs it**; it needs a Kaggle session,
      not more code.

- [x] **`Ensemble` assumes one activation for every layer and every member.**
      Half done, and the half that mattered. It *silently substituted* `tanh`
      for whatever the members declared, which was a correctness bug rather
      than a missing feature (measured: 7.9e-2 disagreement for a `sin` MLP);
      it now infers the activation and **refuses** a mixed population instead
      of unifying it. Varying activation *across* the population — the one of
      DESIGN.md §6's four directions that should batch cleanly — still needs a
      per-member index and a gather. Until then it raises, so a paper reaching
      for it gets a message rather than a wrong number.

- [ ] **`BatchedEvaluator` scores the training objective, never a held-out
      metric.** That is documented and deliberate (no reference solution on
      that path). Now that the two paths genuinely share an objective (they did
      not before 2026-08-28), the remaining question is the interesting one:
      does agreeing on the objective mean agreeing on which candidate
      *generalises*? Worth a test that the two paths' **rankings** agree on a
      real problem — and if they do not, that is a result about multi-fidelity
      search, not a bug.

- [ ] **The two evaluators draw different collocation clouds for one config.**
      `SequentialEvaluator` goes through `build_trainer`, whose stream is
      `derive_seed(seed, "trainer", config_hash)`; `BatchedEvaluator` uses
      `torch.Generator().manual_seed(cfg.seed)`. So a candidate's batched
      fitness is not reproducible by the run that reproduces it, even though
      both are now minimising the same objective. Harmless for *ranking* (every
      candidate is treated alike, and the clouds are drawn from the same
      distribution) but it means "rerun the winner sequentially and you get the
      search's number" is false. Decide whether the batched path should adopt
      the trainer's derivation before quoting a batched fitness in a paper.

---

## training/queue (built 2026-08-08)

One source bug was found and fixed while building this rather than queued: a
crash during *assembly* — after the run directory exists, before `Trainer.fit`
can record anything — left no evidence, so a config that could not be built was
indistinguishable from a session that was killed and never reached the failure
rate. `run_cell` now logs it. What is left:

- [ ] **`select`/`statuses` load and validate every config on every call.** Fine
      for a hand-written sweep of tens of cells; the `search/` layer (§6) will
      generate matrices of 1e4–1e5. At that size the notebook pays a full
      YAML-parse-and-validate pass per selection, and `run_queue` pays a second
      one for the up-front resampling check. Cache by `(path, mtime)` when it
      actually hurts — but measure first, since the alternative is a cache that
      can serve a stale config, which is worse than slow.

- [ ] **The deadline heuristic is untested against real timings.** `run_queue`
      declines a new cell when the time left is under the longest cell seen
      *this session*, which is only a good estimate when cells are similar. A
      matrix mixing a 30s cell with a 3h cell will either overrun (short cells
      first) or leave an hour idle (long cell first). Only the degenerate
      `deadline_seconds=0.0` path is covered. Decide whether per-config
      estimates are worth it before the first sweep with heterogeneous cells.

---

## utils/seeding (reviewed 2026-07-31)

- [ ] **`restore_rng_state` CUDA error paths are uncovered** (`seeding.py:148`,
      `seeding.py:153`). Both look GPU-only but are reachable on CPU:
      - "checkpoint carries CUDA RNG state but this process has no CUDA device"
        fires directly on this machine from a hand-built state dict with a
        non-`None` `"cuda"` key.
      - the device-count mismatch needs one monkeypatch of
        `torch.cuda.device_count`.
      Both currently read as dead code to the suite, and both exist to stop a
      run silently continuing on the wrong hardware.

- [ ] **Seed upper bound untested.** `set_seed(2**63)` must raise; only the
      negative case is checked.

- [ ] **`test_capture_restore_round_trips_every_stream` is not diagnosable.** It
      draws python/numpy/torch into one tuple, so a failure does not say which
      stream broke. Parametrise per stream.

- [ ] **`test_different_seeds_diverge` is near-zero value.** Either strengthen it
      (assert divergence per stream) or delete it.

- [ ] **numpy seed truncation is undocumented behaviour.** `set_seed` does
      `np.random.seed(seed % 2**32)`, so seeds differing by exactly 2**32 collide
      in numpy but not in torch or python. Harmless in practice, but decide
      whether to reject seeds >= 2**32 outright and test whichever way it goes.

---

## registry/schema (reviewed 2026-08-03)

Four source bugs found in this pass were fixed immediately rather than queued:
non-finite metrics were written as bare `NaN` tokens (invalid JSON, and
unrepairable under rule 6); `RunStatus.FAILED` was never written by anything, so
a crashed run left no evidence at all; `record_first` was unreachable from the
trainer; and an all-off `MetricSchedule` silently produced an empty trace. All
four now have tests. What is left:

- [ ] **`_is_log_spaced_step` has no direct test at all** — the only nontrivial
      algorithm in `registry/`, and the suite only ever exercises
      `MetricSchedule(every=N)`. The `n_per_decade` path is entirely uncovered.
      Assert the property that actually matters: exactly `n_per_decade` points
      per decade above step 10, and statelessness (the predicate over
      `range(a, b)` does not depend on where you started iterating — that is what
      makes a resumed trace identical to an uninterrupted one).

- [ ] **First-decade under-sampling is undocumented.** `n_per_decade=20` yields
      9 points below step 10, not 20 — there are only 9 integers to place them
      on. Pin it in a test so nobody later "fixes" it, and note it wherever trace
      point counts get compared across schedules.

- [ ] **A trace point mixes two parameter vectors** (found while fixing
      `record_first`; left alone because fixing it is a loop restructure, not a
      test). `_step_first_order` evaluates the loss at the parameters it is about
      to update, so the point labelled step `k` holds `loss` at θ(k-1) while its
      `eval_fn` metrics and its checkpoint are at θ(k). The ordinary convention
      everywhere, and cheap — recording the post-update loss costs a second
      forward pass per traced step. But note the consequence: with
      `best_metric="loss"`, `save_best` stores θ(k) scored by a loss θ(k) did not
      produce, and `final_metrics["loss"]` is one update staler than `last.pt`.
      With an `eval_fn` metric as `best_metric` it is consistent. Decide whether
      the off-by-one is acceptable, and pin whichever way in a test. Step 0 is
      currently the only point where every metric shares one parameter vector.

- [ ] **`frozen=True` is shallow, and `hash(row)` raises.** `row.tags["k"]="v"`
      mutates a supposedly-immutable row; separately, `frozen=True` asks pydantic
      to generate `__hash__`, which then raises `TypeError: unhashable type:
      'dict'` on any model with a dict field (`ResultRow`, `TracePoint`). It will
      surface far from here — a `set` of rows, a row as a dict key, an
      `lru_cache`. Decide whether rows are hashable and test it either way.

- [x] **No forward-compatibility story for `results/`.** Done: `ResultRow` now
      carries `schema_version` (default `RESULT_SCHEMA_VERSION = 1`, since a row
      without the field *is* a v1 row) and overrides `extra="ignore"`, so older
      code reads newer rows instead of raising. Every other `Spec` keeps
      `extra="forbid"` — those are parsed from hand-written YAML where an unknown
      key is a typo'd hyperparameter; rows are built in code and carry no such
      risk (pinned by `test_config_models_still_reject_unknown_keys`). A frozen
      pre-`schema_version` row lives at `tests/fixtures/result_row_v1.json`.
      Renames and removals still need a reader that branches on the version.

- [ ] **`TracePoint` has the same problem and no version field.** Deliberately
      not fixed alongside `ResultRow`: trace points outnumber rows by ~1e2-1e3,
      so a per-line version is real bytes across a 1e5-run sweep, and the natural
      alternative (one version for the whole file, in a header line or derived
      from the run's `ResultRow`) is a different design. Decide before the first
      trace-schema change, not after.

- [x] **`_read_jsonl` stopped at the *first* bad line, not only a trailing one.**
      Fixed: it now skips unparseable lines and warns once with a count, instead
      of `break`ing. Reading past corruption is safe here because writes are
      one line at a time, fsynced and append-only, so a torn record can only be
      the file's tail at the moment a session died — it cannot cascade. The old
      behaviour reported a run killed at step 10, resumed, and trained to step 40
      as having reached step 10. Pinned by
      `test_records_written_after_a_torn_line_are_recovered`.

---

## training/trainer (reviewed 2026-08-08, audited 2026-08-28)

One source bug found in the 2026-08-28 audit was fixed rather than queued:
`_track_target` used `eval.best_mode` for the *target's* direction, so a config
keeping the highest value of one metric while targeting a low value of another
recorded time-to-target as reached at its first trace point. `EvalSpec` now has
an explicit `target_mode`. This changed every config hash; nothing was on disk.


One source bug found in this pass was fixed immediately: `_fit` inferred "is
this a fresh run" from `global_step == 0`, but a checkpoint saved at step 0
(the stage-boundary save that lands before the first training step) makes
that condition true on a genuine resume too, so a session that died in the
first `checkpoint.every_seconds` window (600s by default) re-recorded the
step-0 baseline into `trace.jsonl` on resume. `_restore` now returns whether
it actually loaded a payload, and `_fit` gates baseline recording on that
instead. Pinned by
`test_a_crash_before_the_first_step_does_not_duplicate_the_baseline`. What is
left:

- [ ] **`geometry.Domain.sample` has no test that points *cover* the domain.**
      `test_interior_points_lie_within_the_domain` only checks the bounding
      box, which a sampler that returned 2540 copies of one corner would also
      pass. Checked by hand during the Burgers bring-up (deciles uniform to
      ~0.02 in both coordinates); worth pinning, since a degenerate cloud
      trains happily and reports a plausible loss.

- [ ] **L-BFGS's closure may be invoked more than once per external `.step()`**
      (strong-Wolfe line search), and `self._last_residuals` — therefore the
      `residual/<name>` trace metrics computed from it in `_record()` —
      reflects whichever closure call happened *last*, while the `loss`
      reported for that same trace point is `orig_loss`, from the *first*
      closure call. So for L-BFGS stages, the recorded `residual/<name>`
      values and the recorded `loss` do not necessarily describe the same
      parameter iterate. A sharper, L-BFGS-specific case of the
      `registry/schema` section's "a trace point mixes two parameter vectors"
      item above; left alone for the same reason (loop restructure, not a
      test).
