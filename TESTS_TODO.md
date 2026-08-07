# TESTS_TODO.md

Test gaps found during the module-by-module review, deliberately deferred to a
single hardening pass once the walkthrough of `pinnslab/` is finished. Items are
removed from this file when the test lands — this is a queue, not a log.

Anything found here that is a *source* bug rather than a test gap gets fixed
immediately instead of being queued.

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

## training/trainer (reviewed 2026-08-08)

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

- [ ] **Best-checkpoint tracking and `time_to_target_steps`/
      `time_to_target_seconds` are only evaluated inside `_record()`, which
      only runs on the (downsampled) trace schedule — not every step.** For a
      cheap `target_metric`/`best_metric` like `"loss"` (already computed
      every step for free), this makes the reported "steps/seconds to target"
      schedule-dependent rather than exact, which cuts against DESIGN.md
      §8/§11's framing of it as a reviewer-facing, compute-parity number.
      Checking it every step is not free in general — an `eval_fn`-derived
      `best_metric` is exactly what the throttling exists to avoid calling
      every step — so this needs a decision, not a reflexive fix. Decide
      whether the schedule-dependent bias is acceptable, and pin whichever way
      in a test.

- [ ] **Collocation points do not survive a resume, and `resample_every` makes
      that visible.** `TrainState.scratch` is deliberately not checkpointed, and
      `build_trainer` draws the first cloud before `fit()`, which is why a
      *fixed-point* run resumes exactly (the draw happens from the generator's
      initial state, which `_restore` then overwrites). With
      `resample_every=K`, a run killed at a step that is not a multiple of K
      resumes holding the **initial** cloud rather than the k-th one, and
      trains on it until the next multiple of K. Invisible in the metrics, and
      it lands directly on paper 1, whose subject is sampling. Fixing it means
      deciding what sampler state is checkpointed — the same decision as
      DESIGN.md §6's outer-loop checkpointing — so it wants a design call, not
      a patch. Until then, do not combine `resample_every` with runs that can
      be interrupted.

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
