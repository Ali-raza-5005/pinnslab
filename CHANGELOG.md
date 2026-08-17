# CHANGELOG

Notable changes per tag. A paper pins a tag (DESIGN.md §2), so what matters here
is what would change a *result*: anything that moves numbers, invalidates a
config hash, or changes what a checkpoint can be resumed from.

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
