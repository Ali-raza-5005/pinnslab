# pinnslab

Personal PINN **methods** research infrastructure: we own the PyTorch training
loop, because the research novelty lives inside it. DeepXDE is a thin dependency
(geometry + baseline oracle), never a foundation.

The research program is metaheuristic/population search over PINN configuration
space — sampling, loss weighting, optimizers, architecture — so the library is
built around three things: a run that is reproducible and resumable to the bit,
a results directory that answers "what did we run?", and a search layer that can
propose configurations and afford to evaluate them.

Not a general-purpose library, not intended for external adoption, no API
stability promises. See `DESIGN.md` for why it is shaped this way and
`CLAUDE.md` for the standing rules.

## Install

```bash
git clone https://github.com/Ali-raza-5005/pinnslab
cd pinnslab
pip install -e ".[dev]"      # runtime deps + pytest + ruff + matplotlib/pandas
pytest -m "unit and not slow"
```

Python >= 3.11. A paper pins an exact tag rather than a branch, so that a result
can always be reproduced by the code that produced it:

```bash
pip install "pinnslab @ git+https://github.com/Ali-raza-5005/pinnslab@v0.2.0"
```

Tags are cut when a paper needs to pin the current state; `git tag -l` is the
authoritative list of what exists.

## Five minutes

`examples/` is a complete experiment — uniform vs residual-adaptive sampling on
1-D Burgers, 2 arms x 5 seeds — that runs on a laptop CPU in about seven
minutes. `examples/README.md` walks through it, including what its numbers do
and do not show.

```bash
# one run
python scripts/run.py examples/configs/burgers_uniform.yaml --results results/

# the whole sweep (--register imports the paper's own sampler)
python scripts/run_sweep.py examples/run_matrix.csv \
    --results results/ --register examples/rad_sampler.py

# every figure and LaTeX table, from the results, with no manual steps
python scripts/make_figures.py results/ --out analysis/ --by method

# a metaheuristic search over the method's hyperparameters
python scripts/run_search.py examples/search.yaml \
    --base examples/configs/burgers_rad.yaml \
    --register examples/rad_sampler.py --root analysis/search
```

## How the pieces fit

| I want to...                    | do this                                                        |
| ------------------------------- | -------------------------------------------------------------- |
| define an experiment            | write a YAML config — **no number ever lives in a script**       |
| run one                         | `scripts/run.py` (or `build_trainer(cfg, ctx, run).fit()`)       |
| run many, on a dying machine    | `scripts/run_sweep.py` over a `run_matrix.csv`                   |
| resume an interrupted run       | re-issue the same command; there is no resume flag               |
| search over configurations      | `scripts/run_search.py` with a `search.yaml`                     |
| turn results into a paper       | `scripts/make_figures.py`                                        |
| add a sampler / weighting / ... | one file with `@register_*`, then `--register` it                |

**Where results go.** Every run writes one directory under `--results`, named
`<config_hash>_s<seed>`, holding `config.yaml`, `provenance.json`,
`result.json`, `trace.jsonl` and `checkpoints/`. That directory *is* the record:
`results/` is append-only and nothing ever rewrites it. Derived artefacts
(figures, tables, aggregates) go to `analysis/`, which is disposable. Both are
gitignored — small precious rows belong in a `paper-NN-results` repo, large
checkpoints in a Kaggle Dataset (DESIGN.md §11).

**Resuming.** A run's identity is the pair `(config_hash, seed)`, so the same
command always names the same directory: finished work is skipped, interrupted
work continues from `last.pt`. That includes the collocation cloud and the
sampler's own state, so a run that resamples resumes on the points it was
training on. The claim is not a hope — `tests/unit/test_queue_survives_a_killed_session.py`
kills a sweep with a real `os._exit` mid-run and asserts the finished parameters
are bit-identical to an uninterrupted sweep.

**Adding a method.** Extension is by registration, never inheritance: a new
sampler, weighting, optimizer, model or benchmark is one new file with one
`@register_*` decorator and zero edits to existing files.
`examples/rad_sampler.py` is the worked example. Method code is born in a paper
repo and enters `pinnslab` only when a *second* paper needs it.

## Tests

```bash
pytest -m "unit and not slow"   # before every commit — budgeted at <60s
pytest                          # before every push/tag — adds slow + golden
ruff check .
```

`tests/golden/` trains a frozen Burgers config end to end and asserts a rel-L2
against an exact (Cole–Hopf) solution, so a break anywhere on the
config -> network -> residual -> metric path shows up as a number rather than as
a silently different experiment. CI (`.github/workflows/tests.yml`) runs both
commands plus the linter on every push and pull request.

## Contributing

This is a solo research repo: work lands on `main`, no PRs, no branch
protection. If you are the author (or a future maintainer), the standing rules
in `CLAUDE.md` are binding — in particular:

- run `pytest -m "unit and not slow"` before every commit, `pytest` before every
  push or tag;
- log a line in `FRICTION.md` every time a paper-level task forces an edit to
  the core, because that file is the evidence base for which abstractions are
  wrong;
- keep a dated entry in `LOG.md`;
- never edit the library on Kaggle — push, tag, `pip install ...@tag`.

## Status

The bootstrap (DESIGN.md §9, steps 1–5) is complete: `utils/`, `registry/`,
`training/`, `geometry/`, `models/`, `physics/`, `losses/`, `eval/`,
`benchmarks/`, `viz/`, `search/`, the Kaggle runner, the scripts and the
examples. 449 tests, and `CHANGELOG.md` records what each tag changed.

Next is P0 on paper 1 (sampling). Infrastructure work from here is
paper-driven.

**Known limits, stated rather than discovered later.** No GPU has run this code:
the batched population evaluator is measured at 1.7–3.4x on CPU and DESIGN.md
§6's "20–50x on a T4" is **unverified** — `scripts/benchmark_population.py`
reproduces the measurement on whatever hardware you have. Only one benchmark
(1-D Burgers) and one architecture (MLP) ship; `random` and `de` are the only
search algorithms.
