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
pip install "pinnslab @ git+https://github.com/Ali-raza-5005/pinnslab@v0.3.0"
```

Tags are cut when a paper needs to pin the current state; `git tag -l` is the
authoritative list of what exists.

## Starting a paper repo

One directory per paper, `pinnslab` installed into it from a tag. Nothing is
copied and nothing is edited here — the library is a dependency, and the paper
owns its configs, its method code and its results (DESIGN.md §2).

```bash
mkdir paper-01-sampling && cd paper-01-sampling
python -m venv .venv && . .venv/Scripts/activate      # or bin/activate
pip install "pinnslab @ git+https://github.com/Ali-raza-5005/pinnslab@v0.3.0"
mkdir -p configs src/method results
```

`src/method/` is where a new sampler, weighting or optimizer is born
(CLAUDE.md rule 2). One file, one `@register_*`, imported before the config is
built — then a config may name it, with **zero** edits to `pinnslab`:

```python
# src/method/my_sampler.py
from pinnslab.geometry.samplers import Sampler, register_sampler

@register_sampler("mine")
class Mine(Sampler):
    def __init__(self, spec, problem): ...
    def __call__(self, state, current=None): ...   # draw from state.generator
```

```yaml
# configs/burgers.yaml
sampling:
  points:
    interior: {region: interior, n: 2000, strategy: mine, options: {bias: 2.0}}
```

A runner is about fifteen lines, and the same shape a Kaggle notebook has:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import method.my_sampler  # noqa: F401  — registers it

from pinnslab.registry.config import load_config
from pinnslab.registry.run import Run
from pinnslab.training.build import build_trainer
from pinnslab.utils.device import configure_runtime

cfg = load_config("configs/burgers.yaml")
ctx = configure_runtime(cfg)
run = Run.create_or_resume(cfg, "results", f"{cfg.identity_hash()[:12]}_s{cfg.seed}")
row = build_trainer(cfg, ctx, run).fit()
```

For sweeps, searches and figures, prefer the queue and the shipped scripts
(`pinnslab.training.queue`, `scripts/run_sweep.py`, `scripts/make_figures.py`)
over hand-rolled loops — they are what the resume, fairness and provenance
guarantees are attached to.

**Every result row records the commit it was produced by**, whichever way
pinnslab got installed: from a working tree (`git`), from
`pip install git+...@tag` (PEP 610 `direct_url.json`), or from an offline wheel
built for a Kaggle Dataset (`hatch_build.py` stamps it at build time). The row's
`git_source` field says which route answered, and `unknown` means none could —
treat that as a result you cannot publish.

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
| check it works on a real GPU    | `scripts/validate_gpu.py --json report.json`                      |

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
examples. 497 tests, and `CHANGELOG.md` records what each tag changed.

v0.3.0 is a research-readiness audit done before the first real experiments. It
found and fixed five things that would have produced wrong numbers quietly —
four of them in one seam, the batched search evaluator, where a field read off
`configs[0]` was applied to the whole population. The largest: that path
optimised a **different objective** than the config declared (a pooled mean over
all residual terms instead of a mean per term, a 6.3x difference on the shipped
Burgers example, with `weighting.coefficients` dropped entirely). CHANGELOG.md
has the full list; DESIGN.md §6 has the correction and the rule that follows.

Next is P0 on paper 1 (sampling). Infrastructure work from here is
paper-driven.

**Known limits, stated rather than discovered later.**

- **No GPU has run this code.** The batched population evaluator is measured at
  1.7–3.4x on CPU and DESIGN.md §6's "20–50x on a T4" is **unverified**, as is
  §5's FP64/FP32 ratio and the behaviour of
  `torch.use_deterministic_algorithms(True)` on CUDA.
  `scripts/validate_gpu.py` closes all of them in one command and prints a
  report; `scripts/benchmark_population.py` is the speedup measurement alone.
- **The batched evaluator is a narrow path, not "the fast version of a run".**
  It runs one flat loop of Adam over one fixed cloud, and now *refuses* — rather
  than approximating — a multi-stage schedule, a per-candidate learning rate or
  optimizer, a varying physical constant, a non-`mean` weighting, ascent,
  gradient clipping, inverse-problem parameters, and `resample_every`. Each of
  those belongs in `SequentialEvaluator`, which is the oracle.
- Only one benchmark (1-D Burgers) and one architecture (MLP) ship; `random` and
  `de` are the only search algorithms. An `Ensemble` cannot yet vary the
  activation across the population (it refuses a mixed one rather than
  silently unifying it), so activation search runs sequentially for now.
