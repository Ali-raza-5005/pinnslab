# examples

A complete experiment, small enough to run on a laptop CPU: **does
residual-adaptive sampling (RAD) beat uniform sampling on 1-D Burgers?**

Everything here is meant to be copied into a paper repo and changed. The
sampler under test lives in this directory rather than in `pinnslab`, because
that is where method code belongs (CLAUDE.md rule 2) — the library owns the
seam, the paper owns the method.

```
examples/
├── configs/burgers_uniform.yaml   the control arm
├── configs/burgers_rad.yaml       the treatment arm — differs in ONE field
├── rad_sampler.py                 the method: ~90 lines, one @register_sampler
├── run_matrix.csv                 2 arms x 5 seeds
└── search.yaml                    a search over the method's two knobs
```

## Run it

From the repository root, with the dependencies installed (`pip install -e
".[dev]"`, or just the runtime deps — the scripts add the checkout to
`sys.path` themselves):

```bash
# 1. one run, ~30s
python scripts/run.py examples/configs/burgers_uniform.yaml --results results/

# 2. the whole comparison: 2 arms x 5 seeds, ~7 min
python scripts/run_sweep.py examples/run_matrix.csv \
    --results results/ --register examples/rad_sampler.py

# 3. figures and tables, ~5s
python scripts/make_figures.py results/ --out analysis/ --by method \
    --order rad uniform

# 4. a search over RAD's k and c, ~4 min
python scripts/run_search.py examples/search.yaml \
    --base examples/configs/burgers_rad.yaml \
    --register examples/rad_sampler.py --root analysis/search
```

Kill any of them at any point and run the same command again: finished runs are
skipped, an interrupted one resumes from its last checkpoint, and the result is
bit-identical to an uninterrupted run.

`--register` is how the config's `strategy: rad` finds `rad_sampler.py`. Without
it, step 2 stops at load time with *unknown sampler 'rad'* — a registry cannot
resolve a name nobody imported.

## What it produced here

Measured 2026-08-17 on one laptop CPU (Windows, torch 2.12+cpu, float64), the
sweep took **7m25s** for 10 runs and gave:

| method  | rel-L2 median | IQR                  | seeds |
| ------- | ------------- | -------------------- | ----- |
| rad     | 6.14e-4       | [5.36e-4, 7.60e-4]   | 5/5   |
| uniform | 6.31e-4       | [4.78e-4, 6.34e-4]   | 5/5   |

**Read that as "no difference", and it is the honest outcome of this
configuration rather than a disappointment.** Two reasons, both worth
understanding before running your own version:

1. **The viscosity is smoothed.** These configs use `nu = 0.1/pi`, ten times the
   literature-standard `0.01/pi`, so the example finishes in minutes. At that
   viscosity the solution has no sharp front — and a front is precisely what
   adaptive sampling exists to resolve. Measured at the standard viscosity, this
   same 1600-step budget reaches rel-L2 **0.62** (a converged *loss* on a wrong
   solution); resolving it takes ~15k Adam steps, which is a GPU-sized job, not
   a laptop one. That is the experiment where the two arms would separate, and
   it is the one a paper would run.
2. **Overlapping IQRs over 5 seeds are not a result** in either direction. The
   table reports the spread and the seed count for that reason.

The per-second figure shows something the per-step one hides: RAD is *behind*
uniform at equal wall-clock, because scoring a 10 000-point pool every 250 steps
is not free. A method that wins per step and loses per second has not won, and
`make_figures.py` draws both axes by default so the question cannot be skipped
(DESIGN.md §8).

The search (step 4) agrees, and says so more sharply. In 4m08s it spent 4800
inner steps over 6 candidates x 2 generations and returned `k = 2.09`, `c =
10.0` at rel-L2 6.48e-4 — where `c = 10.0` is the *upper bound of the search
space*, i.e. the largest uniform floor it was allowed to pick. A search pinned
to the edge of its box is telling you the optimum is outside it: here, that
adaptivity is not buying anything at this viscosity, which is the same
conclusion the sweep reached and a good habit to read for.

## Making it a real experiment

- Set `nu: 0.0031830988618379067` (0.01/pi) and `steps: 15000` in both configs.
- Keep the arms identical in everything except `sampling.points.interior` —
  same point count, same resampling cadence, same optimizer schedule. Compute
  parity is the first thing a reviewer checks.
- Add the mechanism ablation: `k: 0.0` in `burgers_rad.yaml` is RAD's machinery
  running with its adaptivity switched off. If that scores like full RAD, the
  mechanism is not what is doing the work.
- Add the matched-budget random-sampling control, and count the search's own
  compute in the comparison (`SearchSpec.total_inner_steps`).
- Five seeds is the floor, not the target.

## Adding your own sampler

Copy `rad_sampler.py`, change the algorithm, name it in a config:

```python
from pinnslab.geometry.samplers import Sampler, register_sampler

@register_sampler("my_sampler")
class MySampler(Sampler):
    def __init__(self, spec, problem):      # spec is the PointSetSpec
        ...
    def __call__(self, state, current=None):
        # state gives you: nets, step, stage, dtype, device, extra_params and
        # `state.generator` — draw from that generator and nothing else, or the
        # cloud falls outside the checkpoint and resume stops being exact.
        return points                        # (n, dim)
    def state_dict(self):                    # only if you accumulate anything
        return {...}                         # it is checkpointed for you
    def load_state_dict(self, payload):
        ...
```

No edit to `pinnslab` is needed, and none is wanted: if you find yourself making
one, that is a `FRICTION.md` line.
