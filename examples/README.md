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

Re-measured 2026-08-28 on one laptop CPU (Windows, torch 2.12+cpu, float64),
10/10 runs completed:

| method  | rel-L2 median | IQR                  | seeds |
| ------- | ------------- | -------------------- | ----- |
| rad     | 7.46e-4       | [5.52e-4, 7.70e-4]   | 5/5   |
| uniform | 8.82e-4       | [6.82e-4, 9.32e-4]   | 5/5   |

**These are not the same numbers v0.2.0 printed** (rad 6.14e-4, uniform
6.31e-4), and the reason is worth understanding because it will happen again.
v0.3.0 added `EvalSpec.target_mode`, which changed every **config hash**; the
trainer's sampling stream is `derive_seed(seed, "trainer", config_hash)`, so a
new hash means a new collocation cloud and therefore a different draw from the
same distribution. Nothing about the physics or the training changed. The gap
between the two arms is smaller than the gap between two versions of the same
arm, which is the point the next paragraph makes anyway.

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

The search (step 4) agrees. In **4m11s** it spent **6000 inner steps** over 6
candidates x 2 generations and returned `k = 2.31`, `c = 1.08` at rel-L2
**7.15e-4** — no better than either arm of the sweep, from a search that cost
about as much as the whole sweep did.

Two things to read here, both of which changed in v0.3.0:

- **The cost number moved from 4800 to 6000 for the same search.** Not because
  the search got more expensive: `FidelitySchedule.cost` was *under-reporting*
  it. It charged a promoted candidate the increment `rungs[r] - rungs[r-1]`, as
  a warm-started ladder would, but both evaluators retrain a survivor from
  scratch. 6 x 200 + 3 x 600 = 6000 is what actually ran, and it now matches
  what `SearchState.total_inner_steps` measures. Compute parity including search
  cost is a reviewer defence (DESIGN.md §8); an under-reported budget is a hole
  in it.
- **The incumbent moved a long way on a hash change alone.** v0.2.0's run of
  this same search returned `c = 10.0` — pinned to the *upper bound* of the
  space, which reads as "the optimum is outside the box". This run returned
  `c = 1.08`, comfortably inside it. Same search, same seed, same spec; only the
  candidates' collocation clouds differ. A 2-generation search over 6 candidates
  does not have enough evidence to distinguish those, and that is the honest
  reading: **this search is too small to conclude anything**, and it is sized
  for a laptop demo rather than for a result. A paper's version needs the seeds
  and the generations that DESIGN.md §8 asks for.

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
