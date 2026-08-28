# pinnslab — Design Decisions

Living record of *why* the library is shaped the way it is. Read this before
making structural changes. If you're about to violate something here, update
this file with the reason first.

---

## 0. What this is

A personal research infrastructure for PINN **methods** research. The author's
program is: **population-based / metaheuristic search over PINN configuration
space** — sampling strategies now; loss weighting, hybrid optimizers, network
architecture, and activation functions later.

It is NOT a general-purpose PINN library and is NOT intended for external
adoption. Its only success metric is the author's papers-per-year. Do not add
breadth (PDE zoos, tutorials, multi-backend, docs sites) to chase
publishability — that actively harms the tool.

---

## 1. Core architectural decision: build from scratch, depend on DeepXDE thinly

- Built from scratch on **PyTorch**. We OWN the training loop, because the
  research novelty lives *inside* the loop (sampling, weighting, optimizer
  schedule, architecture). Wrapping a framework whose loop is fixed would mean
  monkey-patching for every paper.
- **DeepXDE is a dependency, never a foundation.** It is imported in exactly
  ONE place: `pinnslab/geometry/adapters.py`. No DeepXDE object escapes that
  module — it returns raw `torch.Tensor`s only. If DeepXDE breaks or we outgrow
  it, we rewrite one file.
- DeepXDE is used for two things only:
  1. **Geometry as a point generator** (CSG domains, boundary/IC sampling,
     domain-membership tests). Building this from scratch is the genuinely
     expensive part; borrowing it is the whole point.
  2. **Baseline oracle** — stock, unmodified DeepXDE runs produce the
     comparison numbers (RAR, RAD, importance sampling, uniform). This is a
     rebuttal asset: "baselines implemented in the standard library."
- DeepXDE was also to be a **golden-test reference**: our from-scratch Burgers
  matching DeepXDE's result within tolerance, catching silent implementation
  bugs. **Audited 2026-08-28: this was never built, and it should not be.**
  `tests/golden/test_burgers.py` checks against the **Cole-Hopf exact
  solution** instead, which is a strictly stronger oracle — an analytical
  reference cannot share a bug with us, whereas two PINN implementations
  agreeing on a wrong number is exactly the failure mode this was meant to
  catch. The claim is retired rather than implemented.

  DeepXDE's *other* role is untouched and still owed: **stock, unmodified
  DeepXDE runs produce the baseline comparison numbers** (RAR, RAD, importance
  sampling, uniform), which is the rebuttal asset "baselines implemented in the
  standard library". That is a paper-level task — it belongs in a paper repo
  under rule 2, and it is not due until paper 1's P1.

## 1a. Why not PhysicsNeMo / Modulus

Wrong center of gravity (large surrogate models, digital twins, multi-node
clusters), heavy container-oriented install unsuited to Kaggle/Colab, and high
API churn (Modulus→PhysicsNeMo rename + v2.0 refactor + DGL→PyG migration).
Ruled out.

---

## 2. Two-tier repo architecture

**Tier 1 — `pinnslab`** (this repo): pip-installable package, versioned forever.
Boring, stable, slightly behind the research frontier.

**Tier 2 — `paper-NN-shortname`** (one repo per paper): only what is unique to
that paper — configs, novel method code (`src/method/`), analysis, figures,
LaTeX.

### Promotion rule
New code is born in the paper repo under `src/method/`. It moves into `pinnslab`
**only when a second paper needs it.** Not before. Premature promotion is how
core libraries rot into abstraction soup.

### Deletion rule
Anything in `pinnslab` unused across two consecutive papers gets deleted.
Reviewed between papers, alongside the icebox.

### Versioning contract (this is what makes being wrong safe)
- Every paper repo pins an exact tag: `pinnslab @ git+...@v0.4.2`.
- **Never break a published paper.** Old paper installs old tag, reproduces
  byte-identically. This freedom is what lets us refactor fearlessly.
- Tag at submission time as `paper-NN-submitted` in addition to semver.
- Every results row records `pinnslab.__version__` + git SHA + config hash.
  Non-negotiable.

---

## 3. Package layout

```
pinnslab/
├── pyproject.toml
├── pinnslab/
│   ├── geometry/     # ONLY place deepxde is imported (adapters.py)
│   ├── models/       # MLP, modified-MLP, Fourier features, SIREN
│   ├── physics/      # PDE residual operators, diffops (autograd + torch.func)
│   ├── losses/       # weighting: fixed, NTK, GradNorm, self-adaptive, causal, min-max
│   ├── training/     # Trainer, Adam→L-BFGS, checkpoint/resume, Kaggle launcher
│   ├── search/       # metaheuristic layer (see §6) — the program's heart
│   ├── eval/         # rel-L2, residual norms, conservation, time-to-accuracy
│   ├── benchmarks/   # frozen canonical PDEs: Burgers, AC, KdV, Helmholtz, NS-lid, wave
│   ├── viz/          # plot primitives + one house-style module
│   ├── registry/     # Run object, config hashing, results schema, manifest
│   └── utils/        # seeding, determinism, logging, device management
└── tests/
    ├── unit/         # fast, CPU-only, <60s total excluding `slow`
    └── golden/       # regression: "Burgers @ config X reaches rel-L2 <= 1.2e-3 in 20k steps"
```

### The two test commands (decided 2026-08-08)

- **Pre-commit loop**: `pytest -m "unit and not slow"` — the <60s budget applies
  to *this* number, because this is the command whose latency decides whether
  tests get run at all.
- **Pre-push / pre-tag**: `pytest` — everything, `slow` and `golden` included.

`slow` marks tests that spawn a subprocess and therefore pay a fresh
`import torch` (~4-10s each, and one also pays `import deepxde`). Each is
out-of-process for a real reason that cannot be faked in-process — the DeepXDE
backend is chosen once at first import, `derive_seed` must survive
`PYTHONHASHSEED` varying between processes, and a config hash must be identical
in the Kaggle session that resumes a run as in the one that started it — so none
should be deleted or rewritten in-process. They are simply not worth 16s on
every commit. `test_resume_is_bit_exact` is deliberately **not** marked: it is
slow (~8s) because it genuinely trains twice, but it is the load-bearing test of
the checkpoint layer and belongs in the loop that runs before every commit.

---

## 4. The RunConfig schema (stable axis — invest abstraction here)

Separate **stable** axes (what a run is, checkpointing, results schema, metrics,
geometry→points) from **volatile** axes (architectures, weighting, sampling,
optimizer schedules, PDE families). Invest abstraction ONLY on the stable axes.
Keep volatile pieces as flat, near-copy-pasteable ~40-line files. Duplication in
the volatile column is cheap; wrong abstraction there kills the library.

```python
class RunConfig(Spec):
    problem: ProblemSpec | None           # which frozen benchmark (geometry, BCs, reference)
    nets: dict[str, NetSpec]              # 1..N networks, named (multi-net, XPINN, per-field)
    extra_params: dict[str, ParamSpec]    # inverse-problem trainable unknowns
    residuals: dict[str, ResidualSpec]    # named PDE residual terms (coupled systems)
    weighting: WeightingSpec              # dict[str, Tensor] -> scalar (per-term AND per-point)
    sampling: SamplingSpec                # named point groups: {"interior": PointSetSpec, ...}
    stages: list[StageSpec]               # sequential/staged training
    eval: EvalSpec

class StageSpec(Spec):
    optimizers: list[OptimizerSpec]       # each: param selector + min|max direction
    steps: int
    resample_every: int | None
```

Three refinements made when this was implemented (2026-08-08), all within the
sketch's intent:

- **`output_transform` moved onto `NetSpec`**, not `RunConfig`. With multiple
  networks a single run-level transform cannot express a per-field hard
  constraint, and per-net degenerates to the run-level case when there is one net.
- **`problem` was added.** The PDE is not a hyperparameter: geometry, BC/IC forms
  and the reference solution live in a frozen `benchmarks/` module so every paper
  compares against the same problem. The config chooses *which* benchmark plus the
  physical constants that benchmark declares varyable.
- **Every volatile field defaults to empty**, because `Trainer` takes `nets` and
  `residual_fn` as plain callables — that IS the escape hatch this section
  requires, and it is how the infrastructure tests drive the loop with no
  `physics/` in existence. `assemble()` refuses to build anything the config did
  not declare, so the defaults cannot become a way to smuggle hyperparameters
  back into a script (rule 4).

### Two load-bearing decisions
1. **Residual functions return per-point tensors of shape `(N,)`, never
   scalars.** Reduction happens in the weighting object. If residuals
   pre-reduce, per-point weighting (causal, self-adaptive, RBA) becomes
   impossible without editing every residual.
2. **Optimizers are a `list` with param selectors and a direction**, not a
   single object. Makes min-max / self-adaptive schemes fall out for free.

### Conformance test — the substitute for predicting the future
Before finalizing `Trainer`, confirm all SEVEN are expressible **without editing
core**:
1. Multiple networks (per-field / per-subdomain)
2. Inverse problems (PDE coeffs as trainable params)
3. Min-max / self-adaptive weights (second optimizer doing ascent)
4. Per-point loss weights (not just per-term scalars)
5. Hard constraints (output transform)
6. Sequential/staged training (Adam→L-BFGS, time-marching, curriculum)
7. Coupled systems (vector outputs, multiple residuals, different scales)

If any requires editing core, the seams are wrong.

### Extension by registration, not inheritance
```python
@register_weighting("causal")
class CausalWeighting:
    def __call__(self, residuals: dict[str, Tensor], state) -> Tensor: ...
```
New method = one new file, zero edits to existing files. No deep inheritance,
no base class with 15 hooks. Every abstraction has a documented escape hatch:
a genuinely strange paper must be implementable entirely inside
`paper-NN/src/method/` (custom registered component, or a custom `step_fn` in
the worst case) without editing `pinnslab`.

The `@register_*` decorators live in **`pinnslab/components.py`**, NOT in
`pinnslab/registry/`. Two unrelated senses of the word collided: §3's `registry/`
is *run provenance* (Run object, config hashing, results schema). Separate
modules, no shared meaning.

**A registry nobody reads is not an extension point** (learned 2026-08-17).
`SAMPLERS` existed from the start and nothing consulted it: `build.py` passed a
config's `strategy:` straight to DeepXDE, so the five geometric draws worked and
the *one* axis this whole research program is about — adaptive sampling — could
not be added without editing core. `geometry/samplers.py` now owns the lookup
(`build_sampler(spec, problem)`), one built-in registration per geometric
strategy, and a sampler is called `sampler(state, current)`: the same
`(state, points)` shape as a residual term, so there is one convention to learn.
Everything an adaptive sampler needs — the networks, the step, the cloud it is
replacing, the trainer's generator — is already on `TrainState`; no second
abstraction was added, and none is needed. Worked example:
`examples/rad_sampler.py`.

### What the config hash covers (decided 2026-07-31)
`RunConfig.identity_hash()` hashes the condition, not the invocation. It
**excludes** `seed`, `device`, `name`, `tags`, `logging`, `checkpoint`:

- **seed is excluded** so that five seeds of one condition share a hash — the
  §8 "median + IQR over >=5 seeds" analysis needs something to group on. A run
  is therefore identified by the **pair `(config_hash, seed)`**, and both are on
  every row; checkpoint resume verifies both.
- **logging/checkpoint cadence is excluded** because changing how often you
  write a trace point does not make it a different experiment, and pretending
  it does would fragment the search layer's candidate cache.
- **`dtype` IS hashed** — float32 and float64 results are not comparable (§5).

---

## 5. Numerics, determinism, hardware

- **float64 default**: `torch.set_default_dtype(torch.float64)`. PINNs hit a
  float32 residual noise floor at ~1e-4–1e-5, exactly where papers claim wins.
- **Precision-by-GPU rule** (Kaggle):
  - **T4 ×2** → float32 work: hyperparam search, seed sweeps, ablations
    (T4 FP64 is ~1/32 of FP32).
  - **P100 ×1** → float64 work: final headline runs, machine-precision claims
    (P100 FP64 ~1/2 of FP32).
  - Record `dtype` and `device_profile` in every result row. Never mix
    precisions within a comparison group.
- **diffops** (`physics/diffops.py`): one API, two backends — classic
  `torch.autograd.grad(create_graph=True)` and `torch.func.jacrev/hessian` +
  `vmap`. A/B once, then stop thinking about it.
- **`torch.compile` OFF by default**, behind a config flag. Double-backward
  through compiled graphs breaks silently across versions. Golden tests run it
  both on and off.
- **L-BFGS checkpoints cleanly — measured, not assumed.** `torch.optim.LBFGS` is
  full-batch and closure-based, but its curvature history (`old_dirs`,
  `old_stps`, `ro`, `H_diag`, `prev_flat_grad`, `d`, `t`) *is* in `state_dict`,
  so a mid-stage resume is bit-exact and needs no special handling. Verified on
  torch 2.12 with and without `strong_wolfe`, and pinned by
  `test_lbfgs_resume_is_bit_exact` — if a future torch moves that state out,
  the test fails instead of the run silently becoming a different experiment.
  An L-BFGS phase therefore does **not** have to fit inside one Kaggle session.
  Still give L-BFGS its own stage (it re-evaluates the loss internally, so a
  concurrent ascent optimizer would step from a different iterate) and use
  `line_search_fn='strong_wolfe'` — without a line search it stalls or diverges
  on PINN losses.
- **Determinism** (`utils/seeding.py`, called by every run):
  ```python
  torch.use_deterministic_algorithms(True)   # env: CUBLAS_WORKSPACE_CONFIG=:4096:8
  torch.backends.cudnn.benchmark = False
  ```
  Costs speed. Take it — a PINN result you can't reproduce bit-for-bit can't be
  defended in rebuttal.
- **Hardware-mixing rule**: an entire comparison group (every method × seed in
  one figure) must run on ONE GPU type. Enforce in code: every row records
  `gpu_name`; aggregation asserts uniformity within a group and refuses to plot
  otherwise.

---

## 6. The metaheuristic search layer — the program's heart

The four research directions (sampling / weighting / optimizer / architecture)
are THE SAME CODE with a different `RunConfig` field selected. Write the search
machinery once; each paper is a new search space + fitness function.

```python
@dataclass
class SearchSpec:
    space: dict[str, Domain]        # config paths -> {continuous, integer, categorical}
    algorithm: str                  # registered: de, cmaes, pso, ga, gwo, random
    pop_size: int
    budget: FidelitySchedule        # inner steps per generation (multi-fidelity)
    fitness: FitnessSpec
```

### Population evaluated in parallel via vmap — this is what makes it feasible
Nested optimization: pop 30 × 100 gens = 3,000 inner trainings PER cell; ×seeds
×PDEs ×baselines → 1e4–1e5 trainings. Sequential on Kaggle quota = impossible.
PINNs are tiny MLPs (pop 50 × 5k params = 250k total, nothing). Batch the whole
population as one vmapped model → ~20–50× throughput on a T4.

**This is structurally impossible through DeepXDE** (stateful OO Model). It's a
core reason we build from scratch.

### CORRECTION (measured 2026-08-08): it is a batched graph, not `vmap`

This section originally prescribed `torch.func.vmap` over `stack_module_state` +
`functional_call`. **That does not compose with a PINN residual.** A residual
differentiates the network with respect to its *inputs*, so it must flag the
collocation points with `requires_grad_()`, and `vmap` refuses:

```
RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps
using torch.autograd.functional.* APIs) inside of a function being transformed
by a function transform
```

Making `vmap` work would mean rewriting every residual against
`torch.func.jacrev`/`hessian` — a second way to spell every PDE, i.e. exactly
the per-paper monkey-patching §1 rejects.

**What we do instead keeps the goal and drops the mechanism: put the population
on a leading batch dimension and build one graph.** `search/population.py`'s
`Ensemble` evaluates P identical-shaped MLPs as batched matmuls, so inputs are
`(P, N, d)`, outputs `(P, N, m)`, and plain `torch.autograd.grad` works
unchanged including second derivatives. `physics/diffops.py` indexes with `...`
rather than a leading colon, so **a residual written once serves both a single
run and a whole population** and never learns the population exists.

Three measured facts make it correct rather than merely convenient:

1. **Independence.** `grad_outputs=ones` sums before differentiating, and output
   element `(p, n)` depends only on candidate `p`'s parameters and point
   `(p, n)`, so every cross term is identically zero. Batched and separate
   evaluation of a Burgers residual agree to **0.0e+00**.
2. **One Adam is P Adams.** Adam is elementwise with per-element state, so a
   single optimizer over stacked `(P, ...)` parameters *is* P independent Adams
   — provided what reaches `backward` is the **sum** over candidates. Measured
   parameter drift against P separate trainings after 25 steps: **1.1e-16**.
3. **Speed — and the honest number.** On this CPU with a real Burgers residual
   (width 20, depth 3, N=512): 1.7× at P=4, 2.8× at P=8, **3.4× at P=16**,
   falling to ~2.2× by P=50. The "20–50× on a T4" above is a *GPU* claim about
   kernel-launch overhead dominating for tiny nets and is **untested — there is
   no GPU here.** Do not put it in a paper until it is measured on the hardware
   in question: compute parity is a reviewer defence and an unverified speedup
   is a hole in it.

### CORRECTION 2 (audited 2026-08-28): the batched path is a *narrow* path

The batched evaluator is not "the fast version of a run". It is a different,
much smaller machine that happens to produce a comparable number, and the audit
found it silently pretending otherwise in four places. All four are the same
mistake: **a field read off `configs[0]` and applied to the whole population.**

1. **It optimised a different objective.** `train_population` reduces with one
   *pooled* mean of squares over every residual row. `MeanWeighting` is a mean
   **per term**, then a sum. Equal only when every term has the same point
   count — never. On `examples/configs/burgers_uniform.yaml` (pde 1150, ic 100,
   bc 50) the two differed by 6.3x and the boundary term carried 26x too little
   weight. Fixed by scaling term *k* by `sqrt(coeff_k * N_total / N_k)` before
   the pooled reduction, which reproduces `MeanWeighting` exactly and, being
   per-candidate, makes loss-weight search work on this path.
2. **`Ensemble` always used `tanh`**, whatever the config declared.
3. **Per-candidate `lr`, optimizer, physical constants and multi-stage
   schedules were ignored**, so a search over any of them scored every candidate
   at candidate 0's setting while archiving distinct config hashes.
4. **`resample_every` was ignored** — `train_population` draws one cloud and
   never resamples, so a *sampling* search would have measured nothing.

The rule that follows, and it is now enforced in `_reject_unsupported`: **the
batched path must refuse anything it cannot express, never approximate it.**
`SequentialEvaluator` is the oracle precisely because it has no such gap, and
"use SequentialEvaluator" is an acceptable answer to any of these.

The deeper lesson is about the tests. `test_batched_and_sequential_agree_on_
the_training_objective` called itself "THE test of this module" and computed its
oracle by *re-implementing* the pooled mean instead of calling the config's
weighting object, so both sides computed the same wrong number. That is the
third time in this repo a test has passed on a premise that cannot occur — after
`viz.aggregate.band`'s equal-timestamp fixture and `SAMPLERS`, the registry
nothing read. The pattern: **an oracle that reimplements the thing under test is
not an oracle.** Call the real object, or the test pins the bug.

**What breaks independence**: anything reducing across the population. Global
gradient-norm clipping is the trap — one norm over all P candidates means a
single diverging candidate damps everyone else's step. `train_population`
refuses it rather than producing a quietly coupled search.

Scope caveat (unchanged, and it applies to the batched graph identically):
shared tensor shapes are required → clean for sampling, loss weights,
weight-space optimization. Architecture search over varying depth/width doesn't
batch directly → group candidates by shape, batch within groups. Activation
search batches fine (index into a fixed set).

The batched path also fixes the **point count** across the population, which
looks like a limit on the flagship "how many collocation points?" axis but is
not: §8 already requires an identical collocation count across compared methods
for the comparison to be fair, so a sampling search at fixed budget is the
correct default and *where* the points go is the question. Varying the count
runs through `SequentialEvaluator`, which is also the oracle the batched path is
tested against, and the only path that can score against a reference solution.

### Three things not to skip
1. **Outer-loop checkpointing**: population, generation counter, archive, AND
   the metaheuristic's own RNG state. A search spans many Kaggle sessions.
   Wrong RNG state on resume breaks reproducibility invisibly.
2. **Candidate cache** keyed by config hash — metaheuristics re-evaluate
   duplicates constantly. Free speedup.
3. **Multi-fidelity from day one**: short inner runs early, longer for survivors
   (successive halving). Build the fidelity schedule into SearchSpec, don't bolt
   it on later.

---

## 7. Kaggle / Colab execution

- **Storage split**: GitHub = code + configs + LaTeX. Small results (metrics,
  configs, manifests) → a `paper-NN-results` git repo (versioned, diffable).
  Large artifacts (checkpoints, fields) → Kaggle Datasets (versioned, canonical).
  Frozen bundle at submission → Zenodo/OSF for a DOI. Google Drive = scratch
  mount only (unversioned, don't trust it).
- **Never edit `pinnslab` on Kaggle.** Sessions are ephemeral. Edit locally →
  push → bump tag → `pip install git+...@tag`. Kaggle notebooks are ~20 lines:
  install, load config, `pinnslab.training.run(cfg)`, save artifacts.
  Offline install: build wheel, upload as Kaggle Dataset, `pip install
  --no-index --find-links=... pinnslab`.
- **2-GPU strategy**: do NOT reflexively use DDP — PINN nets are tiny, all-reduce
  costs more than it saves (<1.3×). Run two independent configs concurrently,
  one per GPU via `CUDA_VISIBLE_DEVICES`. True 2× on seed/ablation sweeps.
- **Session-death resilience**: checkpoint (weights + optimizer + RNG state +
  step) every N minutes to `/kaggle/working`; every run starts by looking for a
  checkpoint in a mounted Dataset and resuming. Size runs to finish under the
  session wall clock, or make them resumable across sessions by design.
- **Run queue, not manual notebooks** (`training/queue.py`, built 2026-08-08):
  `run_matrix.csv` lists every cell; the notebook works through them.

  **Status is derived, never written.** The matrix is an immutable declarative
  input — a list of `(config, seed)` pairs — and a cell's `run_id` is a pure
  function of that pair (`config_hash[:12] + "_s" + seed`). So the run directory
  *is* the claim: "claimed" = it exists, "done" = it has `result.json`. This
  file originally specified a mutable status column that the notebook marked
  done; deriving it is strictly better here, and the reasons generalise:
  1. **Rule 6 holds by construction** — no mutable experimental bookkeeping to
     overwrite, and no way for the queue's idea of what ran to disagree with
     what is on disk.
  2. **A killed session leaves no lie** — a status written *before* the work
     strands rows in `claimed` forever; written *after*, it loses every
     interrupted run. The directory is correct either way, because the run
     itself created it.
  3. **Two GPUs never contend** — see below.

  **Claiming is lock-free because workers partition statically.** Worker *k* of
  *n* takes matrix rows where `index % n == k`. No two workers ever consider one
  cell, so a stale claim cannot exist and no lease or heartbeat is needed to
  detect one. Within a worker's slice, unfinished cells are claimed before
  untouched ones: only started work has compute at risk.

  **Only `seed` may live in the matrix**, never a hyperparameter — every other
  axis gets its own YAML file. Seed is already excluded from the config hash
  (§4), so it is the one field that can sit outside a config without breaking
  the no-literals rule, and five seeds of one condition still share a hash for
  the §8 groupby.

  Proven against a real `os._exit` mid-cell, not a raised exception:
  `tests/unit/test_queue_survives_a_killed_session.py` asserts the killed sweep
  is **bit-identical** to an uninterrupted one. A queue that silently restarted
  the interrupted cell would pass every weaker test.

  **Resolved 2026-08-17** (was: "known gap, guarded rather than fixed"). The
  collocation cloud and the sampler's own `state_dict` are part of the
  checkpoint payload (format version 2), so a run with `resample_every` set
  resumes on the cloud it was training on. The queue's refusal of that
  combination, and its `allow_resampling` waiver, are gone.

  The decision the gap was waiting on: **the cloud is stored, not replayed.**
  Replaying it from the RNG stream works only while sampling is a pure function
  of that stream — the moment a sampler reads the network (which is the whole
  of paper 1), the cloud in force at step *k* depends on a network that no
  longer exists after the resume, and the only cheap correct record of it is
  the cloud itself. Cost: one `(N, d)` tensor per point group per checkpoint,
  which is small beside Adam's two moments per parameter.

  Pinned by `tests/unit/test_resampling.py` (in-process, both a plain and an
  adaptive sampler) and by the SIGKILL test above, whose killed cell is now the
  resampling one, killed *between* resamples.

---

## 8. Statistical rigor + reviewer-proofing (bake into every PROTOCOL.md)

- **>=5 seeds** for any headline claim. Report **median + IQR**, not mean±std
  (PINN errors are heavy-tailed; one divergent run wrecks a mean). Report
  failure rate explicitly ("3/5 seeds converged").
- **Baseline fairness**: identical tuning budget, identical collocation point
  count, identical wall-clock across all compared methods. Report
  time-to-target-accuracy alongside final accuracy.
- **Metaheuristic-specific attacks to pre-empt (this is desk-reject territory):**
  1. **Compute parity** — report every comparison at equal total FLOPs / equal
     wall-clock, INCLUDING search cost. A GA that burns 3000× compute winning is
     not a result.
  2. **Random search baseline** at matched budget — always. If the metaheuristic
     doesn't beat random search at equal budget, there is no result. Discover
     this in P1, not from Reviewer 2.
  3. **Serious-optimizer baseline** — also report CMA-ES or differential
     evolution. Frame the contribution as "metaheuristic search improves PINN
     sampling," NOT "GWO is the right algorithm" (there's a real community
     backlash against reskinned nature-inspired metaheuristics). Lets you swap
     algorithms without changing the claim.
  4. **Mechanism ablation** — isolate WHY it works: the search algorithm, or
     just having more points in high-residual regions? A reviewer will ask.
- **Figures are never hand-made.** Every figure produced by a script in
  `phases/p3_analysis/figures/` reading from `results/`. Zero manual edits.
  A single `pinnslab.viz.style` module keeps all papers consistent, and
  `scripts/make_figures.py` is the worked example a paper's own script copies.

### Figure conventions (built 2026-08-08, `viz/`)

The house style exists so these are decided once, not per figure:

- **The primitive is a band, not a line.** `viz/convergence.py` plots the
  median across seeds with the IQR shaded and the seed count in the legend. A
  single-seed convergence plot is a plot of which seed the author picked — the
  Burgers bring-up produced 0.030 to 0.435 across seeds of one *correct*
  config.
- **Diverged runs are excluded from the band and counted in the failure
  annotation.** Both come from the same population, so the legend cannot say
  `n=5` while the note says one of five failed. `include_diverged=True` for the
  figure whose subject *is* divergence.
- **Both x-axes, always.** `step` compares at equal iterations, `wall_time` at
  equal compute. A method that wins per-step and loses per-second has not won.
- **No LaTeX at render time.** `text.usetex` is off; mathtext with the `cm`
  font set gives Computer Modern without a figure that dies in CI or on Kaggle.
  Opt in per figure if a submission needs real macros.
- **Palette is measured, not chosen.** Okabe-Ito, reordered so the worst
  *adjacent* pair is ΔE 9.6 (OKLab ×100, Machado et al. 2009 CVD model) on
  paper white. SciencePlots' default cycle was rejected on measurement: its
  orange `#FF9500` and green `#00B945` land ΔE 2.8 apart under protanopia —
  indistinguishable, and they are the first two colors a "ours vs baseline"
  plot reaches for. Paul Tol's `bright` fails the lightness and chroma checks.
  Colors are assigned in fixed order and **never cycled**; a 7th series is an
  error, not a repeated hue.
- **Color is never the only channel.** Every slot carries a linestyle and a
  marker, for B&W print (IEEE requires it) and because three of the six sit
  below 3:1 contrast on white.
- **Fields**: perceptually uniform only — `viridis` for magnitude, `RdBu_r`
  for signed fields and *always* through `style.symmetric_norm`, which centres
  the neutral midpoint on zero. Matplotlib's default norm centres it on the
  data's midpoint, drawing a sign change the solution does not have. No
  rainbow: `jet` distorts by up to ~8% and is unreadable in grayscale.
- **Tables are generated too** (`viz/tables.py`), booktabs, median [IQR], seed
  count, and a failure column that appears only when something failed. One
  notation per column — `9.34e-4` above `0.00111` is the same quantity written
  two ways, in the cell a reader compares.
- **`results/` is append-only.** Never overwrite. Aggregation reads raw, writes
  derived files to `analysis/`.

---

## 9. Bootstrap order (build infra before science)

1. `utils/seeding`, `registry/` (Run object, config hashing, results schema),
   `training/checkpoint` (save/resume incl. RNG state), `training/trainer`
   (bare Adam→L-BFGS loop). Nothing else.
2. One benchmark end-to-end: 1D Burgers + vanilla MLP. Golden test asserting
   target rel-L2.
3. ~~Kaggle runner: <=20-line notebook, install-from-tag, read run_matrix.csv,
   claim/run/checkpoint/push/mark-done. Prove it survives a killed session.~~
   **Done 2026-08-08** — `training/queue.py` + `notebooks/kaggle_runner.py`.
   "Mark done" turned out to be unnecessary: status is derived (see §7).
4. ~~`viz/style.py` + one figure script reading results → publication-ready
   convergence plot. Full loop config→figure, zero manual steps.~~
   **Done 2026-08-08** — `viz/{style,aggregate,convergence,tables}.py` +
   `scripts/make_figures.py`. Conventions and the measured palette in §8.
5. ~~`search/` layer: SearchSpec + vmap population evaluator + outer-loop
   checkpoint + cache + multi-fidelity.~~ **Done 2026-08-08** —
   `search/{space,spec,algorithms,population,evaluate,cache,state,loop}.py`.
   The population evaluator is a batched graph, not `vmap`; see the correction
   in §6. THEN start P0 on paper 1 (sampling).

Config system: **YAML + pydantic (DECIDED — no Hydra).** YAML on disk → load to
dict → validate into pydantic models → hash the validated object for provenance.
Rationale: (a) our `search/` layer already does what Hydra's multi-run sweeper
does, and Hydra's sweeper would fight it; (b) a pydantic model *is* a schema that
maps directly onto `SearchSpec.space` — the typed field bounds double as the
metaheuristic's search space; (c) solo + Kaggle favors plain loadable YAML over
Hydra's entry-point/working-dir takeover. Hydra can layer on pydantic later if
config *composition* ever becomes genuinely painful; choosing pydantic now
doesn't close that door.

No hyperparameter is ever a Python literal in a script — every number lives in a
validated, hashed config.

---

## 10. Scope — what pinnslab IS and IS NOT for

### pinnslab IS for
- Running YOUR training loop with full control over sampling, weighting,
  optimizer schedule, architecture, activations.
- Metaheuristic/population search over `RunConfig` space, evaluated in parallel
  via vmap (the core capability no other library gives us).
- Reproducible multi-seed experiments with full provenance (version, SHA, config
  hash, gpu, dtype, seed on every row).
- Fast implementation of a NEW loop-level mechanism as one registered file.
- Consistent benchmarks + figures across all the author's papers.

### pinnslab is NOT for
- Being a general-purpose PINN library for strangers. No PDE zoo, no tutorials,
  no multi-backend, no docs site, no API stability promises to outsiders.
  Adding these to chase adoption/publishability actively harms the tool.
- Re-implementing geometry — borrow DeepXDE via `geometry/adapters.py`.
- Re-implementing baselines — run stock DeepXDE as the oracle.
- Wrapping/subclassing another framework's training loop.
- Large surrogate/operator/digital-twin work (that's PhysicsNeMo's world).
- Anything that must run on multi-node clusters (we target Kaggle/Colab).

### The honest test, revisited after every ~3 papers
"If a stranger wrote this, would I use it over DeepXDE?" The answer is expected
to be NO — that's correct; it's a tool shaped to one hand. Only reconsider
publishing if `FRICTION.md` shows repeated things DeepXDE genuinely cannot
express AND colleagues are asking for updates to a zip you sent them.

---

## 11. Data storage — what to save, what not to

Guiding principle: **store what you cannot cheaply recompute; recompute the
rest.** Raw run outputs are precious (GPU-hours to regenerate); derived
aggregates are disposable (seconds to regenerate). Never confuse the two.

### SAVE (precious — append-only, versioned)
Per inner run, one row/record:
- **Identity/provenance**: run_id, config_hash, pinnslab __version__, git SHA,
  seed, gpu_name, dtype, device_profile, timestamp.
- **The config itself** (the validated YAML), or its hash if stored centrally.
- **Final + best metrics**: rel-L2, residual norms (per term), boundary/IC
  error, conservation error where relevant, best-so-far and final.
- **Convergence trace** but DOWNSAMPLED: log metrics on a schedule (e.g. every
  100–500 steps or log-spaced), NOT every step. A full per-step trace for 1e5
  runs is gigabytes of noise; log-spaced captures the curve shape.
- **Checkpoints**: weights + optimizer + RNG state + step. Keep BEST and LAST
  only by default; keep every-N only when a specific analysis needs it.
- **Failure records**: diverged runs are DATA, not garbage. Store the row with
  status=diverged and the reason. Failure rate is a reported metric.

For the SEARCH layer specifically:
- Per generation: best/mean/worst fitness, population diversity, the incumbent
  config. Full population history only if diversity analysis needs it.
- The candidate cache (config_hash → fitness), so resumed/rerun searches skip
  duplicates.
- Outer-loop checkpoint: population, generation, archive, AND metaheuristic RNG
  state.

### TIMING — store these, they are first-class experimental results here
Because compute-parity is a core reviewer defense, timing IS a metric, not
metadata:
- **inner_train_time** per candidate/run (wall-clock). On the result row as
  `timings["train_seconds"]`.
- **loop/generation time** for the search outer loop. `GenerationReport.seconds`
  and `Evaluation.seconds` (**built 2026-08-28** — this section had asked for it
  since the start and nothing recorded it; the search archive held fitnesses and
  no clock at all, so the one number §8's compute-parity defence needs about the
  search itself was the one number the search did not keep).
- **total_search_time** and **total_experiment_time** (cell-level).
  `SearchState.total_seconds` and `total_inner_steps`, **accumulated across
  sessions** rather than measured from the current process — a search spans many
  Kaggle sessions, so a per-process clock would report the last session's cost
  as the whole search's. `SearchState.provenance` carries rule 7 for the search,
  because a search produces numbers a paper quotes.
  Note `SearchSpec.total_inner_steps` is a *bound* and `SearchState.
  total_inner_steps` is the *measurement*; the cache is why they differ, and
  `scripts/run_search.py` prints both.
- **time-to-target-accuracy** (steps AND seconds to reach a fixed rel-L2), plus
  `time_to_target_resolution_steps` — **decided 2026-08-17**. The target was
  only ever checked on the trace schedule, which made a reviewer-facing number
  depend on `logging.trace`, a field deliberately *excluded* from the config
  hash: two runs of one condition tracing at different densities would report
  different times for identical training. Now a target the step already computed
  (`loss`, `residual/<name>`) is checked every step and the resolution is 1; an
  `eval_fn`-derived target (`rel_l2`) stays on the schedule, because paying a
  full evaluation-grid forward pass per step is exactly what the schedule
  exists to avoid — and the run records the cadence it was observed at, so an
  upper bound is never read as an exact value.
- **FLOPs or step-count** per method, for equal-budget comparison including
  search cost.
- gpu_name alongside every timing (T4 vs P100 vs Colab timings are NOT
  comparable — the hardware-uniformity rule applies to timing too).

### DO NOT SAVE (recompute or discard)
- Per-step full-resolution loss for every run (downsample instead).
- Every-epoch checkpoints by default (best+last suffice; storage explodes).
- Derived aggregates (means, medians, IQRs, ranked tables) — these are
  regenerated by `analysis/` scripts from raw rows. Storing them invites
  stale/inconsistent numbers.
- Rendered figures as source-of-truth — figures are outputs of scripts; the
  script + raw data is the source of truth. (Keep the final submitted PDFs, of
  course.)
- Redundant copies of reference solutions — store once, reference by key.
- Large intermediate tensors (full collocation point clouds per step, full
  field snapshots every epoch) unless a specific figure needs them.

### WHERE (from §7, restated as storage tiers)
- **Small & precious** (metric rows, configs, manifests, search traces, timing)
  → `paper-NN-results` git repo. Diffable, versioned, pullable from Kaggle+Colab.
- **Large & precious** (checkpoints, reference fields) → Kaggle Datasets,
  versioned.
- **Derived/disposable** (aggregates, figures) → regenerated locally into
  `analysis/`, not committed as source-of-truth.
- **Submission freeze** → Zenodo/OSF bundle for a DOI.
- Google Drive = scratch only.

### Format
- Tabular metric rows → **parquet** (typed, compact, fast) with a **JSON**
  sidecar for the config/provenance blob per run. CSV only for the tiny
  run_matrix queue (needs to be human-hand-editable status tracking).
- One row per (run_id). One file per (search / experiment cell). Never one giant
  file everything appends to concurrently.
