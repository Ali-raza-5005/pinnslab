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
- DeepXDE is also used as a **golden-test reference**: our from-scratch Burgers
  must match DeepXDE's result within tolerance, catching silent implementation
  bugs.

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
    ├── unit/         # fast, CPU-only, <60s total
    └── golden/       # regression: "Burgers @ config X reaches rel-L2 <= 1.2e-3 in 20k steps"
```

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

```python
from torch.func import functional_call, stack_module_state, vmap
params, buffers = stack_module_state(population)          # [P, ...] pop dim
def fitness(p, b, pts):
    return residual_loss(functional_call(base, (p, b), (pts,)))
losses = vmap(fitness)(params, buffers, collocation_pts)  # all P at once
```
**This is structurally impossible through DeepXDE** (stateful OO Model). It's a
core reason we build from scratch.

Scope caveat: vmap needs shared tensor shapes → clean for sampling, loss
weights, weight-space optimization. Architecture search over varying depth/width
doesn't vectorise directly → group candidates by shape, vmap within groups.
Activation search vmaps fine (index into a fixed set).

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
- **Run queue, not manual notebooks**: `run_matrix.csv` lists every
  (pde, method, seed, hparams) cell with a status column. Notebook claims
  pending rows, runs, writes results, marks done. "Did we run config X?"
  answerable in 5 seconds.

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
  A single `pinnslab.viz.style` module keeps all papers consistent.
- **`results/` is append-only.** Never overwrite. Aggregation reads raw, writes
  derived files to `analysis/`.

---

## 9. Bootstrap order (build infra before science)

1. `utils/seeding`, `registry/` (Run object, config hashing, results schema),
   `training/checkpoint` (save/resume incl. RNG state), `training/trainer`
   (bare Adam→L-BFGS loop). Nothing else.
2. One benchmark end-to-end: 1D Burgers + vanilla MLP. Golden test asserting
   target rel-L2.
3. Kaggle runner: <=20-line notebook, install-from-tag, read run_matrix.csv,
   claim/run/checkpoint/push/mark-done. Prove it survives a killed session.
4. `viz/style.py` + one figure script reading results → publication-ready
   convergence plot. Full loop config→figure, zero manual steps.
5. `search/` layer: SearchSpec + vmap population evaluator + outer-loop
   checkpoint + cache + multi-fidelity. THEN start P0 on paper 1 (sampling).

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
- **inner_train_time** per candidate/run (wall-clock).
- **loop/generation time** for the search outer loop.
- **total_search_time** and **total_experiment_time** (cell-level).
- **time-to-target-accuracy** (steps AND seconds to reach a fixed rel-L2).
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
