# FRICTION.md

Evidence base for **where the abstractions are wrong**.

One line every time a paper-level task forces an edit to `pinnslab` core. Do not
skip entries, do not tidy them, do not invent them — the value of this file is
that it is an honest, unfiltered record. Repeated entries pointing at the same
seam are the signal to refactor that seam; a quiet file means the seams are
holding.

Format:

```
YYYY-MM-DD | paper-NN | what was wanted | what had to change in core
```

## Entries

The first three come from building `examples/` — a dry run of paper 1's actual
task (uniform vs residual-adaptive sampling on Burgers, 5 seeds, figures, a
search over the method's knobs) done against the library from outside it. That
is what a paper does, so the friction it produced belongs here, and all three
point at the same seam: **sampling was the axis the library could not extend.**

```
2026-08-17 | paper-01 (dry run) | register an adaptive sampler from a paper repo and name it in a config | SAMPLERS existed but nothing read it — build.py passed `strategy:` straight to DeepXDE. Added geometry/samplers.py; the resampler now builds through the registry.
2026-08-17 | paper-01 (dry run) | resume a resampling run after a killed session | the cloud was in `state.scratch`, which is explicitly not checkpointed. Promoted it to TrainState.points; CheckpointPayload gained points + sampler_state (format 1 -> 2); removed run_queue's refusal of resample_every.
2026-08-17 | paper-01 (dry run) | draw the per-second convergence figure from the sweep | viz.aggregate.band intersected exact wall-clock timestamps across seeds, which are never equal; the band collapsed to t=0 and savefig raised. Interpolation onto a common time grid.
2026-08-17 | paper-01 (dry run) | run the documented commands from a clone | scripts/ could not import pinnslab (Python puts the script's directory on sys.path, not the cwd). Added scripts/_bootstrap.py and --register (utils/plugins.py).
```

Read together: three of the four are *seams that were declared but never
exercised* — a registry with no lookup, a `resample_every` field the queue
refused, a figure axis whose test used data that cannot occur. The lesson for
the next abstraction is not that these were badly designed; it is that an
extension point nobody has extended from outside is not yet an extension point.
