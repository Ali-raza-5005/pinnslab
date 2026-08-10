"""The metaheuristic search layer — the research program's heart (DESIGN.md §6).

The four research directions (sampling / weighting / optimizer / architecture)
are the same code with a different ``RunConfig`` field selected. The machinery
is written once here; each paper is a new search space and a fitness function.

Layout:

* :mod:`~pinnslab.search.space` — config paths -> domains, and the unit-cube
  encoding every algorithm shares.
* :mod:`~pinnslab.search.spec` — the validated, hashed ``search.yaml``.
* :mod:`~pinnslab.search.algorithms` — ``@register_search``; ``random`` and
  ``de`` ship, the rest are one file each.
* :mod:`~pinnslab.search.population` — the batched population evaluator, and
  the measured reason it is not ``vmap``.
* :mod:`~pinnslab.search.cache` — ``(config_hash, steps) -> fitness``.
* :mod:`~pinnslab.search.state` — outer-loop checkpoint, including the
  metaheuristic's RNG.
* :mod:`~pinnslab.search.loop` — the driver.

Nothing is re-exported: :mod:`~pinnslab.search.population` and
:mod:`~pinnslab.search.loop` pull torch and the benchmark stack, for the same
reason ``training/__init__`` does not re-export ``build`` or ``queue``. Import
the module you need.
"""
