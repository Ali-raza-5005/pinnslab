"""Plot primitives and the single house-style module shared by all papers.

Figures are never hand-made (DESIGN.md §8): every one is the output of a script
reading ``results/``, and :mod:`pinnslab.viz.style` is the only place that says
what one looks like.

Nothing is re-exported here. ``import pinnslab.viz.style`` imports matplotlib,
which is neither free nor wanted inside a training session — the same reason
``training/__init__`` does not re-export ``build`` or ``queue``. Import the
module you need:

    from pinnslab.viz import style
    from pinnslab.viz.convergence import write_convergence_figure

Layout:

* :mod:`~pinnslab.viz.style` — rcParams, the validated palette, colormaps, save.
* :mod:`~pinnslab.viz.aggregate` — ``results/`` -> median/IQR/failure rate,
  with the hardware-uniformity check of DESIGN.md §5.
* :mod:`~pinnslab.viz.convergence` — the error-vs-budget figure.
* :mod:`~pinnslab.viz.tables` — the same numbers as `booktabs` LaTeX.
"""
