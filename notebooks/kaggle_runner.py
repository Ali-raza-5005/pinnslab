"""The Kaggle runner notebook (DESIGN.md §7, §9 step 3).

Kept as a ``.py`` rather than a ``.ipynb`` so it diffs. Paste the cells below
into a Kaggle notebook; there is nothing else to write, and nothing here should
be edited except the tag, the paths and the deadline.

**Never edit pinnslab on Kaggle.** Edit locally, push, bump a tag, install the
tag. A session that pip-installs from a branch cannot say what it ran, and the
provenance on every result row would be a lie.

The one operational subtlety: `root` must survive the session
--------------------------------------------------------------
Everything else here is bookkeeping, but get this wrong and the resume
machinery never fires — every session starts from nothing and the sweep never
finishes.

* ``/kaggle/working`` is **wiped between sessions**. A `root` under it alone
  makes every session a fresh start.
* ``/kaggle/input`` is **read-only**. A `root` under it cannot be written at
  all.

So the session copies the previous sessions' output out of the mounted Dataset
into the working directory, runs there, and publishes the working directory
back as a new version of that Dataset at the end. Checkpoints are large and
precious, which is exactly what Kaggle Datasets are the tier for
(DESIGN.md §11). The small precious files — metric rows, configs, traces — also
go to the ``paper-NN-results`` git repo, which is what makes "did we run config
X?" answerable in five seconds without mounting anything.

Two GPUs
--------
Do not use DDP: PINN nets are tiny and all-reduce costs more than it saves. Run
the notebook twice, once per GPU, with `CUDA_VISIBLE_DEVICES` and a different
`PINNSLAB_WORKER`. The queue partitions the matrix statically by row index, so
the two sessions never consider the same cell and need no coordination — but
give them **separate** `root` directories, since two sessions cannot publish to
one Dataset version.

Killing a session
-----------------
Nothing to do. Re-run the notebook unchanged: finished cells are skipped, the
interrupted cell resumes from its last checkpoint, and the result is
bit-identical to an uninterrupted sweep (pinned by
``tests/unit/test_queue_survives_a_killed_session.py``).
"""

# ---------------------------------------------------------------- cell 1 ----
# !pip install -q git+https://github.com/Ali-raza-5005/pinnslab@v0.2.0
#
# The tag, never a branch: `git tag -l` in the checkout is the list of what
# exists. A session that installs from a branch cannot say what it ran.

# ---------------------------------------------------------------- cell 2 ----
# Seed the working root from the previous sessions' Dataset (see the docstring).
# `cp -rn` never overwrites, so a rerun of this cell cannot clobber the session.
#
# !mkdir -p /kaggle/working/results
# !cp -rn /kaggle/input/paper-01-checkpoints/results/. /kaggle/working/results/

# ---------------------------------------------------------------- cell 3 ----
import os

from pinnslab.training.queue import load_matrix, run_queue, statuses

MATRIX = "/kaggle/input/paper-01-results/run_matrix.csv"
ROOT = "/kaggle/working/results"

cells = load_matrix(MATRIX)
report = run_queue(
    cells,
    ROOT,
    worker=int(os.environ.get("PINNSLAB_WORKER", 0)),
    workers=int(os.environ.get("PINNSLAB_WORKERS", 1)),
    deadline_seconds=8 * 3600,  # Kaggle's wall clock, minus room to publish
)
print(report)
for cell, status in statuses(cells, ROOT):
    print(f"{status.value:10} {cell.config.name} seed={cell.seed}")

# ---------------------------------------------------------------- cell 4 ----
# Publish. Both halves matter: the Dataset carries the checkpoints that make the
# next session resumable, the git repo carries the rows that make the sweep
# queryable. This stays in the notebook and out of pinnslab on purpose — which
# repo a paper's results go to is paper-level policy, and the library must not
# grow a git dependency to express it (DESIGN.md §7).
#
# !kaggle datasets version -p /kaggle/working/results -m "session $(date -u +%FT%TZ)"
# !cd /kaggle/working/paper-01-results \
#   && rsync -a --exclude checkpoints /kaggle/working/results/ results/ \
#   && git add -A && git commit -m "session $(date -u +%FT%TZ)" && git push
