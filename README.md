# pinnslab

Personal PINN **methods** research infrastructure: we own the PyTorch training
loop, because the research novelty lives inside it. DeepXDE is a thin dependency
(geometry + baseline oracle), never a foundation.

Not a general-purpose library, not intended for external adoption, no API
stability promises. See `DESIGN.md` for why it is shaped this way and
`CLAUDE.md` for the standing rules.

## Install

Papers pin an exact tag — never a branch:

```bash
pip install "pinnslab @ git+https://github.com/<user>/pinnslab@v0.1.0"
```

Local development:

```bash
pip install -e ".[dev]"
pytest tests/unit -q
```

## Status

Bootstrap, DESIGN.md §9 step 1 only: `utils/seeding`, `registry/`,
`training/checkpoint`, `training/trainer`. Everything else in the package tree is
an empty namespace awaiting its turn in the build order.
