"""The DeepXDE seam (DESIGN.md §1).

Two kinds of test here. The first is structural: rule 1 says deepxde is imported
in exactly one file, and a rule nobody can violate by accident is worth more
than a rule everybody remembers. The rest pin the three DeepXDE behaviours the
adapter exists to neutralise — ambient backend, numpy-global RNG, float32
default — because each of them is silent when it goes wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import pinnslab
from pinnslab.geometry import Domain, interval, with_time
from pinnslab.geometry.adapters import DETERMINISTIC_STRATEGIES
from pinnslab.utils.seeding import make_generator

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(pinnslab.__file__).parent
ADAPTER = PACKAGE_ROOT / "geometry" / "adapters.py"


@pytest.fixture
def domain() -> Domain:
    """The Burgers domain: x in [-1, 1], t in [0, 1]."""
    return with_time(interval(-1.0, 1.0), 0.0, 1.0)


# -- rule 1, enforced rather than remembered ----------------------------------


def test_deepxde_is_imported_in_exactly_one_file():
    """CLAUDE.md rule 1. The value of the rule is that rewriting one file is the
    whole cost of dropping DeepXDE; a second import site silently doubles it."""
    pattern = re.compile(r"^\s*(?:import|from)\s+deepxde\b", re.MULTILINE)
    offenders = [
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if path != ADAPTER and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"deepxde is imported outside geometry/adapters.py: {offenders}. "
        "See DESIGN.md §1 — no deepxde object may escape that module."
    )


def test_no_deepxde_object_escapes_the_adapter(domain):
    """Everything the seam hands back is a plain torch tensor."""
    generator = make_generator(0)
    for region in ("interior", "boundary", "initial"):
        points = domain.sample(region, 8, generator=generator)
        assert type(points) is torch.Tensor

    lo, hi = domain.bounds()
    assert type(lo) is torch.Tensor and type(hi) is torch.Tensor


# -- the backend is ambient, and its default is wrong here --------------------


@pytest.mark.slow
def test_the_pytorch_backend_is_selected_without_help_from_the_environment():
    """A fresh interpreter with no DDE_BACKEND set must still land on pytorch.

    Run out-of-process because the backend is chosen once, at first import, and
    this test suite has long since made that choice. On a machine with
    TensorFlow installed the unguarded import picks TensorFlow and then dies on
    a missing tensorflow_probability, so this is the difference between the
    package importing and not importing at all.
    """
    import subprocess
    import sys

    env = {k: v for k, v in __import__("os").environ.items() if k != "DDE_BACKEND"}
    result = subprocess.run(
        [sys.executable, "-c", "import pinnslab.geometry.adapters as a;"
         " import deepxde; print(deepxde.backend.backend_name)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=PACKAGE_ROOT.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "pytorch" in result.stdout


# -- deepxde samples from numpy's global RNG ----------------------------------


def test_sampling_is_reproducible_from_the_generator_alone(domain):
    """The property resume depends on: same generator state -> same points."""
    a = domain.sample("interior", 16, generator=make_generator(3))
    b = domain.sample("interior", 16, generator=make_generator(3))
    assert torch.equal(a, b)


def test_successive_draws_from_one_generator_differ(domain):
    """Resampling must actually move the points."""
    generator = make_generator(3)
    a = domain.sample("interior", 16, generator=generator)
    b = domain.sample("interior", 16, generator=generator)
    assert not torch.equal(a, b)


def test_sampling_does_not_depend_on_unrelated_numpy_draws(domain):
    """The whole reason the adapter seeds numpy itself.

    Without the isolation, an unrelated ``np.random`` call anywhere in the
    process — in a metric, a baseline, a notebook cell — would shift every
    subsequent point cloud, and the run would still look perfectly healthy.
    """
    expected = domain.sample("interior", 16, generator=make_generator(5))

    np.random.random(37)
    actual = domain.sample("interior", 16, generator=make_generator(5))

    assert torch.equal(expected, actual)


def test_sampling_leaves_numpys_global_state_untouched(domain):
    """Isolation runs both ways: seeding numpy for our own use must not become
    a hidden reseed of everyone else's stream."""
    np.random.seed(1234)
    before = np.random.random(4)

    np.random.seed(1234)
    domain.sample("interior", 16, generator=make_generator(7))
    after = np.random.random(4)

    assert np.array_equal(before, after)


def test_quasirandom_strategies_repeat_themselves(domain):
    """Pinning the trap named in DETERMINISTIC_STRATEGIES.

    Halton/Hammersley/Sobol ignore the RNG entirely, so resampling under one of
    them returns the same cloud forever — ``resample_every`` becomes a no-op
    that changes nothing and reports nothing.
    """
    pytest.importorskip("skopt", reason="deepxde's quasirandom samplers need skopt")
    generator = make_generator(11)
    for strategy in sorted(DETERMINISTIC_STRATEGIES):
        a = domain.sample("interior", 8, generator=generator, strategy=strategy)
        b = domain.sample("interior", 8, generator=generator, strategy=strategy)
        assert torch.equal(a, b), f"{strategy} unexpectedly varies between draws"


# -- deepxde's default float is float32 ---------------------------------------


def test_points_are_generated_at_float64_and_cast_down(domain):
    """A float32 run must still get points computed in float64.

    DeepXDE's own default is float32; sampling there and widening afterwards
    would bake float32-resolution coordinates into every float64 run, which is
    precisely the ~1e-4..1e-5 noise floor DESIGN.md §5 chose float64 to avoid.
    """
    wide = domain.sample(
        "interior", 64, generator=make_generator(2), dtype=torch.float64
    )
    narrow = domain.sample(
        "interior", 64, generator=make_generator(2), dtype=torch.float32
    )
    assert wide.dtype == torch.float64
    assert narrow.dtype == torch.float32
    # Same points, one merely stored narrower: if generation had happened at
    # float32 the wide tensor would hold float32 values widened, and this would
    # be an exact equality rather than a rounding.
    assert torch.equal(narrow, wide.to(torch.float32))
    assert not torch.equal(wide, wide.to(torch.float32).to(torch.float64))


def test_importing_the_adapter_does_not_change_torchs_default_dtype():
    """``dde.config.set_default_float`` calls ``torch.set_default_dtype``.

    Letting that stand would make precision an ambient property of import order
    rather than a hashed config field (DESIGN.md §5), and would break the
    no-side-effects promise in ``pinnslab/__init__``.
    """
    import importlib

    torch.set_default_dtype(torch.float32)
    importlib.reload(importlib.import_module("pinnslab.geometry.adapters"))
    assert torch.get_default_dtype() is torch.float32


def test_sample_defaults_to_the_ambient_default_dtype(domain):
    """Precision comes from ``configure_runtime``, not from this module."""
    torch.set_default_dtype(torch.float32)
    assert domain.sample("interior", 4, generator=make_generator(0)).dtype is (
        torch.float32
    )


# -- shapes, regions, bounds --------------------------------------------------


@pytest.mark.parametrize("region", ["interior", "boundary", "initial"])
def test_every_region_returns_n_by_dim(domain, region):
    points = domain.sample(region, 12, generator=make_generator(0))
    assert points.shape == (12, 2)


def test_initial_points_sit_exactly_on_t0(domain):
    points = domain.sample("initial", 32, generator=make_generator(0))
    assert torch.all(points[:, 1] == 0.0)


def test_boundary_points_sit_exactly_on_the_spatial_boundary(domain):
    points = domain.sample("boundary", 32, generator=make_generator(0))
    assert torch.all(points[:, 0].abs() == 1.0)


def test_interior_points_lie_within_the_domain(domain):
    points = domain.sample("interior", 256, generator=make_generator(0))
    lo, hi = domain.bounds()
    assert torch.all(points >= lo) and torch.all(points <= hi)


def test_bounds_report_the_product_domain(domain):
    lo, hi = domain.bounds()
    assert torch.equal(lo, torch.tensor([-1.0, 0.0]))
    assert torch.equal(hi, torch.tensor([1.0, 1.0]))


def test_time_is_the_last_coordinate(domain):
    assert domain.dim == 2
    assert domain.time_dependent


# -- misuse is rejected at the seam -------------------------------------------


def test_initial_region_on_a_timeless_domain_is_rejected():
    with pytest.raises(ValueError, match="time-dependent"):
        interval(-1.0, 1.0).sample("initial", 4, generator=make_generator(0))


def test_an_unknown_strategy_names_the_available_ones():
    domain = interval(-1.0, 1.0)
    with pytest.raises(ValueError, match="pseudo"):
        domain.sample("interior", 4, generator=make_generator(0), strategy="sobal")


def test_an_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="unknown region"):
        interval(-1.0, 1.0).sample("edge", 4, generator=make_generator(0))


def test_a_nonpositive_count_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        interval(-1.0, 1.0).sample("interior", 0, generator=make_generator(0))


def test_a_degenerate_interval_is_rejected():
    with pytest.raises(ValueError, match="upper > lower"):
        interval(1.0, 1.0)


def test_a_degenerate_time_domain_is_rejected():
    with pytest.raises(ValueError, match="t1 > t0"):
        with_time(interval(-1.0, 1.0), 1.0, 1.0)


def test_time_cannot_be_added_twice(domain):
    with pytest.raises(ValueError, match="already carries a time coordinate"):
        with_time(domain, 0.0, 1.0)
