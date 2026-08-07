"""Metrics and evaluation grids.

rel-L2 is the number every Burgers result in every paper will be quoted as, so
what matters here is that it is defined once and cannot be silently computed
two different ways.
"""

from __future__ import annotations

import math

import pytest
import torch

from pinnslab.eval.metrics import l2_error, max_error, relative_l2, uniform_grid
from pinnslab.geometry import interval, with_time

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def float64():
    torch.set_default_dtype(torch.float64)


# -- relative L2 --------------------------------------------------------------


def test_a_perfect_prediction_scores_zero():
    reference = torch.tensor([1.0, -2.0, 3.0])
    assert relative_l2(reference.clone(), reference) == 0.0


def test_the_denominator_is_the_reference_norm():
    """Not the error norm, so the number is comparable across problems whose
    solutions differ in magnitude."""
    reference = torch.tensor([3.0, 4.0])  # norm 5
    predicted = torch.tensor([3.0, 0.0])  # error norm 4
    assert relative_l2(predicted, reference) == pytest.approx(4 / 5)


def test_shape_mismatch_is_rejected_rather_than_broadcast():
    """The dangerous case: ``(N,)`` against ``(N, 1)`` broadcasts into an
    ``(N, N)`` error matrix and returns a plausible-looking number instead of
    raising."""
    with pytest.raises(ValueError, match="same points"):
        relative_l2(torch.zeros(4), torch.zeros(5))


def test_a_column_and_a_flat_vector_are_interchangeable():
    """Networks return ``(N, 1)`` and references may be either; both are the
    same N points and must score identically."""
    predicted = torch.tensor([[1.0], [2.0]])
    reference = torch.tensor([1.5, 2.5])
    assert relative_l2(predicted, reference) == relative_l2(
        predicted.reshape(-1), reference
    )


def test_an_identically_zero_reference_is_refused():
    """Relative error against zero is undefined; returning inf or nan would
    propagate a meaningless number into a results row."""
    with pytest.raises(ValueError, match="identically zero"):
        relative_l2(torch.ones(3), torch.zeros(3))


# -- the other two ------------------------------------------------------------


def test_l2_error_is_an_rms():
    predicted = torch.tensor([1.0, 1.0, 1.0, 1.0])
    reference = torch.tensor([0.0, 0.0, 0.0, 2.0])
    assert l2_error(predicted, reference) == pytest.approx(1.0)


def test_max_error_finds_the_worst_point():
    """Reported alongside rel-L2 because they disagree exactly where it matters:
    excellent everywhere except across a front is a fine L2 and a terrible max."""
    predicted = torch.zeros(100)
    reference = torch.zeros(100)
    reference[42] = 7.0
    assert max_error(predicted, reference) == pytest.approx(7.0)


# -- evaluation grids ---------------------------------------------------------


@pytest.fixture
def domain():
    return with_time(interval(-1.0, 1.0), 0.0, 1.0)


def test_the_grid_is_a_tensor_product_over_the_bounding_box(domain):
    grid = uniform_grid(domain, (5, 3))

    assert grid.shape == (15, 2)
    assert grid[:, 0].min() == -1.0 and grid[:, 0].max() == 1.0
    assert grid[:, 1].min() == 0.0 and grid[:, 1].max() == 1.0


def test_the_grid_varies_the_last_coordinate_fastest(domain):
    """``indexing="ij"``: reshaping to ``(nx, nt, 2)`` must recover the axes.

    Every figure and every per-time-slice metric depends on this layout, and
    getting it backwards produces a transposed solution that still has the
    right shape and the right value range.
    """
    grid = uniform_grid(domain, (5, 3)).reshape(5, 3, 2)

    assert torch.all(grid[:, 0, 1] == 0.0)  # first t-slice is t = 0
    assert torch.allclose(grid[0, :, 0], torch.full((3,), -1.0))  # first x-slice


def test_the_grid_is_deterministic(domain):
    """Not drawn from the sampler: a metric on random points would move between
    runs, and comparing two methods would compare two point clouds as much as
    two solutions."""
    assert torch.equal(uniform_grid(domain, (7, 4)), uniform_grid(domain, (7, 4)))


def test_the_grid_takes_the_requested_dtype(domain):
    assert uniform_grid(domain, (4, 4), dtype=torch.float32).dtype is torch.float32


def test_a_resolution_of_the_wrong_rank_is_rejected(domain):
    with pytest.raises(ValueError, match="2 coordinates"):
        uniform_grid(domain, (10,))


def test_a_degenerate_axis_is_rejected(domain):
    """``linspace`` with one point silently returns the lower bound, which would
    evaluate the whole run at t = 0."""
    with pytest.raises(ValueError, match="at least 2 points"):
        uniform_grid(domain, (10, 1))


def test_the_grid_matches_a_hand_built_meshgrid(domain):
    """Pins the convention against an independent construction."""
    x = torch.linspace(-1.0, 1.0, 4)
    t = torch.linspace(0.0, 1.0, 3)
    expected = torch.stack(
        [c.reshape(-1) for c in torch.meshgrid(x, t, indexing="ij")], dim=1
    )
    assert torch.equal(uniform_grid(domain, (4, 3)), expected)


def test_relative_l2_on_a_known_function(domain):
    """End-to-end sanity: a constant offset from a known field."""
    grid = uniform_grid(domain, (33, 17))
    truth = torch.sin(math.pi * grid[:, 0:1])
    predicted = truth + 0.1

    expected = float(
        torch.linalg.vector_norm(torch.full_like(truth, 0.1))
        / torch.linalg.vector_norm(truth)
    )
    assert relative_l2(predicted, truth) == pytest.approx(expected)
