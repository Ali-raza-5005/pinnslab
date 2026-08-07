"""Differential operators, checked against closed-form derivatives.

A wrong diffop is the most dangerous class of bug in this library: it does not
raise, it does not diverge, it just trains to a confident number for the wrong
equation. Every operator here is therefore checked against an analytic
derivative rather than against itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from pinnslab.physics.diffops import (
    gradient,
    laplacian,
    partial,
    requires_grad,
    second_partial,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def float64():
    """These are exactness tests; float32 noise would force useless tolerances."""
    torch.set_default_dtype(torch.float64)


@pytest.fixture
def points() -> torch.Tensor:
    torch.manual_seed(0)
    return requires_grad(torch.rand(16, 2) * 2 - 1)


# -- first derivatives --------------------------------------------------------


def test_gradient_matches_a_closed_form(points):
    """``f(x, t) = sin(pi x) exp(t)`` -> ``[pi cos(pi x) exp(t), sin(pi x) exp(t)]``."""
    x, t = points[:, 0:1], points[:, 1:2]
    f = torch.sin(math.pi * x) * torch.exp(t)

    grad = gradient(f, points)

    expected_x = math.pi * torch.cos(math.pi * x) * torch.exp(t)
    expected_t = torch.sin(math.pi * x) * torch.exp(t)
    assert torch.allclose(grad[:, 0:1], expected_x)
    assert torch.allclose(grad[:, 1:2], expected_t)


def test_gradient_is_per_point_not_summed(points):
    """The ``grad_outputs=ones`` trick sums over the batch before
    differentiating; that is only exact because rows are independent. A
    coupling between rows would show up here as a constant offset."""
    x = points[:, 0:1]
    f = x**2

    grad = gradient(f, points)

    assert torch.allclose(grad[:, 0:1], 2 * x)
    assert grad.shape == points.shape


def test_component_selects_the_output_field(points):
    """Coupled systems (DESIGN.md §4 conformance item 7) have vector outputs."""
    x, t = points[:, 0:1], points[:, 1:2]
    y = torch.cat([x**2, t**3], dim=1)

    assert torch.allclose(gradient(y, points, component=0)[:, 0:1], 2 * x)
    assert torch.allclose(gradient(y, points, component=1)[:, 1:2], 3 * t**2)


def test_partial_returns_a_column(points):
    x = points[:, 0:1]
    assert partial(x**3, points, wrt=0).shape == (16, 1)
    assert torch.allclose(partial(x**3, points, wrt=0), 3 * x**2)


# -- second derivatives -------------------------------------------------------


def test_second_partial_matches_a_closed_form(points):
    x = points[:, 0:1]
    f = torch.sin(math.pi * x)

    assert torch.allclose(
        second_partial(f, points, wrt=0), -(math.pi**2) * torch.sin(math.pi * x)
    )


def test_the_documented_cheap_route_agrees_with_second_partial(points):
    """The docstring tells callers to reuse one gradient for u_x and u_xx.

    That advice is only safe if it gives the same answer, so pin it — a residual
    written the cheap way and a test written the expensive way silently
    disagreeing would be very hard to find.
    """
    x = points[:, 0:1]
    f = torch.exp(x) * torch.sin(math.pi * x)

    du = gradient(f, points)
    cheap = gradient(du[:, 0:1], points)[:, 0:1]
    expensive = second_partial(f, points, wrt=0)

    assert torch.allclose(cheap, expensive)


def test_laplacian_sums_only_the_requested_dimensions(points):
    """Time is not part of a Laplacian, and the default includes every
    coordinate — which on a space-time domain is the wrong operator."""
    x, t = points[:, 0:1], points[:, 1:2]
    f = x**2 + t**4

    spatial = laplacian(f, points, dims=(0,))
    everything = laplacian(f, points)

    assert torch.allclose(spatial, torch.full_like(spatial, 2.0))
    assert torch.allclose(everything, 2.0 + 12 * t**2)


def test_a_third_derivative_is_reachable(points):
    """KdV needs u_xxx; nothing may cap the differentiation order at two."""
    x = points[:, 0:1]
    f = x**4

    first = gradient(f, points)[:, 0:1]
    second = gradient(first, points)[:, 0:1]
    third = gradient(second, points)[:, 0:1]

    assert torch.allclose(third, 24 * x)


# -- the graph must survive far enough to train -------------------------------


def test_derivatives_stay_differentiable_by_default(points):
    """A PINN optimizes *through* the residual: the loss backward pass has to
    reach the network weights via the second derivative."""
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(),
                              torch.nn.Linear(8, 1))
    u = net(points)
    u_xx = second_partial(u, points, wrt=0)

    loss = (u_xx**2).mean()
    loss.backward()

    hidden, output = net[0], net[2]
    for param in (hidden.weight, hidden.bias, output.weight):
        assert param.grad is not None and torch.any(param.grad != 0)


def test_a_pde_term_alone_leaves_the_output_bias_ungradiented(points):
    """Not a bug, and worth pinning so it is never "fixed".

    A second derivative annihilates the additive output bias, so ``u_xx`` does
    not depend on it and its gradient is ``None`` after a PDE-only backward
    pass. In a real run the IC/BC terms supply that gradient — which is also
    why a PINN trained on the PDE residual alone is not merely under-determined
    by a constant, it never learns one at all.
    """
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(),
                              torch.nn.Linear(8, 1))
    u_xx = second_partial(net(points), points, wrt=0)
    (u_xx**2).mean().backward()

    assert net[2].bias.grad is None


def test_create_graph_false_detaches(points):
    x = points[:, 0:1]
    assert not gradient(x**2, points, create_graph=False).requires_grad


# -- misuse is caught, not silently wrong -------------------------------------


def test_points_without_requires_grad_are_rejected():
    """The failure this replaces is a torch error far from the cause; the
    fix — flagging the points before the forward pass — is not guessable
    from it."""
    plain = torch.rand(8, 2)
    with pytest.raises(ValueError, match="before.*forward pass"):
        gradient(plain**2, plain)


def test_pre_reduced_outputs_are_rejected(points):
    """Catching CLAUDE.md rule 5's failure mode one layer earlier."""
    with pytest.raises(ValueError, match=r"must be \(N, m\)"):
        gradient((points**2).sum(), points)


def test_requires_grad_leaves_a_ready_leaf_alone():
    leaf = torch.rand(4, 2, requires_grad=True)
    assert requires_grad(leaf) is leaf


def test_requires_grad_detaches_a_non_leaf():
    """Points produced by an adaptive sampler that itself differentiates arrive
    already attached to a graph; the residual wants derivatives with respect to
    the coordinates, not with respect to whatever generated them."""
    source = torch.rand(4, 2, requires_grad=True)
    derived = source * 2

    ready = requires_grad(derived)

    assert ready.is_leaf and ready.requires_grad
    assert torch.equal(ready, derived.detach())


def test_requires_grad_flags_a_plain_tensor():
    ready = requires_grad(torch.rand(4, 2))
    assert ready.requires_grad and ready.is_leaf
