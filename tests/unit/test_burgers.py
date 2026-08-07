"""The frozen Burgers benchmark, and above all its reference solution.

The reference is the yardstick every Burgers number in every paper will be
measured against, so it gets checked four ways. Three of them (quadrature
convergence, initial condition, boundaries) only prove the formula was
*evaluated* correctly. The fourth — an independent finite-difference solve —
is the one that can catch a wrong formula, which is the failure that would
otherwise silently invalidate everything downstream.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinnslab.benchmarks.burgers import (
    DEFAULT_NU,
    QUADRATURE_NODES,
    exact_solution,
)
from pinnslab.benchmarks.problem import build_problem, resolve_params
from pinnslab.components import RESIDUALS
from pinnslab.registry.config import ProblemSpec, ResidualSpec

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def float64():
    torch.set_default_dtype(torch.float64)


def grid(nx: int, nt: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)


# -- the reference solution ---------------------------------------------------


def test_the_initial_condition_is_exact():
    x = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
    points = torch.stack([x, torch.zeros_like(x)], dim=1)

    u = exact_solution(points)

    assert torch.allclose(u.squeeze(-1), -torch.sin(math.pi * x), atol=1e-14)


def test_both_boundaries_are_zero():
    t = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
    for side in (-1.0, 1.0):
        points = torch.stack([torch.full_like(t, side), t], dim=1)
        assert torch.max(torch.abs(exact_solution(points))) < 1e-12


def test_the_quadrature_is_converged():
    """200 nodes already agrees with 800 to ~1e-15, so the shipped 400 is
    comfortably converged rather than tuned to a tolerance."""
    from scipy.special import roots_hermite

    import pinnslab.benchmarks.burgers as burgers

    points = grid(129, 7)
    reference = exact_solution(points)

    original = burgers.QUADRATURE_NODES
    try:
        burgers.QUADRATURE_NODES = 200
        coarse = exact_solution(points)
        burgers.QUADRATURE_NODES = 800
        fine = exact_solution(points)
    finally:
        burgers.QUADRATURE_NODES = original

    assert torch.max(torch.abs(coarse - reference)) < 1e-12
    assert torch.max(torch.abs(fine - reference)) < 1e-12
    # And the node source stays finite at the degree we use — numpy's
    # hermgauss overflows to nan past ~100 nodes, which is why scipy's is used.
    _, weights = roots_hermite(QUADRATURE_NODES)
    assert np.all(np.isfinite(weights))


def test_the_reference_matches_an_independent_finite_difference_solve():
    """The only check here that can catch a wrong *formula*.

    An explicit conservative-flux FD solve on a fine grid is a completely
    different method with completely different error behaviour; agreement to
    ~1e-4 in rel-L2 is the FD scheme's own discretisation error across a front
    whose gradient exceeds 150, not slack in the Cole-Hopf evaluation.
    """
    nu, t_end, nx = DEFAULT_NU, 0.5, 2001
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    u = -np.sin(np.pi * x)

    dt = min(0.2 * dx / max(np.abs(u).max(), 1e-12), 0.2 * dx**2 / nu)
    steps = int(np.ceil(t_end / dt))
    dt = t_end / steps
    for _ in range(steps):
        flux = 0.5 * u**2
        du = np.zeros_like(u)
        du[1:-1] = -(flux[2:] - flux[:-2]) / (2 * dx) + nu * (
            u[2:] - 2 * u[1:-1] + u[:-2]
        ) / dx**2
        u = u + dt * du
        u[0] = u[-1] = 0.0

    points = torch.stack(
        [
            torch.as_tensor(x, dtype=torch.float64),
            torch.full((nx,), t_end, dtype=torch.float64),
        ],
        dim=1,
    )
    reference = exact_solution(points).squeeze(-1).numpy()

    rel_l2 = np.linalg.norm(u - reference) / np.linalg.norm(reference)
    assert rel_l2 < 1e-3, f"FD and Cole-Hopf disagree at rel-L2 {rel_l2:.2e}"


def test_the_solution_steepens_into_a_front():
    """Why this benchmark discriminates between methods at all.

    A problem every sampling strategy solves equally well measures nothing;
    the near-shock at the origin is the whole reason nu = 0.01/pi is standard.
    """
    x = torch.linspace(-0.1, 0.1, 401, dtype=torch.float64)
    points = torch.stack([x, torch.full_like(x, 0.5)], dim=1)
    u = exact_solution(points).squeeze(-1)

    gradient = torch.abs(torch.gradient(u, spacing=(x,))[0]).max()
    assert gradient > 100


def test_viscosity_changes_the_solution():
    """``nu`` is a real parameter, not decoration: if options were ignored the
    reference would silently describe a different equation than the residual."""
    points = grid(65, 5)
    assert not torch.allclose(
        exact_solution(points, nu=DEFAULT_NU),
        exact_solution(points, nu=10 * DEFAULT_NU),
    )


def test_the_reference_is_computed_in_float64_from_float32_input():
    """Ground truth must not inherit the noise floor of the run it judges."""
    points = grid(33, 3)
    narrow = exact_solution(points.to(torch.float32))
    wide = exact_solution(points)

    assert narrow.dtype is torch.float32
    assert torch.allclose(narrow, wide.to(torch.float32), atol=1e-6)


def test_malformed_points_are_rejected():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        exact_solution(torch.zeros(4, 3))


# -- the problem object -------------------------------------------------------


def test_the_benchmark_is_registered_and_frozen():
    problem = build_problem(ProblemSpec(name="burgers1d"))

    assert problem.name == "burgers1d"
    assert problem.params["nu"] == pytest.approx(DEFAULT_NU)
    assert problem.domain.dim == 2
    assert problem.domain.lower == (-1.0, 0.0)
    assert problem.domain.upper == (1.0, 1.0)


def test_viscosity_is_configurable():
    problem = build_problem(ProblemSpec(name="burgers1d", options={"nu": 0.05}))
    assert problem.params["nu"] == 0.05


def test_a_misspelled_physical_constant_is_rejected():
    """The worst silent failure available here: the run solves the default
    equation while the recorded config claims otherwise, and every number
    downstream is wrong and self-consistent."""
    with pytest.raises(ValueError, match="no parameter"):
        build_problem(ProblemSpec(name="burgers1d", options={"mu": 0.05}))


def test_resolve_params_fills_defaults():
    assert resolve_params(
        ProblemSpec(name="x"), {"a": 1.0, "b": 2.0}, name="x"
    ) == {"a": 1.0, "b": 2.0}


# -- the residual terms -------------------------------------------------------


class _StubState:
    def __init__(self, net):
        self.nets = {"u": net}


def test_the_exact_solution_actually_satisfies_burgers():
    """Validates the *reference*, by a route with no autograd in it.

    Everything else about the reference checks that a formula was evaluated
    correctly. This checks that it is a solution of the equation the residual
    terms encode — if the two ever drifted apart, a perfectly-trained network
    would score badly and the bug would look like a training problem.

    Evaluated away from the front, where central differences at h=1e-4 can
    still resolve the curvature.
    """
    problem = build_problem(ProblemSpec(name="burgers1d"))
    torch.manual_seed(0)
    points = torch.stack(
        [
            torch.rand(64, dtype=torch.float64) * 1.2 - 0.6,
            torch.rand(64, dtype=torch.float64) * 0.4 + 0.05,
        ],
        dim=1,
    )

    residual = _residual_of_exact_solution(points, problem.params["nu"])

    assert torch.max(torch.abs(residual)) < 1e-4


def test_the_pde_term_computes_the_burgers_operator():
    """The registered term against a closed-form residual.

    ``v(x, t) = sin(pi x) e^{-t}`` is not a solution of Burgers, which is the
    point: its residual is a nonzero closed-form expression, so a sign error, a
    missing viscosity or a swapped x/t coordinate all show up as a mismatch.
    Comparing against zero could not distinguish those from a term that returns
    zero for the wrong reason.
    """
    problem = build_problem(ProblemSpec(name="burgers1d"))
    nu = problem.params["nu"]
    term = RESIDUALS.get("burgers1d.pde")(ResidualSpec(kind="burgers1d.pde"), problem)

    class Analytic(torch.nn.Module):
        def forward(self, pts):
            x, t = pts[:, 0:1], pts[:, 1:2]
            return torch.sin(math.pi * x) * torch.exp(-t)

    torch.manual_seed(0)
    points = torch.rand(64, 2, dtype=torch.float64) * torch.tensor([2.0, 1.0]) - (
        torch.tensor([1.0, 0.0])
    )
    x, t = points[:, 0:1], points[:, 1:2]
    v = torch.sin(math.pi * x) * torch.exp(-t)
    v_t = -v
    v_x = math.pi * torch.cos(math.pi * x) * torch.exp(-t)
    v_xx = -(math.pi**2) * v
    expected = (v_t + v * v_x - nu * v_xx).squeeze(-1)

    actual = term(_StubState(Analytic()), points)

    assert torch.allclose(actual, expected, atol=1e-12)


def test_the_pde_term_is_sensitive_to_viscosity():
    """Guards the wiring between ProblemSpec.options and the residual: a term
    that ignored ``nu`` would pass every test above."""
    term_default = RESIDUALS.get("burgers1d.pde")(
        ResidualSpec(kind="burgers1d.pde"), build_problem(ProblemSpec(name="burgers1d"))
    )
    term_viscous = RESIDUALS.get("burgers1d.pde")(
        ResidualSpec(kind="burgers1d.pde"),
        build_problem(ProblemSpec(name="burgers1d", options={"nu": 1.0})),
    )

    class Analytic(torch.nn.Module):
        def forward(self, pts):
            return torch.sin(math.pi * pts[:, 0:1]) * torch.exp(-pts[:, 1:2])

    points = torch.rand(16, 2, dtype=torch.float64)
    net = Analytic()

    assert not torch.allclose(
        term_default(_StubState(net), points), term_viscous(_StubState(net), points)
    )


def _residual_of_exact_solution(points: torch.Tensor, nu: float) -> torch.Tensor:
    """``u_t + u u_x - nu u_xx`` by central finite differences on the exact u.

    Independent of ``diffops`` on purpose: this checks the *equation* the PDE
    term encodes, and using autograd here would let a shared misunderstanding
    of the coordinate order cancel out.
    """
    h = 1e-4
    x, t = points[:, 0:1], points[:, 1:2]

    def u(xx, tt):
        return exact_solution(torch.cat([xx, tt], dim=1))

    u0 = u(x, t)
    u_x = (u(x + h, t) - u(x - h, t)) / (2 * h)
    u_t = (u(x, t + h) - u(x, t - h)) / (2 * h)
    u_xx = (u(x + h, t) - 2 * u0 + u(x - h, t)) / h**2
    return (u_t + u0 * u_x - nu * u_xx).squeeze(-1)


def test_the_ic_residual_vanishes_on_the_exact_initial_condition():
    problem = build_problem(ProblemSpec(name="burgers1d"))
    term = RESIDUALS.get("burgers1d.ic")(ResidualSpec(kind="burgers1d.ic"), problem)

    x = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).unsqueeze(-1)
    points = torch.cat([x, torch.zeros_like(x)], dim=1)

    class Exact(torch.nn.Module):
        def forward(self, pts):
            return exact_solution(pts)

    residual = term(_StubState(Exact()), points)
    assert torch.max(torch.abs(residual)) < 1e-12


def test_the_bc_residual_vanishes_on_the_exact_boundary():
    problem = build_problem(ProblemSpec(name="burgers1d"))
    term = RESIDUALS.get("burgers1d.bc")(ResidualSpec(kind="burgers1d.bc"), problem)

    t = torch.linspace(0.0, 1.0, 32, dtype=torch.float64).unsqueeze(-1)
    points = torch.cat([torch.ones_like(t), t], dim=1)

    class Exact(torch.nn.Module):
        def forward(self, pts):
            return exact_solution(pts)

    residual = term(_StubState(Exact()), points)
    assert torch.max(torch.abs(residual)) < 1e-12


def test_residuals_are_per_point():
    """CLAUDE.md rule 5, at the point where it is easiest to get wrong."""
    problem = build_problem(ProblemSpec(name="burgers1d"))
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).to(torch.float64)
    points = torch.rand(17, 2, dtype=torch.float64)

    for kind in ("burgers1d.pde", "burgers1d.ic", "burgers1d.bc"):
        term = RESIDUALS.get(kind)(ResidualSpec(kind=kind), problem)
        assert term(_StubState(net), points).shape == (17,)


def test_the_pde_residual_reaches_the_network_weights():
    problem = build_problem(ProblemSpec(name="burgers1d"))
    term = RESIDUALS.get("burgers1d.pde")(ResidualSpec(kind="burgers1d.pde"), problem)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).to(torch.float64)

    points = torch.rand(16, 2, dtype=torch.float64)
    (term(_StubState(net), points) ** 2).mean().backward()

    assert net[0].weight.grad is not None
    assert torch.any(net[0].weight.grad != 0)


def test_the_residual_names_the_network_from_the_spec():
    """Multi-net configs must not depend on a network happening to be "u"."""
    problem = build_problem(ProblemSpec(name="burgers1d"))
    term = RESIDUALS.get("burgers1d.bc")(
        ResidualSpec(kind="burgers1d.bc", net="phi"), problem
    )

    class State:
        nets = {"phi": torch.nn.Linear(2, 1).to(torch.float64)}

    assert term(State(), torch.rand(5, 2, dtype=torch.float64)).shape == (5,)
