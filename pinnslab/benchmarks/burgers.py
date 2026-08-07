r"""1-D viscous Burgers — the canonical PINN benchmark.

.. math::

    u_t + u u_x = \nu u_{xx},
    \qquad x \in [-1, 1],\ t \in [0, 1],

with :math:`u(x, 0) = -\sin(\pi x)`, :math:`u(\pm 1, t) = 0` and
:math:`\nu = 0.01/\pi`. That viscosity is the standard choice because the
solution steepens into a near-shock at the origin — by :math:`t = 0.5` the
gradient at :math:`x = 0` exceeds 150 — so the benchmark actually discriminates
between sampling strategies instead of being solved by any of them.

Boundary and initial conditions are **soft** here (residual terms, not an output
transform). That is deliberate: the DeepXDE oracle this is validated against
uses soft constraints, and a golden test comparing a hard-constrained run to a
soft-constrained reference would be comparing two different problems.

The reference solution
----------------------
Computed by Cole-Hopf, not shipped as a data file. The transformation
:math:`u = -2\nu \phi_x/\phi` linearises Burgers into the heat equation, giving

.. math::

    u(x, t) = -\frac{\int \sin(\pi(x - \eta))\, f(x - \eta)\,
                       e^{-\eta^2/(4\nu t)}\, d\eta}
                    {\int f(x - \eta)\, e^{-\eta^2/(4\nu t)}\, d\eta},
    \qquad f(y) = e^{-\cos(\pi y)/(2\pi\nu)}.

Three implementation notes, each of which produces silent garbage if skipped:

* **Gauss-Hermite, not general-purpose quadrature.** Substituting
  :math:`\eta = \sqrt{4\nu t}\,z` turns the Gaussian factor into the Hermite
  weight exactly. Adaptive quadrature on the raw integrand misses the peak at
  small :math:`t`, where it is only :math:`\sqrt{2\nu t}` wide.
* **The nodes come from scipy, not numpy.** ``numpy.polynomial.hermite.hermgauss``
  overflows to ``nan`` past roughly 100 nodes; ``scipy.special.roots_hermite``
  uses asymptotics and stays finite.
* **Log-sum-exp, always.** :math:`f` spans :math:`e^{\pm 50}` at the default
  viscosity, so the maximum exponent is factored out before exponentiating.

Validated in ``tests/unit/test_burgers.py`` against an independent
finite-difference solve (agreement to rel-L2 1.3e-4, the FD scheme's own
discretisation error on that front), plus quadrature convergence, the initial
condition and both boundaries. The independent solve is the part that catches a
wrong *formula*; convergence tests only catch a wrong *evaluation* of it.
"""

from __future__ import annotations

import math

import torch
from scipy.special import roots_hermite

from pinnslab.benchmarks.problem import Problem, ResidualTerm, resolve_params
from pinnslab.components import register_problem, register_residual
from pinnslab.geometry import interval, with_time
from pinnslab.physics.diffops import gradient, requires_grad
from pinnslab.registry.config import ProblemSpec, ResidualSpec
from pinnslab.training.trainer import TrainState

NAME = "burgers1d"

#: The literature-standard viscosity. Small enough that the solution develops a
#: steep front; large enough that a reference solution is unambiguous.
DEFAULT_NU = 0.01 / math.pi

DEFAULTS: dict[str, float] = {"nu": DEFAULT_NU}

#: Quadrature nodes for the Cole-Hopf integral. 200 already agrees with 800 to
#: 1.2e-15, so this is comfortably converged rather than tuned.
QUADRATURE_NODES = 400

#: Evaluation grid, ``(n_x, n_t)``. The standard 256x100 of the PINN
#: literature, so rel-L2 here is comparable with published numbers.
EVAL_RESOLUTION = (256, 100)

#: Points per Cole-Hopf batch. Caps the ``(chunk, QUADRATURE_NODES)``
#: intermediates rather than materialising one for the whole grid at once.
CHUNK_SIZE = 4096


def exact_solution(points: torch.Tensor, *, nu: float = DEFAULT_NU) -> torch.Tensor:
    """``(N, 2)`` points ``(x, t)`` -> ``(N, 1)`` exact ``u``.

    Evaluated in float64 regardless of the ambient dtype and cast back at the
    end: this is ground truth, and computing it at the same precision as the
    thing it is judging would fold the judged run's noise floor into the
    yardstick.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"expected (N, 2) points of (x, t), got {tuple(points.shape)}"
        )

    original_dtype = points.dtype
    work = points.detach().to(torch.float64)

    nodes, weights = roots_hermite(QUADRATURE_NODES)
    z = torch.as_tensor(nodes, dtype=torch.float64, device=work.device)
    w = torch.as_tensor(weights, dtype=torch.float64, device=work.device)

    # Chunked because the intermediates are (chunk, nodes): the standard 256x100
    # evaluation grid against 400 nodes is 10.2M float64 per intermediate, and
    # there are several of them live at once.
    out = [_cole_hopf(chunk, z, w, nu) for chunk in work.split(CHUNK_SIZE, dim=0)]
    return torch.cat(out, dim=0).to(original_dtype)


def _cole_hopf(
    points: torch.Tensor, z: torch.Tensor, w: torch.Tensor, nu: float
) -> torch.Tensor:
    x, t = points[:, 0:1], points[:, 1:2]

    # eta = sqrt(4 nu t) z, so the Gaussian factor becomes the Hermite weight.
    y = x - torch.sqrt(4.0 * nu * t) * z  # (N, nodes)
    log_f = -torch.cos(math.pi * y) / (2.0 * math.pi * nu)
    log_f = log_f - log_f.max(dim=1, keepdim=True).values  # spans e^{+-50}

    kernel = torch.exp(log_f) * w
    numerator = (torch.sin(math.pi * y) * kernel).sum(dim=1, keepdim=True)
    denominator = kernel.sum(dim=1, keepdim=True)
    u = -numerator / denominator

    # t = 0 would divide sqrt(0) into a degenerate integral; the limit is just
    # the initial condition, and it is exact rather than approached.
    return torch.where(t == 0.0, -torch.sin(math.pi * x), u)


@register_problem(NAME)
def build_burgers(spec: ProblemSpec) -> Problem:
    params = resolve_params(spec, DEFAULTS, name=NAME)
    nu = params["nu"]
    return Problem(
        name=NAME,
        domain=with_time(interval(-1.0, 1.0), 0.0, 1.0),
        params=params,
        reference=lambda points: exact_solution(points, nu=nu),
        eval_resolution=EVAL_RESOLUTION,
    )


@register_residual(f"{NAME}.pde")
def make_pde(spec: ResidualSpec, problem: Problem) -> ResidualTerm:
    """``u_t + u u_x - nu u_xx`` on the interior."""
    nu = problem.params["nu"]
    net_name = spec.net

    def pde(state: TrainState, points: torch.Tensor) -> torch.Tensor:
        x = requires_grad(points)
        u = state.nets[net_name](x)
        # One gradient call gives u_x and u_t together; u_xx then differentiates
        # the column that is already in the graph. Two backward passes, not
        # three (see diffops.second_partial's note).
        du = gradient(u, x)
        u_x, u_t = du[:, 0:1], du[:, 1:2]
        u_xx = gradient(u_x, x)[:, 0:1]
        return (u_t + u * u_x - nu * u_xx).squeeze(-1)

    return pde


@register_residual(f"{NAME}.ic")
def make_ic(spec: ResidualSpec, problem: Problem) -> ResidualTerm:
    """``u(x, 0) + sin(pi x)`` on the initial slice."""
    net_name = spec.net

    def ic(state: TrainState, points: torch.Tensor) -> torch.Tensor:
        u = state.nets[net_name](points)
        target = -torch.sin(math.pi * points[:, 0:1])
        return (u - target).squeeze(-1)

    return ic


@register_residual(f"{NAME}.bc")
def make_bc(spec: ResidualSpec, problem: Problem) -> ResidualTerm:
    """``u(±1, t)`` — homogeneous Dirichlet on the spatial boundary."""
    net_name = spec.net

    def bc(state: TrainState, points: torch.Tensor) -> torch.Tensor:
        return state.nets[net_name](points).squeeze(-1)

    return bc


__all__ = [
    "DEFAULT_NU",
    "EVAL_RESOLUTION",
    "NAME",
    "QUADRATURE_NODES",
    "build_burgers",
    "exact_solution",
]
