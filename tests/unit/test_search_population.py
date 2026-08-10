"""The batched population evaluator.

The load-bearing claim is that training P candidates in one graph is the same
experiment as training them one at a time. If it is not, the whole search layer
reports numbers that no single-run reproduction will match — and the failure
would be quiet, because a coupled population still converges to something
plausible. So the tests here are equivalence tests against the sequential path,
not smoke tests.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pinnslab.search.population import Ensemble, train_population

pytestmark = pytest.mark.unit

#: Measured tolerance, not a guess. Batched matmuls (baddbmm) reduce in a
#: different order from the per-candidate ones (addmm), so agreement is to
#: machine epsilon rather than bit-exact; 25 Adam steps in float64 drift by
#: ~1e-16. A regression that actually coupled the population would blow
#: through this by many orders of magnitude.
TOLERANCE = 1e-12


def make_net(width: int = 12, depth: int = 2, inputs: int = 2) -> nn.Module:
    layers: list[nn.Module] = []
    size = inputs
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.Tanh()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, 1))


@pytest.fixture(autouse=True)
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def burgers_like(ensemble, points: torch.Tensor) -> torch.Tensor:
    """A real second-order residual: u_t + u u_x - nu u_xx, over (P, N, 2)."""
    outputs = ensemble(points)
    (grad,) = torch.autograd.grad(
        outputs, points, torch.ones_like(outputs), create_graph=True
    )
    u_x, u_t = grad[..., 0:1], grad[..., 1:2]
    (grad_x,) = torch.autograd.grad(
        u_x, points, torch.ones_like(u_x), create_graph=True
    )
    return (u_t + outputs * u_x - 0.01 * grad_x[..., 0:1]).squeeze(-1)


def burgers_single(net: nn.Module, points: torch.Tensor) -> torch.Tensor:
    """The same residual on one candidate, (N, 2) -> (N,)."""
    outputs = net(points)
    (grad,) = torch.autograd.grad(
        outputs, points, torch.ones_like(outputs), create_graph=True
    )
    u_x, u_t = grad[:, 0:1], grad[:, 1:2]
    (grad_x,) = torch.autograd.grad(
        u_x, points, torch.ones_like(u_x), create_graph=True
    )
    return (u_t + outputs * u_x - 0.01 * grad_x[:, 0:1]).squeeze(-1)


# -- the ensemble --------------------------------------------------------------


def test_the_ensemble_reproduces_each_member():
    """Before anything is trained: the batched forward must *be* the members.

    To machine epsilon, not bit-exactly — ``baddbmm`` reduces in a different
    order from ``addmm``, so the difference is ~1e-17 in float64. That is the
    honest bound and the tolerance is set at it, five orders above the observed
    difference and many below anything a real coupling bug would produce.
    """
    nets = [make_net() for _ in range(4)]
    points = torch.rand(4, 32, 2)

    batched = Ensemble(nets)(points)

    for index, net in enumerate(nets):
        assert torch.allclose(batched[index], net(points[index]), atol=TOLERANCE)


def test_members_must_share_a_shape():
    """The binding constraint, and the one DESIGN.md §6 flagged for vmap too:
    architecture search does not batch and must group by shape first."""
    with pytest.raises(ValueError, match="same shape"):
        Ensemble([make_net(width=8), make_net(width=16)])


def test_a_member_can_be_extracted_back_into_a_plain_mlp():
    """A winning candidate has to leave the search and enter the ordinary
    single-run path — retrained, checkpointed, plotted."""
    nets = [make_net() for _ in range(3)]
    ensemble = Ensemble(nets)
    points = torch.rand(3, 16, 2)
    before = ensemble(points)

    extracted = make_net()
    extracted.load_state_dict(ensemble.member_state_dict(1))

    assert torch.allclose(extracted(points[1]), before[1], atol=TOLERANCE)


def test_a_wrong_population_size_is_rejected():
    ensemble = Ensemble([make_net() for _ in range(3)])
    with pytest.raises(ValueError, match=r"\(3, N, d\)"):
        ensemble(torch.rand(2, 16, 2))


# -- the equivalence that everything rests on ----------------------------------


def test_batched_training_equals_independent_training():
    """THE test of this module.

    Each candidate has its own collocation cloud, as in a sampling search, and
    a genuine second-order PDE residual. If any cross-candidate coupling
    existed — in the forward, the shared autograd graph, or the single Adam
    over stacked parameters — these numbers would diverge.
    """
    size, steps, lr = 5, 25, 1e-2
    nets = [make_net() for _ in range(size)]
    separate = [make_net() for _ in range(size)]
    for source, target in zip(nets, separate, strict=True):
        target.load_state_dict(source.state_dict())

    points = torch.rand(size, 48, 2)

    result = train_population(
        Ensemble(nets), points, burgers_like, steps=steps, lr=lr
    )

    for index, net in enumerate(separate):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x = points[index].detach().requires_grad_(True)
            (burgers_single(net, x) ** 2).mean().backward()
            optimizer.step()

        x = points[index].detach().requires_grad_(True)
        expected = float((burgers_single(net, x) ** 2).mean().detach())
        assert float(result.losses[index]) == pytest.approx(expected, abs=TOLERANCE)


def test_one_candidate_diverging_does_not_move_the_others():
    """The independence property stated as the failure it prevents. A search
    whose population is coupled would let one bad candidate drag the rest, and
    the resulting figure would look entirely reasonable."""
    size, steps = 4, 15
    nets = [make_net() for _ in range(size)]
    clones = [make_net() for _ in range(size)]
    for source, target in zip(nets, clones, strict=True):
        target.load_state_dict(source.state_dict())

    points = torch.rand(size, 32, 2)
    healthy = train_population(
        Ensemble(nets), points, burgers_like, steps=steps, lr=1e-2
    )

    # Candidate 0 now trains at a learning rate that blows it up. Achieved
    # through its residual so the others' inputs are untouched.
    def poisoned(ensemble, pts):
        residuals = burgers_like(ensemble, pts)
        scale = torch.ones(size, 1, dtype=residuals.dtype)
        scale[0] = 1e8
        return residuals * scale

    sick = train_population(Ensemble(clones), points, poisoned, steps=steps, lr=1e-2)

    assert sick.losses[0] != healthy.losses[0], "the poison did nothing"
    for index in range(1, size):
        assert sick.losses[index] == pytest.approx(
            float(healthy.losses[index]), abs=TOLERANCE
        ), f"candidate {index} moved because candidate 0 diverged"


def test_global_gradient_clipping_is_refused():
    """It computes one norm across the whole stacked population, so a single
    diverging candidate damps everyone's step. Silent coupling, so it raises."""
    with pytest.raises(ValueError, match="couples the population"):
        train_population(
            Ensemble([make_net() for _ in range(3)]),
            torch.rand(3, 8, 2),
            burgers_like,
            steps=1,
            max_grad_norm=1.0,
        )


def test_a_residual_of_the_wrong_shape_is_rejected():
    """CLAUDE.md rule 5 extended to the population: (P, N), never a scalar."""
    with pytest.raises(ValueError, match=r"\(P, N\)"):
        train_population(
            Ensemble([make_net() for _ in range(3)]),
            torch.rand(3, 8, 2),
            lambda ensemble, points: ensemble(points).mean(),
            steps=1,
        )


def test_each_candidate_keeps_its_own_collocation_points():
    """The property the sampling search depends on: candidate p is trained on
    points[p] and on nothing else."""
    size, steps = 3, 20
    nets = [make_net() for _ in range(size)]
    for net in nets[1:]:
        net.load_state_dict(nets[0].state_dict())

    # Identical networks, deliberately different point clouds.
    points = torch.stack(
        [torch.rand(24, 2), torch.rand(24, 2) * 0.1, torch.rand(24, 2)]
    )
    losses = train_population(
        Ensemble(nets), points, burgers_like, steps=steps, lr=1e-2
    ).losses

    assert losses[1] != losses[0], "identical nets on different points scored alike"


def test_resampling_is_applied_every_step():
    """`resample_every` at population scale: the hook must actually be called,
    since a search over sampling strategies is exactly what this is for."""
    calls = []

    def resample(step: int) -> torch.Tensor:
        calls.append(step)
        return torch.rand(2, 8, 2)

    train_population(
        Ensemble([make_net() for _ in range(2)]),
        torch.rand(2, 8, 2),
        burgers_like,
        steps=4,
        resample=resample,
    )
    assert calls == [0, 1, 2, 3]
