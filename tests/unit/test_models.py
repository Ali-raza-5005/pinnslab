"""The MLP and the activation registry.

Most of what matters here is reproducibility rather than accuracy: a golden
test's tolerance is only meaningful if two runs at one seed start from
identical weights, and if the initialisation is ours rather than whatever
``nn.Linear`` happens to default to in the installed torch.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pinnslab.components import ACTIVATIONS, MODELS, TRANSFORMS
from pinnslab.models.mlp import MLP, build_net, count_parameters, initialise
from pinnslab.registry.config import NetSpec
from pinnslab.utils.seeding import set_seed

pytestmark = pytest.mark.unit


@pytest.fixture
def spec() -> NetSpec:
    return NetSpec(inputs=2, outputs=1, width=20, depth=3, activation="tanh")


# -- shape and structure ------------------------------------------------------


def test_depth_counts_hidden_layers(spec):
    net = build_net(spec)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 4  # 3 hidden + 1 output
    assert [(m.in_features, m.out_features) for m in linears] == [
        (2, 20),
        (20, 20),
        (20, 20),
        (20, 1),
    ]


def test_the_output_layer_has_no_activation(spec):
    """An activation on the output would bound the solution to that
    activation's range — a silently unsolvable problem for any PDE whose
    solution leaves it."""
    net = build_net(spec)
    assert isinstance(net.net[-1], nn.Linear)


def test_forward_maps_n_by_inputs_to_n_by_outputs(spec):
    net = build_net(spec)
    assert net(torch.zeros(7, 2)).shape == (7, 1)


def test_parameter_count_is_reported(spec):
    # 4 weight matrices + 4 bias vectors
    expected = (2 * 20 + 20) + 2 * (20 * 20 + 20) + (20 * 1 + 1)
    assert count_parameters(build_net(spec)) == expected


# -- reproducibility ----------------------------------------------------------


def test_one_seed_gives_identical_weights(spec):
    """The precondition for every seed-matched comparison in DESIGN.md §8."""
    set_seed(11)
    first = build_net(spec)
    set_seed(11)
    second = build_net(spec)

    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_different_seeds_give_different_weights(spec):
    set_seed(11)
    first = build_net(spec)
    set_seed(12)
    second = build_net(spec)
    assert not torch.equal(
        next(iter(first.parameters())), next(iter(second.parameters()))
    )


def test_initialisation_is_ours_not_torchs_default(spec):
    """Glorot normal with zero biases, matching what DeepXDE starts from.

    If this silently fell back to ``nn.Linear``'s default (Kaiming-uniform
    weights *and* uniform non-zero biases), every golden-test tolerance would
    become a statement about the installed torch version.
    """
    net = build_net(spec)
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            assert torch.all(layer.bias == 0.0)
            fan_in, fan_out = layer.in_features, layer.out_features
            expected_std = (2.0 / (fan_in + fan_out)) ** 0.5
            assert layer.weight.std().item() == pytest.approx(expected_std, rel=0.35)


def test_parameters_take_the_ambient_dtype(spec):
    """Precision comes from the config via ``configure_runtime``, never from
    the model builder (DESIGN.md §5)."""
    torch.set_default_dtype(torch.float64)
    assert next(build_net(spec).parameters()).dtype is torch.float64
    torch.set_default_dtype(torch.float32)
    assert next(build_net(spec).parameters()).dtype is torch.float32


# -- activations --------------------------------------------------------------


def test_the_spec_selects_the_activation(spec):
    net = build_net(spec.model_copy(update={"activation": "sin"}))
    from pinnslab.models.mlp import Sin

    assert any(isinstance(m, Sin) for m in net.modules())


def test_sin_is_differentiable_twice():
    """PINN residuals need second derivatives; an activation that cannot supply
    one produces an identically-zero PDE term rather than an error."""
    from pinnslab.models.mlp import Sin

    x = torch.tensor([0.7], requires_grad=True)
    y = Sin()(x)
    (first,) = torch.autograd.grad(y, x, create_graph=True)
    (second,) = torch.autograd.grad(first, x)
    assert second.item() == pytest.approx(-torch.sin(torch.tensor(0.7)).item())


def test_relu_is_registered_but_kills_second_derivatives():
    """Pinning the trap rather than removing the activation.

    ReLU's second derivative is identically zero, so a residual containing a
    Laplacian evaluates to a constant regardless of the weights: the PDE term
    trains to a perfect score while meaning nothing. Keeping it registered
    makes it available as a documented-bad baseline for the activation-search
    axis; this test is why nobody reaches for it by accident.
    """
    net = build_net(NetSpec(inputs=1, outputs=1, width=8, depth=2, activation="relu"))
    x = torch.rand(16, 1, requires_grad=True)
    y = net(x)
    (first,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), x, create_graph=True)
    assert torch.all(second == 0.0)


def test_an_unknown_activation_lists_the_registered_ones(spec):
    with pytest.raises(KeyError, match="tanh"):
        build_net(spec.model_copy(update={"activation": "tanhh"}))


# -- extension points ---------------------------------------------------------


def test_an_output_transform_is_applied(spec):
    """Hard constraints, DESIGN.md §4 conformance item 5. Registered here rather
    than shipped: which constraint is right is a property of the problem."""
    key = "_test_zero_at_origin"
    TRANSFORMS.register(key)(lambda x, y: y * x[:, :1])
    try:
        net = build_net(spec.model_copy(update={"output_transform": key}))
        x = torch.zeros(4, 2)
        assert torch.all(net(x) == 0.0)
    finally:
        TRANSFORMS._items.pop(key)


def test_a_new_architecture_needs_no_edits_here():
    """CLAUDE.md rule 9: one new file, zero edits to existing files."""
    key = "_test_constant"
    MODELS.register(key)(lambda spec: nn.Linear(spec.inputs, spec.outputs))
    try:
        net = build_net(NetSpec(arch=key, inputs=3, outputs=2))
        assert net(torch.zeros(5, 3)).shape == (5, 2)
    finally:
        MODELS._items.pop(key)


def test_an_unknown_architecture_lists_the_registered_ones():
    with pytest.raises(KeyError, match="mlp"):
        build_net(NetSpec(arch="siren", inputs=2))


def test_registering_a_duplicate_name_is_refused():
    """Two components silently sharing a name would make a config ambiguous and
    a result row a lie about what ran."""
    with pytest.raises(KeyError, match="already registered"):
        ACTIVATIONS.register("tanh")(nn.Tanh)


# -- misuse -------------------------------------------------------------------


def test_options_on_the_plain_mlp_are_rejected(spec):
    """``options`` is forwarded verbatim by architectures that understand it;
    silently dropping a knob the author believes is in effect is the failure
    mode this prevents."""
    with pytest.raises(ValueError, match="takes no options"):
        build_net(spec.model_copy(update={"options": {"omega_0": 30.0}}))


def test_an_unknown_initialiser_is_rejected():
    with pytest.raises(ValueError, match="unknown initialiser"):
        initialise(MLP(
            inputs=1, outputs=1, width=4, depth=1, activation=nn.Tanh()
        ), "orthogonal")
