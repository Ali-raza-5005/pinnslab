"""The plain MLP, and the activations it can be built from.

A *volatile* axis (DESIGN.md §4): this file is meant to stay flat and
near-copy-pasteable rather than grow into a hierarchy. Modified-MLP, Fourier
features and SIREN each arrive as their own ``@register_model`` file when a
paper needs them, and none of them will edit this one.

Two choices here are load-bearing for reproducibility rather than for accuracy:

* **Initialisation is explicit**, not torch's ``nn.Linear`` default. Glorot
  (Xavier) normal with zero biases is what the PINN literature and DeepXDE both
  use, so a from-scratch run and the DeepXDE oracle it is checked against start
  from the same distribution. Leaving torch's default in place would make every
  golden-test tolerance a statement about torch's init, which changes between
  versions.
* **Parameters are allocated in the ambient default dtype and device**, which
  ``configure_runtime`` has already set from the config. The builder never picks
  a precision of its own (DESIGN.md §5).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from pinnslab.components import ACTIVATIONS, MODELS, TRANSFORMS, register_model
from pinnslab.registry.config import NetSpec

#: A hard constraint: ``(inputs, outputs) -> outputs``. It takes the coordinates
#: as well as the prediction because it has to know which boundary it is
#: enforcing.
OutputTransform = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Sin(nn.Module):
    """``sin`` as a module. Not in torch, and needed by SIREN and by the
    activation-search axis of DESIGN.md §6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


for _name, _factory in {
    "tanh": nn.Tanh,
    "sin": Sin,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
    # ReLU is here to be *ruled out*, not used: its second derivative is
    # identically zero, so any residual with a Laplacian in it is exactly zero
    # everywhere and the PDE term trains to a perfect, meaningless score.
    "relu": nn.ReLU,
}.items():
    ACTIVATIONS.register(_name)(_factory)


#: Weight initialisers, keyed by the ``NetSpec.init`` value. Not a registry:
#: initialisation is not one of the research axes of DESIGN.md §6, and a paper
#: that needs an exotic one registers a whole model instead.
_INITIALISERS = {
    "glorot_normal": nn.init.xavier_normal_,
    "glorot_uniform": nn.init.xavier_uniform_,
    "he_normal": nn.init.kaiming_normal_,
    "he_uniform": nn.init.kaiming_uniform_,
}


class MLP(nn.Module):
    """``depth`` hidden layers of ``width`` units, one activation throughout."""

    def __init__(
        self,
        *,
        inputs: int,
        outputs: int,
        width: int,
        depth: int,
        activation: nn.Module,
        output_transform: OutputTransform | None = None,
    ) -> None:
        super().__init__()
        dims = [inputs, *([width] * depth), outputs]
        layers: list[nn.Module] = []
        for index, (fan_in, fan_out) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(nn.Linear(fan_in, fan_out))
            if index < len(dims) - 2:  # no activation on the output layer
                layers.append(activation)
        self.net = nn.Sequential(*layers)
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.output_transform is not None:
            y = self.output_transform(x, y)
        return y


@register_model("mlp")
def build_mlp(spec: NetSpec) -> MLP:
    """``NetSpec`` -> an initialised network on the ambient dtype/device."""
    if spec.options:
        raise ValueError(
            f"the 'mlp' architecture takes no options, got {sorted(spec.options)}; "
            "arch-specific knobs belong to the architecture that understands them"
        )

    activation = ACTIVATIONS.get(spec.activation)()
    transform = TRANSFORMS.get(spec.output_transform) if spec.output_transform else None

    net = MLP(
        inputs=spec.inputs,
        outputs=spec.outputs,
        width=spec.width,
        depth=spec.depth,
        activation=activation,
        output_transform=transform,
    )
    initialise(net, spec.init)
    return net


def initialise(module: nn.Module, kind: str) -> None:
    """Apply ``kind`` to every linear layer, zeroing biases.

    Draws from the *global* torch RNG, which ``configure_runtime`` seeded from
    the config. That is deliberate: initialisation happens once, before the
    trainer's dedicated sampling stream exists, and two runs sharing a seed must
    start from identical weights.
    """
    try:
        init_fn = _INITIALISERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown initialiser {kind!r}; available: {sorted(_INITIALISERS)}"
        ) from None

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            init_fn(layer.weight)
            nn.init.zeros_(layer.bias)


def build_net(spec: NetSpec) -> nn.Module:
    """``NetSpec`` -> a network, via whichever architecture it names."""
    return MODELS.get(spec.arch)(spec)


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count — a compute-parity number (DESIGN.md §8)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "MLP",
    "OutputTransform",
    "Sin",
    "build_mlp",
    "build_net",
    "count_parameters",
    "initialise",
]
