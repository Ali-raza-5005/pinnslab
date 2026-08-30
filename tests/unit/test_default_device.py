"""DeepXDE must not leave a default-device mode installed in this process.

``deepxde.backend.pytorch.tensor`` calls ``torch.set_default_device("cuda")`` at
import time whenever CUDA is available. That is a process-global
``__torch_function__`` mode: from then on **every tensor factory called without
an explicit device allocates on the GPU**, including ones handed a CPU
generator, which is a hard error rather than a slow path.

This library assumes the opposite everywhere — ``Domain.sample`` takes a
``device``, ``_to_tensor`` honours it, and ``Trainer`` builds its sampling
generator on CPU *on purpose*, so that a collocation cloud is a function of the
seed and not of the hardware it was drawn on. The mode silently contradicts
that for every device-less call.

It cost a full GPU sweep to find. On 2026-08-30 the first Kaggle run of
paper-01 failed at the first collocation draw of all five seeds with::

    RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

from ``_numpy_stream``'s ``torch.randint(..., generator=<cpu generator>)``.

**The bug is invisible on CPU** — ``torch.cuda.is_available()`` is False, so
DeepXDE never installs the mode — which is why the whole test suite passed
while the pipeline was broken. The tests below therefore do not wait for a GPU:
they install a non-CPU default device with ``torch.device("meta")``, which
hijacks factories the same way, and check the two invariants that matter.
"""

from __future__ import annotations

import pytest
import torch

from pinnslab.geometry import Domain, interval, with_time
from pinnslab.geometry.adapters import _numpy_stream
from pinnslab.utils.seeding import make_generator

pytestmark = pytest.mark.unit


@pytest.fixture
def domain() -> Domain:
    """The Burgers domain: x in [-1, 1], t in [0, 1]."""
    return with_time(interval(-1.0, 1.0), 0.0, 1.0)


def test_importing_the_adapter_leaves_the_default_device_alone():
    """The regression proper. Trivial on CPU, load-bearing on any GPU box.

    ``pinnslab.geometry.adapters`` is imported by the fixture above, and by
    everything that builds a problem. If DeepXDE's mode is still installed
    afterwards, this fails on exactly the machines the study runs on.
    """
    assert torch.empty(0).device.type == "cpu", (
        "a default-device mode is installed after importing the geometry "
        "adapter; every device-less tensor factory in this process now "
        "allocates somewhere else, and sampling with a CPU generator raises"
    )


def test_the_numpy_stream_draws_on_its_generators_device():
    """Under a hijacked default device, the seed draw must still be explicit.

    Without ``device=generator.device`` the draw lands on the ambient device and
    ``.item()`` fails — the CPU-reproducible form of the CUDA error above.
    """
    generator = make_generator(7, device="cpu")
    with torch.device("meta"):
        with _numpy_stream(generator):
            pass  # entering is the whole test; the draw happens in __enter__


def test_sampling_survives_a_hijacked_default_device(domain):
    """End to end: a cloud drawn while some library owns the default device.

    This is the failure the GPU sweep hit, in the form a CPU can reproduce.
    """
    generator = make_generator(11, device="cpu")
    with torch.device("meta"):
        points = domain.sample(
            "interior", 32, generator=generator, dtype=torch.float64, device="cpu"
        )
    assert points.device.type == "cpu"
    assert points.shape == (32, 2)
    assert torch.isfinite(points).all()


def test_the_cloud_does_not_depend_on_the_ambient_device(domain):
    """Same seed, different ambient device, identical points.

    The property that makes a collocation cloud a function of the seed rather
    than of the machine, which is what lets a local CPU check reproduce a Kaggle
    run and what PROTOCOL-style pairing across arms relies on.
    """
    reference = domain.sample(
        "interior", 64, generator=make_generator(3), dtype=torch.float64, device="cpu"
    )
    with torch.device("meta"):
        under_mode = domain.sample(
            "interior",
            64,
            generator=make_generator(3),
            dtype=torch.float64,
            device="cpu",
        )
    assert torch.equal(reference, under_mode)
