"""The DeepXDE seam — the ONLY module in which deepxde may be imported.

DESIGN.md §1: DeepXDE is a dependency, never a foundation. It is used here for
one thing, geometry as a *point generator* (CSG domains, boundary/initial
sampling), because building that from scratch is the genuinely expensive part
and borrowing it is the whole point. Nothing else in ``pinnslab`` knows DeepXDE
exists; :class:`Domain`'s public methods are the entire contract, and if
DeepXDE breaks or we outgrow it, exactly this file gets rewritten.

``tests/unit/test_geometry.py`` enforces the single-import-site rule by scanning
the package, so the rule is structural rather than a convention people remember.

Three properties of DeepXDE that this module exists to neutralise
-----------------------------------------------------------------

**1. The backend is ambient and its default is wrong for us.** ``import deepxde``
picks a backend from ``DDE_BACKEND``, falling back to whatever it finds
installed — on a machine with TensorFlow present that is TensorFlow, and the
import then dies on an unrelated missing ``tensorflow_probability``. The default
is set below *before* the import, and the resulting backend is verified after
it.

**2. DeepXDE samples from numpy's global RNG.** ``geometry.sampler.pseudorandom``
calls ``np.random.random`` directly, so point clouds would depend on how many
unrelated numpy draws happened first — and a resumed run would silently get a
different cloud than the uninterrupted one. Every sampling call here is wrapped
in :func:`_numpy_stream`, which seeds numpy from an explicit
:class:`torch.Generator` and restores the global state afterwards. Sampling is
then reproducible from the trainer's own checkpointed generator alone.

**3. DeepXDE's default float is float32, and changing it mutates torch.**
``dde.config.set_default_float`` calls ``torch.set_default_dtype`` as a side
effect, which would break the promise in ``pinnslab/__init__`` that importing
the package touches no global torch state, and would take precision out of the
config where DESIGN.md §5 requires it to live. We set DeepXDE to float64 and
immediately put torch's default back, so points are always *generated* at full
precision and cast to the run's dtype on the way out. Sampling at float32 and
widening afterwards would bake float32-resolution coordinates into a float64
run.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

# Must precede the deepxde import: the backend is chosen at import time.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402  (see above — ordering is load-bearing)

Region = Literal["interior", "boundary", "initial"]

#: Our sampler names -> DeepXDE's. A stable boundary, so a config committed to
#: ``results/`` never encodes DeepXDE's capitalisation.
_STRATEGIES: dict[str, str] = {
    "pseudo": "pseudo",
    "lhs": "LHS",
    "halton": "Halton",
    "hammersley": "Hammersley",
    "sobol": "Sobol",
}

#: The geometric draws this seam offers, as a public set: ``samplers.py``
#: registers one built-in sampler per name, and a config's ``strategy:`` is
#: validated against the registry rather than against this dict directly.
GEOMETRY_STRATEGIES = frozenset(_STRATEGIES)

#: Strategies that produce a fixed sequence for a given ``n``, ignoring any RNG.
#: Resampling with one of these returns *the same points every time*, which
#: makes ``StageSpec.resample_every`` a silent no-op — a trap worth naming,
#: since a sampling paper whose resampling does nothing still trains fine.
DETERMINISTIC_STRATEGIES = frozenset({"halton", "hammersley", "sobol"})


def _use_float64_without_touching_torch() -> None:
    """Put DeepXDE in float64 and undo its torch side effect."""
    previous = torch.get_default_dtype()
    dde.config.set_default_float("float64")
    torch.set_default_dtype(previous)


def _restore_default_device() -> None:
    """Undo DeepXDE's *device* side effect, as we already undo its dtype one.

    ``deepxde.backend.pytorch.tensor`` calls ``torch.set_default_device("cuda")``
    at import time whenever CUDA is available. That installs a process-global
    ``__torch_function__`` mode, so from then on **every tensor factory called
    without an explicit device allocates on the GPU** — including ones handed a
    CPU generator, which is a hard error rather than a slow path.

    That is exactly how it surfaced: ``_numpy_stream`` draws its numpy seed with
    ``torch.randint(..., generator=<cpu generator>)`` and died with
    ``Expected a 'cuda' device type for generator but found 'cpu'`` at the first
    collocation draw of every run in the first GPU sweep (2026-08-30). It cannot
    reproduce on CPU: ``torch.cuda.is_available()`` is False there, so DeepXDE
    never sets the mode and the whole class of bug is invisible.

    The rest of this library places tensors explicitly and assumes the default
    is CPU — ``Domain.sample`` takes a ``device`` argument, ``_to_tensor``
    honours it, ``Trainer`` builds its sampling generator on CPU on purpose so
    that a collocation cloud is a function of the seed and not of the hardware
    it was drawn on. Leaving DeepXDE's mode installed silently contradicts that
    assumption for every future device-less call. Restore it here, immediately
    after import, for the same reason and in the same place as the dtype fix.

    DeepXDE itself does not need the mode: pinnslab uses its geometry only, and
    ``random_points`` and friends return numpy arrays.
    """
    torch.set_default_device("cpu")


def _verify_backend() -> None:
    name = getattr(dde.backend, "backend_name", None)
    if name != "pytorch":
        raise RuntimeError(
            f"deepxde is running on the {name!r} backend, but pinnslab requires "
            "'pytorch' — geometry output is converted to torch tensors here and "
            "every other backend brings its own array type into the process. "
            "Set DDE_BACKEND=pytorch in the environment before anything imports "
            "deepxde."
        )


_verify_backend()
_use_float64_without_touching_torch()
_restore_default_device()


@contextlib.contextmanager
def _numpy_stream(generator: torch.Generator | None) -> Iterator[None]:
    """Drive DeepXDE's numpy-global sampling from an explicit torch stream.

    Draws one integer from ``generator`` (advancing it, so successive calls
    differ), seeds numpy with it, and restores numpy's previous global state on
    the way out. ``None`` leaves numpy alone, for callers that genuinely want
    ambient randomness — no production path should pass it.
    """
    if generator is None:
        yield
        return
    # ``device=`` is explicit even though _restore_default_device() should make
    # it redundant: this exact call is the one that broke, and a generator's
    # own device is the only correct answer for a draw taken from it.
    seed = int(
        torch.randint(
            0, 2**31 - 1, (1,), generator=generator, device=generator.device
        ).item()
    )
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


@dataclass(frozen=True)
class Domain:
    """A geometry, exposed as a point generator.

    The wrapped DeepXDE object is reachable only as :attr:`_geometry` and must
    not be touched outside this module — it is held rather than copied because
    training resamples repeatedly and needs a live handle.
    """

    _geometry: Any
    dim: int
    #: True when the last coordinate is time (the domain is a space x time
    #: product), which is what makes ``region="initial"`` meaningful.
    time_dependent: bool
    #: ``(lo, hi)`` per coordinate, for input normalisation.
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def bounds(
        self, *, dtype: torch.dtype | None = None, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kw = {"dtype": dtype or torch.get_default_dtype(), "device": device}
        return torch.tensor(self.lower, **kw), torch.tensor(self.upper, **kw)

    def sample(
        self,
        region: Region,
        n: int,
        *,
        generator: torch.Generator | None = None,
        strategy: str = "pseudo",
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """``(n, dim)`` points drawn from ``region``. Never returns a DeepXDE type.

        Args:
            region: ``interior`` (the full space-time volume), ``boundary`` (the
                spatial boundary at random times) or ``initial`` (t = t0).
            generator: the stream to drive sampling from. Omitting it makes the
                run irreproducible and is never right in training code.
            strategy: a key of :data:`_STRATEGIES`. See
                :data:`DETERMINISTIC_STRATEGIES` before combining a quasirandom
                strategy with resampling.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"unknown sampling strategy {strategy!r}; available: "
                f"{sorted(_STRATEGIES)}"
            )
        if region == "initial" and not self.time_dependent:
            raise ValueError(
                "region='initial' needs a time-dependent domain; this one has no "
                "time coordinate, so there is no initial slice to sample"
            )

        sampler = {
            "interior": self._geometry.random_points,
            "boundary": self._geometry.random_boundary_points,
            "initial": getattr(self._geometry, "random_initial_points", None),
        }.get(region)
        if sampler is None:
            raise ValueError(
                f"unknown region {region!r}; expected one of "
                "'interior', 'boundary', 'initial'"
            )

        with _numpy_stream(generator):
            points = sampler(n, random=_STRATEGIES[strategy])

        return self._to_tensor(points, n=n, region=region, dtype=dtype, device=device)

    def _to_tensor(
        self,
        points: np.ndarray,
        *,
        n: int,
        region: str,
        dtype: torch.dtype | None,
        device: torch.device | str,
    ) -> torch.Tensor:
        if points.shape != (n, self.dim):
            # DeepXDE quietly returns fewer points than asked for in some CSG
            # cases. Silently training on 480 points while the config records
            # 500 would corrupt the compute-parity numbers of DESIGN.md §8.
            raise RuntimeError(
                f"deepxde returned {points.shape} points for "
                f"region={region!r}, n={n}, dim={self.dim}; expected "
                f"{(n, self.dim)}. Point counts are part of the experimental "
                "condition and must not drift silently."
            )
        return torch.as_tensor(
            np.ascontiguousarray(points),
            dtype=dtype or torch.get_default_dtype(),
            device=device,
        )


def interval(lower: float, upper: float) -> Domain:
    """The 1-D spatial interval ``[lower, upper]``."""
    if not upper > lower:
        raise ValueError(f"interval needs upper > lower, got [{lower}, {upper}]")
    return Domain(
        _geometry=dde.geometry.Interval(lower, upper),
        dim=1,
        time_dependent=False,
        lower=(float(lower),),
        upper=(float(upper),),
    )


def with_time(space: Domain, t0: float, t1: float) -> Domain:
    """``space x [t0, t1]``, with time as the last coordinate."""
    if space.time_dependent:
        raise ValueError("domain already carries a time coordinate")
    if not t1 > t0:
        raise ValueError(f"time domain needs t1 > t0, got [{t0}, {t1}]")
    return Domain(
        _geometry=dde.geometry.GeometryXTime(
            space._geometry, dde.geometry.TimeDomain(t0, t1)
        ),
        dim=space.dim + 1,
        time_dependent=True,
        lower=(*space.lower, float(t0)),
        upper=(*space.upper, float(t1)),
    )


__all__ = [
    "DETERMINISTIC_STRATEGIES",
    "GEOMETRY_STRATEGIES",
    "Domain",
    "Region",
    "interval",
    "with_time",
]
