"""Device resolution and runtime configuration (DESIGN.md §5).

Precision is applied here and nowhere else. Importing ``pinnslab`` must not
mutate global torch state; :func:`configure_runtime` is the single point where a
validated config takes effect, which is what keeps ``dtype`` an experimental
variable (hashed into the config, recorded on every row) rather than an ambient
property of whoever's notebook is running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pinnslab.utils.logging import get_logger
from pinnslab.utils.seeding import set_seed

if TYPE_CHECKING:
    from pinnslab.registry.config import RunConfig

log = get_logger(__name__)

DTYPES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
}

#: Substring -> the precision that GPU is worth using (DESIGN.md §5). T4 FP64 is
#: ~1/32 of its FP32; P100 FP64 is ~1/2. Advisory only — never blocking.
_PROFILE_PRECISION_HINT = {"T4": "float32", "P100": "float64"}


@dataclass(frozen=True)
class RuntimeContext:
    """What the process was actually configured to, for the result row."""

    device: torch.device
    dtype: torch.dtype
    dtype_name: str
    gpu_name: str
    device_profile: str


def torch_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r}; pinnslab uses {sorted(DTYPES)} only "
            "(mixed/half precision is not a supported experimental condition)"
        ) from None


def resolve_device(spec: str = "auto") -> torch.device:
    """``'auto' | 'cpu' | 'cuda' | 'cuda:1'`` -> a concrete device."""
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config requests device {spec!r} but CUDA is unavailable")
    return device


def gpu_name() -> str:
    """Name of the GPU this process would use, or ``'cpu'``.

    Recorded on every result row; aggregation refuses to plot a comparison group
    that spans more than one value (DESIGN.md §5).
    """
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def device_profile() -> str:
    """Coarse hardware identity: ``'2xTesla-T4'``, ``'1xTesla-P100'``, ``'cpu'``."""
    if not torch.cuda.is_available():
        return "cpu"
    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    if len(names) > 1:
        # Heterogeneous GPUs would break the uniformity rule in a way that a
        # single label cannot express; make it loud in the recorded value.
        joined = "+".join(sorted(n.replace(" ", "-") for n in names))
        return f"MIXED[{joined}]"
    return f"{torch.cuda.device_count()}x{names.pop().replace(' ', '-')}"


def configure_runtime(cfg: RunConfig) -> RuntimeContext:
    """Apply a validated config to the process. Call once, at run start."""
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    dtype = torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)
    device = resolve_device(cfg.device)

    ctx = RuntimeContext(
        device=device,
        dtype=dtype,
        dtype_name=cfg.dtype,
        gpu_name=gpu_name(),
        device_profile=device_profile(),
    )
    _warn_on_precision_mismatch(ctx)
    log.info(
        "runtime: device=%s dtype=%s profile=%s seed=%d deterministic=%s",
        ctx.device,
        ctx.dtype_name,
        ctx.device_profile,
        cfg.seed,
        cfg.deterministic,
    )
    return ctx


def _warn_on_precision_mismatch(ctx: RuntimeContext) -> None:
    for marker, preferred in _PROFILE_PRECISION_HINT.items():
        if marker in ctx.device_profile and ctx.dtype_name != preferred:
            log.warning(
                "%s prefers %s work (DESIGN.md §5) but this run is %s; fine if "
                "deliberate, but never mix precisions within a comparison group",
                marker,
                preferred,
                ctx.dtype_name,
            )


__all__ = [
    "DTYPES",
    "RuntimeContext",
    "configure_runtime",
    "device_profile",
    "gpu_name",
    "resolve_device",
    "torch_dtype",
]
