"""The validated config (DESIGN.md §4, §9).

YAML on disk -> dict -> pydantic -> hash. No Hydra: the ``search/`` layer already
owns multi-run sweeping, and a pydantic model *is* a schema whose typed field
bounds double as the metaheuristic's search space.

No hyperparameter is ever a Python literal in a script (CLAUDE.md rule 4), so
anything a run needs in order to differ from another run belongs in here.

Two groups of fields, deliberately treated differently (DESIGN.md §4). The
*stable* axes — identity, precision, stages, optimizers, evaluation — are worth
abstraction because everything downstream joins on them. The *volatile* axes —
``problem``, ``nets``, ``residuals``, ``weighting``, ``sampling`` — are kept
flat and near-copy-pasteable, because a wrong abstraction there is far more
expensive than the duplication it would save.

The volatile fields all default to empty. That is not laxness: the ``Trainer``
takes its networks and residual function as plain callables, which is the
escape hatch DESIGN.md §4 requires (a genuinely strange paper must be
implementable without editing core) and is how the infrastructure tests drive
the loop with no ``physics/`` at all. What the defaults must never become is a
way to smuggle hyperparameters back into a script — :func:`assemble` refuses to
build anything the config did not declare.

What the config hash covers
---------------------------
``identity_hash()`` deliberately excludes ``seed``, ``device`` and the purely
operational fields (``name``, ``tags``, ``logging``, ``checkpoint``):

* **seed** — five seeds of one condition must share a hash, or the DESIGN.md §8
  "median + IQR over >=5 seeds" groupby has nothing to group on. A *run* is
  identified by the pair ``(config_hash, seed)``, and both are on every row.
* **device / logging / checkpoint** — changing checkpoint cadence or trace
  density does not make it a different experiment, and pretending otherwise
  would fragment the search layer's candidate cache.

``dtype`` IS hashed: float32 and float64 results are not comparable
(DESIGN.md §5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from pinnslab.registry.hashing import config_hash, to_jsonable
from pinnslab.registry.schema import MetricSchedule, Spec

#: Fields that describe how a run is operated, not what condition it tests.
HASH_EXCLUDE = frozenset({"name", "tags", "seed", "device", "logging", "checkpoint"})

Scalar = float | int | bool | str


class ProblemSpec(Spec):
    """Which frozen benchmark this run solves.

    The PDE itself is not a hyperparameter — geometry, residual form, boundary
    and initial conditions and the reference solution all live in a frozen
    ``benchmarks/`` module (DESIGN.md §3) so that every paper compares against
    the same problem. What the config chooses is *which* benchmark, plus the few
    physical constants a benchmark declares as varyable.
    """

    name: str
    options: dict[str, Scalar] = Field(default_factory=dict)


class NetSpec(Spec):
    """One network.

    Architecture is a *volatile* axis (DESIGN.md §4): this stays flat and
    near-copy-pasteable rather than growing a class hierarchy. The fields that
    are typed are exactly the ones architecture search will search over — a
    pydantic field's bounds double as the metaheuristic's search space (§9) —
    while genuinely arch-specific knobs (Fourier feature scale, SIREN omega_0)
    go in ``options`` and are forwarded verbatim to the registered factory.
    """

    arch: str = "mlp"
    inputs: int = Field(gt=0)
    outputs: int = Field(default=1, gt=0)
    #: Neurons per hidden layer.
    width: int = Field(default=32, gt=0)
    #: Number of hidden layers.
    depth: int = Field(default=4, gt=0)
    activation: str = "tanh"
    init: str = "glorot_normal"
    #: Registered output transform, for hard-constrained BCs/ICs (DESIGN.md §4
    #: conformance item 5). Per-net rather than per-run: with multiple networks
    #: each field carries its own hard constraint.
    output_transform: str | None = None
    options: dict[str, Scalar] = Field(default_factory=dict)


class ParamSpec(Spec):
    """A trainable unknown that is not a network weight.

    DESIGN.md §4 conformance item 2: an inverse problem is an ordinary run whose
    PDE coefficients happen to be parameters. They reach the optimizer through
    the ``extra.<key>`` selector namespace, so no special casing is needed
    anywhere in the loop.
    """

    init: float
    shape: tuple[int, ...] = ()
    trainable: bool = True


class ResidualSpec(Spec):
    """One named term of the loss.

    Residual functions return per-point tensors of shape ``(N,)`` (CLAUDE.md
    rule 5); which points is decided by ``points``, naming a group declared in
    :class:`SamplingSpec`. Several residuals may share one point group.
    """

    kind: str
    points: str = "interior"
    options: dict[str, Scalar] = Field(default_factory=dict)


class WeightingSpec(Spec):
    """How the per-point residuals are reduced to one scalar.

    All reduction lives here (DESIGN.md §4 decision 1). ``coefficients`` are the
    per-term scalars every scheme understands; anything a specific scheme needs
    (causal tolerance, NTK update period) goes in ``options``.
    """

    kind: str = "mean"
    coefficients: dict[str, float] = Field(default_factory=dict)
    options: dict[str, Scalar] = Field(default_factory=dict)


class PointSetSpec(Spec):
    """One named group of collocation points."""

    #: Which part of the benchmark's geometry to draw from.
    region: Literal["interior", "boundary", "initial"] = "interior"
    n: int = Field(gt=0)
    #: Registered sampler. ``pseudo`` is uniform-random; the adaptive strategies
    #: that are paper 1's subject register their own names.
    strategy: str = "pseudo"
    options: dict[str, Scalar] = Field(default_factory=dict)


class SamplingSpec(Spec):
    """The point groups a run trains on.

    Resampling *cadence* is deliberately not here — it is ``StageSpec.
    resample_every``, because it is a property of a training stage (warm up on
    fixed points, then resample every K steps) rather than of the point set.
    """

    points: dict[str, PointSetSpec] = Field(default_factory=dict)


class OptimizerSpec(Spec):
    """One optimizer over one slice of the parameters, in one direction.

    Optimizers are a *list* with a param selector and a direction rather than a
    single object (DESIGN.md §4): min-max and self-adaptive weighting schemes
    then fall out for free as a second optimizer doing ascent.
    """

    name: str = "adam"
    lr: float = Field(default=1e-3, gt=0.0)
    #: Regex matched against parameter paths, ``"<net>.<param>"`` or
    #: ``"extra.<key>"``. Default selects everything.
    params: str = ".*"
    direction: Literal["min", "max"] = "min"
    max_grad_norm: float | None = Field(default=None, gt=0.0)
    #: Forwarded verbatim to the registered optimizer factory.
    options: dict[str, Scalar] = Field(default_factory=dict)


class StageSpec(Spec):
    """A contiguous block of steps under one set of optimizers."""

    name: str
    optimizers: list[OptimizerSpec] = Field(min_length=1)
    steps: int = Field(gt=0)
    #: Steps between resampling hooks; ``None`` disables resampling.
    resample_every: int | None = Field(default=None, gt=0)


class EvalSpec(Spec):
    """What counts as "better" and what counts as "done"."""

    #: Metric defining the best checkpoint; ``None`` means best-tracking is off
    #: and only the last checkpoint is kept.
    best_metric: str | None = None
    best_mode: Literal["min", "max"] = "min"
    #: Time-to-target-accuracy is reported alongside final accuracy
    #: (DESIGN.md §8); set both fields to record it.
    target_metric: str | None = None
    target_value: float | None = None
    #: A non-finite loss ends the run with status=diverged rather than a
    #: traceback — failure rate is a reported metric (DESIGN.md §11).
    stop_on_nonfinite: bool = True

    @model_validator(mode="after")
    def _target_is_complete(self) -> EvalSpec:
        if (self.target_metric is None) != (self.target_value is None):
            raise ValueError("target_metric and target_value must be set together")
        return self


class LoggingSpec(Spec):
    """Trace density. Not part of the config identity."""

    trace: MetricSchedule = Field(default_factory=MetricSchedule)


class CheckpointSpec(Spec):
    """Checkpoint cadence. Not part of the config identity.

    Best + last only by default (DESIGN.md §11); every-N is opt-in for the rare
    analysis that needs it.
    """

    every_seconds: float | None = Field(default=600.0, gt=0.0)
    every_steps: int | None = Field(default=None, gt=0)
    save_best: bool = True
    save_last: bool = True


class RunConfig(Spec):
    """Everything needed to reproduce one training run."""

    name: str = "run"
    tags: dict[str, str] = Field(default_factory=dict)

    seed: int = Field(default=0, ge=0)
    dtype: Literal["float64", "float32"] = "float64"
    device: str = "auto"
    deterministic: bool = True

    # --- volatile axes: what is being solved, with what -----------------------
    problem: ProblemSpec | None = None
    nets: dict[str, NetSpec] = Field(default_factory=dict)
    extra_params: dict[str, ParamSpec] = Field(default_factory=dict)
    residuals: dict[str, ResidualSpec] = Field(default_factory=dict)
    weighting: WeightingSpec = Field(default_factory=WeightingSpec)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    # --------------------------------------------------------------------------

    stages: list[StageSpec] = Field(min_length=1)
    eval: EvalSpec = Field(default_factory=EvalSpec)
    logging: LoggingSpec = Field(default_factory=LoggingSpec)
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)

    @field_validator("stages")
    @classmethod
    def _stage_names_unique(cls, stages: list[StageSpec]) -> list[StageSpec]:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")
        return stages

    @model_validator(mode="after")
    def _volatile_axes_are_consistent(self) -> RunConfig:
        """Cross-field checks that turn a typo'd YAML key into a load error.

        These only fire once a config declares residuals at all, so the
        callable-driven path (empty volatile axes) is untouched. Every one of
        them is a mistake that would otherwise surface as a confusing failure
        deep inside a Kaggle session rather than at load time.
        """
        if not self.residuals:
            return self

        if not self.nets:
            raise ValueError(
                "residuals are declared but nets is empty; a residual has "
                "nothing to differentiate"
            )

        known_points = set(self.sampling.points)
        missing = {
            name: spec.points
            for name, spec in self.residuals.items()
            if spec.points not in known_points
        }
        if missing:
            raise ValueError(
                f"residuals reference point groups that sampling.points does not "
                f"declare: {missing}; declared groups are {sorted(known_points)}"
            )

        unknown = sorted(set(self.weighting.coefficients) - set(self.residuals))
        if unknown:
            raise ValueError(
                f"weighting.coefficients names terms that are not residuals: "
                f"{unknown}; residuals are {sorted(self.residuals)}. A coefficient "
                "on a misspelled term would silently do nothing."
            )
        return self

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)

    def identity(self) -> dict[str, Any]:
        """The JSON-native subset of the config that defines the condition."""
        payload = self.model_dump(mode="json")
        return {k: v for k, v in payload.items() if k not in HASH_EXCLUDE}

    def identity_hash(self) -> str:
        return config_hash(self.identity())

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def load_config(path: str | Path) -> RunConfig:
    """YAML -> validated :class:`RunConfig`. The only supported entry point."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping, got {type(raw).__name__}"
        )
    return RunConfig(**raw)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=True, default_flow_style=False)


__all__ = [
    "HASH_EXCLUDE",
    "CheckpointSpec",
    "EvalSpec",
    "LoggingSpec",
    "NetSpec",
    "OptimizerSpec",
    "ParamSpec",
    "PointSetSpec",
    "ProblemSpec",
    "ResidualSpec",
    "RunConfig",
    "SamplingSpec",
    "StageSpec",
    "WeightingSpec",
    "dump_config",
    "load_config",
]
