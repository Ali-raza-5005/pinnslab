"""The search space — where a metaheuristic is allowed to look.

DESIGN.md §9's claim that "a pydantic model *is* a schema that maps directly
onto ``SearchSpec.space``" is made literal here: a space names **config paths**
(``sampling.points.interior.n``), and applying a candidate produces a real
:class:`RunConfig` that is re-validated by pydantic. A search cannot therefore
propose a configuration a human could not have written by hand, and every
candidate has an ``identity_hash`` — which is what the cache, the results row
and the figure all join on.

Everything is encoded as a point in the unit cube ``[0, 1]^d``. That is the one
representation DE, CMA-ES, PSO and random search all share, so the algorithm
layer never learns what a "width" or a "collocation count" is: it optimises a
vector, and this module owns the meaning. Bounds are enforced on decode rather
than trusted, because a metaheuristic's whole job is to propose points near the
edge of the box.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from pinnslab.registry.config import RunConfig
from pinnslab.registry.schema import Spec

Scalar = float | int | bool | str


class Domain(Spec):
    """One searchable axis. Subclasses map ``[0, 1]`` onto a config value."""

    kind: str

    def decode(self, unit: float) -> Scalar:
        raise NotImplementedError

    def encode(self, value: Scalar) -> float:
        """The inverse, for seeding a search from a known-good config."""
        raise NotImplementedError


class Continuous(Domain):
    """A real interval, optionally searched in log space.

    ``log=True`` for anything spanning decades — learning rates, loss weights,
    tolerances. Searching a learning rate uniformly on ``[1e-5, 1e-1]`` spends
    99.99% of its proposals in the top decade, which is not a search.
    """

    kind: Literal["continuous"] = "continuous"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> Continuous:
        if not self.low < self.high:
            raise ValueError(f"need low < high, got [{self.low}, {self.high}]")
        if self.log and self.low <= 0:
            raise ValueError(
                f"a log-scaled domain needs low > 0, got {self.low}; log of a "
                "non-positive bound is undefined"
            )
        return self

    def decode(self, unit: float) -> float:
        unit = _clip_unit(unit)
        if self.log:
            return float(
                math.exp(
                    math.log(self.low)
                    + unit * (math.log(self.high) - math.log(self.low))
                )
            )
        return float(self.low + unit * (self.high - self.low))

    def encode(self, value: Scalar) -> float:
        value = float(value)  # type: ignore[arg-type]
        if self.log:
            span = math.log(self.high) - math.log(self.low)
            return _clip_unit((math.log(value) - math.log(self.low)) / span)
        return _clip_unit((value - self.low) / (self.high - self.low))


class Integer(Domain):
    """An inclusive integer range — widths, depths, collocation counts."""

    kind: Literal["integer"] = "integer"
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> Integer:
        if not self.low < self.high:
            raise ValueError(f"need low < high, got [{self.low}, {self.high}]")
        if self.log and self.low <= 0:
            raise ValueError(f"a log-scaled domain needs low > 0, got {self.low}")
        return self

    def decode(self, unit: float) -> int:
        unit = _clip_unit(unit)
        if self.log:
            span = math.log(self.high + 1) - math.log(self.low)
            raw = math.exp(math.log(self.low) + unit * span)
        else:
            # +1 then floor, so every integer gets an equal slice of [0, 1]
            # instead of the two endpoints getting half-width slices.
            raw = self.low + unit * (self.high - self.low + 1)
        return int(min(self.high, max(self.low, math.floor(raw))))

    def encode(self, value: Scalar) -> float:
        """The **midpoint** of the value's slice, in whichever space decode uses.

        Encoding the slice's left edge instead round-trips wrong: ``decode``
        floors, so an edge that floating-point rounds a hair low lands on
        ``value - 1``. Measured on this domain — ``encode(45)`` at the edge
        decoded back to 44.
        """
        value = int(value)  # type: ignore[arg-type]
        if self.log:
            span = math.log(self.high + 1) - math.log(self.low)
            return _clip_unit((math.log(value + 0.5) - math.log(self.low)) / span)
        return _clip_unit((value - self.low + 0.5) / (self.high - self.low + 1))


class Categorical(Domain):
    """An unordered set — sampler names, activations, initialisers.

    Order in ``choices`` is part of the space's identity: a metaheuristic
    treats the encoded axis as continuous, so neighbouring indices are
    neighbouring proposals. Reordering the list is a different search.
    """

    kind: Literal["categorical"] = "categorical"
    choices: tuple[Scalar, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _distinct(self) -> Categorical:
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"choices must be distinct, got {self.choices}")
        return self

    def decode(self, unit: float) -> Scalar:
        index = math.floor(_clip_unit(unit) * len(self.choices))
        return self.choices[min(index, len(self.choices) - 1)]

    def encode(self, value: Scalar) -> float:
        try:
            index = self.choices.index(value)
        except ValueError:
            raise ValueError(
                f"{value!r} is not one of {self.choices}"
            ) from None
        return (index + 0.5) / len(self.choices)


def _clip_unit(unit: float) -> float:
    """Clamp to ``[0, 1]``. Proposals outside the box are the normal case —
    DE's mutation and CMA-ES's sampling both routinely overshoot — so this is
    the box constraint, not an error path."""
    return min(1.0, max(0.0, float(unit)))


#: The domain types a YAML space may name.
DOMAIN_TYPES = {
    "continuous": Continuous,
    "integer": Integer,
    "categorical": Categorical,
}


def build_domain(spec: Mapping[str, Any]) -> Domain:
    """``{"kind": "integer", "low": 8, "high": 128}`` -> a :class:`Domain`."""
    payload = dict(spec)
    kind = payload.get("kind")
    if kind not in DOMAIN_TYPES:
        raise ValueError(
            f"unknown domain kind {kind!r}; expected one of {sorted(DOMAIN_TYPES)}"
        )
    return DOMAIN_TYPES[kind](**payload)


class SearchSpace:
    """An ordered set of config paths and the domains they range over.

    Ordered because the unit-cube vector's axes are positional and a search's
    checkpoint stores vectors: reordering the space would silently reinterpret
    every stored population. The order is the insertion order of the mapping.
    """

    def __init__(self, domains: Mapping[str, Domain | Mapping[str, Any]]) -> None:
        if not domains:
            raise ValueError("a search space with no axes has nothing to search")
        self.domains: dict[str, Domain] = {
            path: d if isinstance(d, Domain) else build_domain(d)
            for path, d in domains.items()
        }
        self.paths: tuple[str, ...] = tuple(self.domains)

    def __len__(self) -> int:
        return len(self.domains)

    @property
    def dim(self) -> int:
        return len(self.domains)

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """``n`` uniform points in the unit cube, shaped ``(n, dim)``."""
        return rng.random((n, self.dim))

    def decode(self, vector: Sequence[float]) -> dict[str, Scalar]:
        """A unit-cube point -> ``{config path: value}``."""
        if len(vector) != self.dim:
            raise ValueError(
                f"expected a vector of length {self.dim} for space "
                f"{self.paths}, got {len(vector)}"
            )
        return {
            path: domain.decode(unit)
            for (path, domain), unit in zip(
                self.domains.items(), vector, strict=True
            )
        }

    def encode(self, values: Mapping[str, Scalar]) -> np.ndarray:
        """``{config path: value}`` -> a unit-cube point.

        For seeding a search from a hand-tuned config, so generation 0 contains
        the incumbent rather than starting behind it.
        """
        missing = [p for p in self.paths if p not in values]
        if missing:
            raise KeyError(f"no value given for {missing}")
        return np.array(
            [self.domains[p].encode(values[p]) for p in self.paths], dtype=float
        )

    def apply(self, base: RunConfig, vector: Sequence[float]) -> RunConfig:
        """Decode a candidate and produce a re-validated :class:`RunConfig`.

        Re-validated, not patched: pydantic's own field constraints are the
        final word on what a legal configuration is, so a space whose bounds
        disagree with the schema fails here rather than producing a run that
        silently means something else.
        """
        payload = base.to_dict()
        for path, value in self.decode(vector).items():
            _set_path(payload, path, value)
        return RunConfig(**payload)

    def validate_against(self, base: RunConfig) -> None:
        """Fail now if a path does not exist on this config.

        A typo'd path would otherwise create a *new* key, which
        ``extra="forbid"`` rejects only on the axes that reach a nested model —
        and a search that silently optimises nothing is the worst possible
        failure, because it produces a full set of plausible results.
        """
        payload = base.to_dict()
        for path in self.paths:
            _resolve(payload, path)


def _resolve(payload: Any, path: str) -> Any:
    """Walk a dotted path, raising a message that names the wrong segment."""
    node = payload
    walked: list[str] = []
    for key in path.split("."):
        walked.append(key)
        try:
            node = node[int(key)] if isinstance(node, list) else node[key]
        except (KeyError, IndexError, ValueError, TypeError):
            raise KeyError(
                f"search path {path!r} does not exist on this config: "
                f"{'.'.join(walked)} is not reachable. The space must name "
                "fields the base config already declares — a search cannot add "
                "a field the schema does not have."
            ) from None
    return node


def _set_path(payload: Any, path: str, value: Scalar) -> None:
    keys = path.split(".")
    node = _resolve(payload, ".".join(keys[:-1])) if len(keys) > 1 else payload
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        if last not in node:
            raise KeyError(f"search path {path!r} does not exist on this config")
        node[last] = value


__all__ = [
    "DOMAIN_TYPES",
    "Categorical",
    "Continuous",
    "Domain",
    "Integer",
    "SearchSpace",
    "build_domain",
]
