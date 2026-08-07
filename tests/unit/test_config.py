"""The volatile axes of ``RunConfig`` (DESIGN.md §4).

These fields describe *what is being solved, with what*. They all default to
empty so that the callable-driven ``Trainer`` path — the escape hatch, and how
the infrastructure tests run with no ``physics/`` — keeps working. The cost of
that permissiveness is that a half-declared config could otherwise sail through
validation and fail deep inside a Kaggle session, so the cross-field checks
below are load-bearing rather than decorative.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pinnslab.registry.config import (
    NetSpec,
    ParamSpec,
    PointSetSpec,
    ProblemSpec,
    ResidualSpec,
    RunConfig,
    SamplingSpec,
    WeightingSpec,
    dump_config,
    load_config,
)
from tests.conftest import toy_config

pytestmark = pytest.mark.unit


def burgers_like(**overrides) -> RunConfig:
    """A config shaped like the real thing, without needing the real thing."""
    base = {
        "problem": ProblemSpec(name="burgers1d", options={"nu": 0.01}),
        "nets": {"u": NetSpec(inputs=2, outputs=1, width=20, depth=3)},
        "residuals": {
            "pde": ResidualSpec(kind="burgers1d", points="interior"),
            "ic": ResidualSpec(kind="dirichlet", points="initial"),
        },
        "sampling": SamplingSpec(
            points={
                "interior": PointSetSpec(region="interior", n=2540),
                "initial": PointSetSpec(region="initial", n=80),
            }
        ),
        "weighting": WeightingSpec(kind="mean", coefficients={"ic": 10.0}),
    }
    base.update(overrides)
    return toy_config(**base)


# -- the permissive default is still a valid config ---------------------------


def test_a_config_with_no_volatile_axes_validates():
    """The callable-driven path: nets and residual_fn handed to the Trainer."""
    cfg = toy_config()
    assert cfg.nets == {}
    assert cfg.residuals == {}
    assert cfg.sampling.points == {}
    assert cfg.weighting.kind == "mean"


def test_a_fully_declared_config_validates():
    cfg = burgers_like()
    assert cfg.problem is not None
    assert cfg.problem.options["nu"] == 0.01
    assert cfg.nets["u"].inputs == 2
    assert cfg.residuals["pde"].points == ("interior",)


# -- cross-field checks -------------------------------------------------------


def test_residuals_without_nets_are_rejected():
    with pytest.raises(ValidationError, match="nothing to differentiate"):
        burgers_like(nets={})


def test_a_residual_naming_an_undeclared_point_group_is_rejected():
    """The single most likely YAML typo, and otherwise a KeyError mid-run."""
    with pytest.raises(ValidationError, match="point groups"):
        burgers_like(
            residuals={"pde": ResidualSpec(kind="burgers1d", points="interor")}
        )


def test_a_typo_inside_a_point_group_list_is_rejected():
    with pytest.raises(ValidationError, match="initail"):
        burgers_like(
            residuals={
                "pde": ResidualSpec(
                    kind="burgers1d", points=["interior", "initail"]
                )
            }
        )


def test_a_bare_string_and_a_single_element_list_are_the_same_thing():
    """YAML should stay writable as ``points: interior`` for the common case,
    without that being a different experiment from ``points: [interior]``."""
    bare = burgers_like(
        residuals={"pde": ResidualSpec(kind="burgers1d", points="interior")},
        weighting=WeightingSpec(kind="mean"),
    )
    listed = burgers_like(
        residuals={"pde": ResidualSpec(kind="burgers1d", points=["interior"])},
        weighting=WeightingSpec(kind="mean"),
    )
    assert bare.residuals["pde"].points == ("interior",)
    assert bare.identity_hash() == listed.identity_hash()


def test_a_residual_on_several_point_groups_validates():
    """A PDE residual holds on the closed domain, not just the interior —
    enforcing it on the interior alone costs ~6x in rel-L2 on Burgers, silently
    and while *lowering* the loss."""
    cfg = burgers_like(
        residuals={
            "pde": ResidualSpec(
                kind="burgers1d", points=["interior", "initial"]
            ),
            "ic": ResidualSpec(kind="dirichlet", points="initial"),
        }
    )
    assert cfg.residuals["pde"].points == ("interior", "initial")


def test_a_residual_with_no_point_groups_is_rejected():
    with pytest.raises(ValidationError, match="no point group"):
        burgers_like(residuals={"pde": ResidualSpec(kind="burgers1d", points=[])})


def test_the_point_groups_enter_the_config_hash():
    """Interior-only and closed-domain enforcement are different experiments
    with materially different accuracy; they must not share a hash."""
    interior_only = burgers_like(
        residuals={"pde": ResidualSpec(kind="burgers1d", points=["interior"])},
        weighting=WeightingSpec(kind="mean"),
    )
    closed = burgers_like(
        residuals={
            "pde": ResidualSpec(kind="burgers1d", points=["interior", "initial"])
        },
        weighting=WeightingSpec(kind="mean"),
    )
    assert interior_only.identity_hash() != closed.identity_hash()


def test_a_coefficient_on_a_misspelled_term_is_rejected():
    """Silently doing nothing is the worst possible outcome for a weight.

    A run with ``coefficients={"ic_": 10.0}`` trains happily at weight 1.0 and
    reports a config claiming 10.0 — an unfalsifiable result.
    """
    with pytest.raises(ValidationError, match="not residuals"):
        burgers_like(weighting=WeightingSpec(coefficients={"ic_": 10.0}))


def test_an_unused_point_group_is_allowed():
    """Sampling more groups than the residuals consume is legitimate.

    An evaluation-only grid, or a group a later stage's residuals pick up.
    Rejecting it would make staged configs awkward for no safety gain.
    """
    cfg = burgers_like(
        sampling=SamplingSpec(
            points={
                "interior": PointSetSpec(region="interior", n=2540),
                "initial": PointSetSpec(region="initial", n=80),
                "eval_grid": PointSetSpec(region="interior", n=10_000),
            }
        )
    )
    assert "eval_grid" in cfg.sampling.points


# -- these fields are part of the experimental condition ----------------------


def test_the_volatile_axes_enter_the_config_hash():
    """A different architecture is a different experiment, not a different run.

    If these were excluded, two architectures would share a config hash and the
    §8 "median + IQR over >=5 seeds" groupby would silently merge them.
    """
    base = burgers_like()
    wider = burgers_like(nets={"u": NetSpec(inputs=2, outputs=1, width=64, depth=3)})
    more_points = burgers_like(
        sampling=SamplingSpec(
            points={
                "interior": PointSetSpec(region="interior", n=5000),
                "initial": PointSetSpec(region="initial", n=80),
            }
        )
    )
    reweighted = burgers_like(weighting=WeightingSpec(coefficients={"ic": 100.0}))

    hashes = {
        base.identity_hash(),
        wider.identity_hash(),
        more_points.identity_hash(),
        reweighted.identity_hash(),
    }
    assert len(hashes) == 4


def test_seed_still_does_not_enter_the_hash_once_the_axes_are_declared():
    assert burgers_like(seed=1).identity_hash() == burgers_like(seed=2).identity_hash()


# -- strictness ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (NetSpec, {"inputs": 2, "widht": 20}),
        (ResidualSpec, {"kind": "burgers1d", "pionts": "interior"}),
        (PointSetSpec, {"n": 100, "stratgey": "pseudo"}),
        (WeightingSpec, {"kind": "mean", "coeffs": {}}),
        (ProblemSpec, {"name": "burgers1d", "opts": {}}),
        (ParamSpec, {"init": 0.1, "trainble": True}),
    ],
)
def test_every_new_spec_rejects_a_typo(model, kwargs):
    """Each of these is parsed from hand-written YAML, where an unknown key is a
    hyperparameter the author believes is in effect and which is not."""
    with pytest.raises(ValidationError):
        model(**kwargs)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (NetSpec, {"inputs": 0}),
        (NetSpec, {"inputs": 2, "width": 0}),
        (NetSpec, {"inputs": 2, "depth": -1}),
        (PointSetSpec, {"n": 0}),
        (PointSetSpec, {"n": 100, "region": "edge"}),
    ],
)
def test_the_search_space_bounds_are_enforced(model, kwargs):
    """These bounds double as the metaheuristic's search space (DESIGN.md §9),
    so a candidate outside them must fail loudly rather than train nonsense."""
    with pytest.raises(ValidationError):
        model(**kwargs)


# -- YAML is the only supported entry point -----------------------------------


def test_a_declared_config_round_trips_through_yaml(tmp_path):
    """No hyperparameter is ever a Python literal (CLAUDE.md rule 4), so the
    disk format has to carry every one of these fields losslessly — including
    the hash, which is what a result row joins on."""
    cfg = burgers_like()
    path = tmp_path / "cfg.yaml"
    dump_config(cfg, path)
    reloaded = load_config(path)

    assert reloaded == cfg
    assert reloaded.identity_hash() == cfg.identity_hash()


def test_shape_survives_the_yaml_round_trip(tmp_path):
    """``ParamSpec.shape`` is the one tuple in the schema; YAML has no tuple, so
    it lands as a list and must validate back to a tuple or the hash drifts
    between a fresh config and a reloaded one."""
    cfg = burgers_like(
        extra_params={"nu": ParamSpec(init=0.01, shape=(3,), trainable=True)}
    )
    path = tmp_path / "cfg.yaml"
    dump_config(cfg, path)
    reloaded = load_config(path)

    assert reloaded.extra_params["nu"].shape == (3,)
    assert reloaded.identity_hash() == cfg.identity_hash()
