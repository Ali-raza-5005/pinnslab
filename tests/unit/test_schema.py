"""The results contract: what a run is allowed to write down."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pinnslab.registry.schema import MetricSchedule

pytestmark = pytest.mark.unit


def test_a_schedule_that_records_nothing_is_rejected():
    """An empty trace is a config typo, and it costs a whole sweep to discover."""
    with pytest.raises(ValidationError, match="can never record anything"):
        MetricSchedule(
            every=None, n_per_decade=None, record_first=False, record_last=False
        )


def test_endpoints_alone_are_a_legitimate_schedule():
    """Two points is a real choice for a 1e5-run sweep, not a mistake."""
    schedule = MetricSchedule(every=None, n_per_decade=None)
    assert schedule.should_record(0)
    assert schedule.should_record(4321, is_last=True)
    assert not schedule.should_record(4321)


def test_record_last_wins_wherever_the_schedule_landed():
    assert MetricSchedule(every=100).should_record(1337, is_last=True)
    assert not MetricSchedule(every=100, record_last=False).should_record(1337)


def test_config_models_still_reject_unknown_keys():
    """``ResultRow`` opting out of ``extra="forbid"`` must not loosen the rest.

    Everything parsed from hand-written YAML keeps it: there, an unknown key is a
    typo'd hyperparameter that would otherwise be silently ignored and cost a
    whole sweep. Rows are built in code, so they carry no such risk.
    """
    with pytest.raises(ValidationError):
        MetricSchedule(every=100, evry=10)
