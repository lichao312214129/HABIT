# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for the typed endpoint accessors."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import (
    BinaryOutcome,
    ContinuousOutcome,
    FeatureTable,
    MulticlassOutcome,
    Outcome,
    SurvivalOutcome,
)
from habit.domain.outcome_access import (
    outcome_series,
    require_outcome,
    structured_survival_array,
    survival_target,
)


def _table(
    outcome: Optional[Outcome],
    *,
    time: Any = (12.0, 30.0, 5.0),
    event: Any = (1, 0, 1),
) -> FeatureTable:
    """Three-subject table carrying every endpoint column the tests need."""
    frame = pd.DataFrame(
        {
            "subject": ["a", "b", "c"],
            "f1": [1.0, 2.0, 3.0],
            "y": [0, 1, 1],
            "grade": ["I", "II", "III"],
            "volume": [10.5, 20.25, 30.0],
            "os_time": list(time),
            "os_event": list(event),
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1",),
        outcome=outcome,
    )


@pytest.mark.unit
def test_require_outcome_rejects_a_table_without_an_endpoint() -> None:
    """The message keeps naming the caller, as the selectors relied on."""
    with pytest.raises(HABITAPIError, match="is supervised and requires"):
        require_outcome(_table(None), owner="feature_selector.anova")


@pytest.mark.unit
def test_require_outcome_rejects_an_unsupported_family() -> None:
    """A classification-only component refuses a survival table by name."""
    table = _table(SurvivalOutcome(time_column="os_time", event_column="os_event"))
    with pytest.raises(HABITAPIError, match="'survival'"):
        require_outcome(
            table, owner="feature_selector.chi2", tasks=("binary", "multiclass")
        )


@pytest.mark.unit
def test_require_outcome_accepts_any_family_by_default() -> None:
    """Without a ``tasks`` filter every declared endpoint passes through."""
    outcome = ContinuousOutcome("volume")
    assert require_outcome(_table(outcome), owner="probe") is outcome


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (BinaryOutcome("y"), [0, 1, 1]),
        (MulticlassOutcome("grade"), ["I", "II", "III"]),
        (ContinuousOutcome("volume"), [10.5, 20.25, 30.0]),
    ],
)
def test_outcome_series_serves_every_one_column_family(
    outcome: Outcome,
    expected: list,
) -> None:
    """Binary, multiclass and continuous endpoints all yield one vector."""
    assert outcome_series(_table(outcome), owner="probe").tolist() == expected


@pytest.mark.unit
def test_outcome_series_refuses_survival_instead_of_returning_time() -> None:
    """Silently answering with follow-up time would train on the wrong y."""
    table = _table(SurvivalOutcome(time_column="os_time", event_column="os_event"))
    with pytest.raises(HABITAPIError, match="'survival'"):
        outcome_series(table, owner="classifier.logistic")


@pytest.mark.unit
def test_survival_target_returns_times_and_the_event_mask() -> None:
    """Times become floats; the mask is true only for observed events."""
    table = _table(SurvivalOutcome(time_column="os_time", event_column="os_event"))
    time, event = survival_target(table, owner="probe")
    assert time.tolist() == [12.0, 30.0, 5.0]
    assert time.dtype == np.float64
    assert event.tolist() == [True, False, True]


@pytest.mark.unit
def test_survival_target_honours_a_string_event_coding() -> None:
    """``event_value`` covers Dead/Alive style columns without recoding."""
    table = _table(
        SurvivalOutcome(
            time_column="os_time", event_column="os_event", event_value="Dead"
        ),
        event=("Dead", "Alive", "Dead"),
    )
    _, event = survival_target(table, owner="probe")
    assert event.tolist() == [True, False, True]


@pytest.mark.unit
def test_survival_target_rejects_a_non_survival_endpoint() -> None:
    """A survival accessor on a binary table fails at the call site."""
    with pytest.raises(HABITAPIError, match="'binary'"):
        survival_target(_table(BinaryOutcome("y")), owner="survival_model.cox")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("time", "message"),
    [
        ((12.0, np.nan, 5.0), "missing follow-up time"),
        ((12.0, "not-a-number", 5.0), "missing follow-up time"),
        ((12.0, -1.0, 5.0), "negative"),
    ],
)
def test_survival_target_validates_the_time_column(time: Any, message: str) -> None:
    """Missing, non-numeric and negative durations are all caught here."""
    table = _table(
        SurvivalOutcome(time_column="os_time", event_column="os_event"),
        time=time,
    )
    with pytest.raises(HABITAPIError, match=message):
        survival_target(table, owner="probe")


@pytest.mark.unit
def test_survival_target_rejects_a_fully_censored_table() -> None:
    """No observed event means no survival information, and hints at coding."""
    table = _table(
        SurvivalOutcome(time_column="os_time", event_column="os_event"),
        event=(0, 0, 0),
    )
    with pytest.raises(HABITAPIError, match="fully censored"):
        survival_target(table, owner="probe")


@pytest.mark.unit
def test_structured_survival_array_matches_the_sksurv_layout() -> None:
    """Event-then-time structured dtype, one record per row, in table order."""
    table = _table(SurvivalOutcome(time_column="os_time", event_column="os_event"))
    target = structured_survival_array(table, owner="probe")
    assert target.dtype.names == ("event", "time")
    assert target["event"].dtype == np.bool_
    assert target["time"].tolist() == [12.0, 30.0, 5.0]
    assert target["event"].tolist() == [True, False, True]


@pytest.mark.unit
def test_structured_survival_array_is_accepted_by_scikit_survival() -> None:
    """The hand-built array equals what ``Surv.from_arrays`` produces."""
    sksurv_util = pytest.importorskip("sksurv.util")
    table = _table(SurvivalOutcome(time_column="os_time", event_column="os_event"))
    time, event = survival_target(table, owner="probe")
    expected = sksurv_util.Surv.from_arrays(
        event=event.to_numpy(), time=time.to_numpy()
    )
    target = structured_survival_array(table, owner="probe")
    assert target.dtype == expected.dtype
    assert np.array_equal(target, expected)
