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
"""Contract tests for the four endpoint declarations."""

from __future__ import annotations

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


def _survival_frame() -> pd.DataFrame:
    """Two-subject frame with a follow-up time and an event indicator."""
    return pd.DataFrame(
        {
            "subject": ["a", "b"],
            "f1": [1.0, 2.0],
            "os_time": [12.0, 30.0],
            "os_event": [1, 0],
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected_task", "expected_columns"),
    [
        (BinaryOutcome("y"), "binary", ("y",)),
        (MulticlassOutcome("grade"), "multiclass", ("grade",)),
        (ContinuousOutcome("volume"), "continuous", ("volume",)),
        (
            SurvivalOutcome(time_column="t", event_column="e"),
            "survival",
            ("t", "e"),
        ),
    ],
)
def test_endpoints_declare_task_and_columns(
    outcome: Outcome,
    expected_task: str,
    expected_columns: tuple,
) -> None:
    """Every endpoint reports a task string and the columns it occupies."""
    assert outcome.task == expected_task
    assert outcome.columns == expected_columns
    # Structural typing is what lets third-party endpoints participate.
    assert isinstance(outcome, Outcome)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda: BinaryOutcome(""),
        lambda: MulticlassOutcome("   "),
        lambda: ContinuousOutcome(None),
        lambda: SurvivalOutcome(time_column="t", event_column=""),
    ],
)
def test_empty_column_names_are_rejected(factory) -> None:
    """A blank or non-string column name fails at declaration time."""
    with pytest.raises(HABITAPIError):
        factory()


@pytest.mark.unit
def test_survival_requires_two_distinct_columns() -> None:
    """One column cannot serve as both the time and the event indicator."""
    with pytest.raises(HABITAPIError, match="different columns"):
        SurvivalOutcome(time_column="t", event_column="t")


@pytest.mark.unit
def test_multiclass_class_order_is_validated_and_frozen() -> None:
    """Declared class order is normalised to a tuple and must be unique."""
    outcome = MulticlassOutcome("grade", classes=["I", "II", "III"])
    assert outcome.classes == ("I", "II", "III")
    with pytest.raises(HABITAPIError, match="duplicates"):
        MulticlassOutcome("grade", classes=("I", "I"))
    with pytest.raises(HABITAPIError, match="at least two"):
        MulticlassOutcome("grade", classes=("I",))


@pytest.mark.unit
def test_binary_positive_mask_follows_declared_label() -> None:
    """The positive class is the declared value, never inferred."""
    frame = pd.DataFrame({"y": ["responder", "non-responder"]})
    outcome = BinaryOutcome("y", positive_label="responder")
    assert outcome.positive_mask(frame).tolist() == [True, False]


@pytest.mark.unit
def test_survival_accessors_read_time_and_event() -> None:
    """Times come back as-is; the event mask honours ``event_value``."""
    frame = _survival_frame()
    outcome = SurvivalOutcome(time_column="os_time", event_column="os_event")
    assert outcome.times(frame).tolist() == [12.0, 30.0]
    assert outcome.event_mask(frame).tolist() == [True, False]


@pytest.mark.unit
def test_feature_table_validates_every_endpoint_column() -> None:
    """A survival endpoint whose event column is absent fails fast."""
    frame = _survival_frame().drop(columns=["os_event"])
    with pytest.raises(HABITAPIError, match="os_event"):
        FeatureTable(
            frame=frame,
            id_columns=("subject",),
            feature_columns=("f1",),
            outcome=SurvivalOutcome(
                time_column="os_time", event_column="os_event"
            ),
        )


@pytest.mark.unit
def test_outcome_column_shortcut_serves_one_column_endpoints() -> None:
    """The compatibility property answers for single-column endpoints."""
    frame = pd.DataFrame({"subject": ["a"], "f1": [1.0], "y": [1]})
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1",),
        outcome=BinaryOutcome("y"),
    )
    assert table.outcome_column == "y"
    assert (
        FeatureTable(
            frame=frame, id_columns=("subject",), feature_columns=("f1",)
        ).outcome_column
        is None
    )


@pytest.mark.unit
def test_outcome_column_refuses_to_halve_a_survival_endpoint() -> None:
    """Returning the time column alone would look like a label; it raises."""
    table = FeatureTable(
        frame=_survival_frame(),
        id_columns=("subject",),
        feature_columns=("f1",),
        outcome=SurvivalOutcome(time_column="os_time", event_column="os_event"),
    )
    with pytest.raises(HABITAPIError, match="spans"):
        _ = table.outcome_column


@pytest.mark.unit
def test_join_keeps_the_endpoint_of_whichever_side_declares_one() -> None:
    """Joining a feature-only table onto an endpoint table preserves it."""
    left = FeatureTable(
        frame=_survival_frame(),
        id_columns=("subject",),
        feature_columns=("f1",),
        outcome=SurvivalOutcome(time_column="os_time", event_column="os_event"),
    )
    right = FeatureTable(
        frame=pd.DataFrame({"subject": ["a", "b"], "f2": [5.0, 6.0]}),
        id_columns=("subject",),
        feature_columns=("f2",),
    )
    joined = left.join(right)
    assert joined.outcome is left.outcome
    assert joined.feature_columns == ("f1", "f2")
