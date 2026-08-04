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
"""Typed access to a feature table's endpoint.

Every component that consumes an endpoint goes through this module rather
than reading ``table.frame[...]`` itself, for two reasons. First, a component
states the endpoint families it supports ONCE, at the top of ``fit``, and gets
a precise error for everything else -- instead of a chi-squared test silently
running on follow-up times, or a Cox model failing three frames deep because
the event column was never there. Second, the survival accessors centralise
the data validation (censoring coding, non-negative times, at least one
observed event) that every survival estimator and metric would otherwise
re-implement slightly differently.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.outcome import Outcome, SurvivalOutcome
from habit.contracts.table import FeatureTable

__all__ = [
    "SINGLE_COLUMN_TASKS",
    "require_outcome",
    "outcome_series",
    "survival_target",
    "structured_survival_array",
]

#: Endpoint families whose declaration occupies exactly one frame column.
SINGLE_COLUMN_TASKS: Tuple[str, ...] = ("binary", "multiclass", "continuous")


def require_outcome(
    table: FeatureTable,
    *,
    owner: str,
    tasks: Optional[Sequence[str]] = None,
) -> Outcome:
    """
    Return the table's endpoint, checking it belongs to a supported family.

    Args:
        table: Table expected to declare an endpoint.
        owner: Human-readable component name for the error message, e.g.
            ``"feature_selector.anova"``.
        tasks: Endpoint families the caller supports, as ``task`` strings.
            ``None`` accepts any declared endpoint.

    Returns:
        The declared endpoint.

    Raises:
        HABITAPIError: If the table declares no endpoint, or declares one
            outside ``tasks``.
    """
    outcome = table.outcome
    if outcome is None:
        raise HABITAPIError(
            f"{owner} is supervised and requires a table with an outcome "
            "column; the table passed declares none."
        )
    if tasks is not None and outcome.task not in tuple(tasks):
        raise HABITAPIError(
            f"{owner} supports {list(tasks)} endpoints but the table declares "
            f"a {outcome.task!r} endpoint. Use a component built for "
            f"{outcome.task!r} data."
        )
    return outcome


def outcome_series(table: FeatureTable, *, owner: str) -> pd.Series:
    """
    Return a one-column endpoint as a Series.

    Serves the binary, multiclass and continuous families, whose endpoint is
    a single column of values. Survival endpoints are rejected here on
    purpose: they have no single "y" vector, and quietly answering with the
    time column would train classifiers on follow-up duration.

    Args:
        table: Table expected to carry a one-column endpoint.
        owner: Human-readable component name for the error message.

    Returns:
        The endpoint column aligned to the table's rows.

    Raises:
        HABITAPIError: If the table declares no endpoint, or declares a
            multi-column one such as survival.
    """
    outcome = require_outcome(table, owner=owner, tasks=SINGLE_COLUMN_TASKS)
    return table.frame[outcome.columns[0]]


def survival_target(
    table: FeatureTable,
    *,
    owner: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return the follow-up times and the observed-event mask.

    Args:
        table: Table expected to declare a survival endpoint.
        owner: Human-readable component name for the error message.

    Returns:
        Tuple ``(time, event)``: the follow-up durations as floats and a
        boolean mask that is true for observed events and false for
        right-censored rows. Both are aligned to the table's rows.

    Raises:
        HABITAPIError: If the endpoint is not survival, if any follow-up time
            is missing or negative, or if no row records an observed event
            (every survival estimate and metric is undefined on a fully
            censored table).
    """
    outcome = require_outcome(table, owner=owner, tasks=("survival",))
    if not isinstance(outcome, SurvivalOutcome):
        raise HABITAPIError(
            f"{owner} received an endpoint declaring task 'survival' that is "
            f"not a SurvivalOutcome: {type(outcome).__name__}."
        )
    raw_time = outcome.times(table.frame)
    time = pd.to_numeric(raw_time, errors="coerce").astype(float)
    if time.isna().any():
        offending = int(time.isna().sum())
        raise HABITAPIError(
            f"{owner}: column {outcome.time_column!r} has {offending} "
            "non-numeric or missing follow-up time(s); survival analysis "
            "needs a duration for every row."
        )
    if (time < 0).any():
        raise HABITAPIError(
            f"{owner}: column {outcome.time_column!r} contains negative "
            "follow-up times."
        )
    event = outcome.event_mask(table.frame).astype(bool)
    if not event.any():
        raise HABITAPIError(
            f"{owner}: no row in column {outcome.event_column!r} equals "
            f"{outcome.event_value!r}, so the table is fully censored and "
            "carries no survival information. Check the event coding declared "
            "by SurvivalOutcome.event_value."
        )
    return time, event


def structured_survival_array(table: FeatureTable, *, owner: str) -> np.ndarray:
    """
    Return the survival endpoint in scikit-survival's structured layout.

    scikit-survival estimators and metrics take ``y`` as a structured array
    with a boolean event field followed by a float time field (what
    ``sksurv.util.Surv.from_arrays`` builds). Producing it here keeps
    scikit-survival an OPTIONAL dependency: the array is assembled with numpy
    alone, so the contract holds even where scikit-survival is not installed.

    Args:
        table: Table expected to declare a survival endpoint.
        owner: Human-readable component name for the error message.

    Returns:
        Structured array of dtype ``[("event", bool), ("time", float)]`` with
        one record per table row, in table order.

    Raises:
        HABITAPIError: Same conditions as :func:`survival_target`.
    """
    time, event = survival_target(table, owner=owner)
    target = np.empty(
        len(time),
        dtype=[("event", np.bool_), ("time", np.float64)],
    )
    target["event"] = event.to_numpy()
    target["time"] = time.to_numpy()
    return target
