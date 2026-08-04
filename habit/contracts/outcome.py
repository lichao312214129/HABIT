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
"""Endpoint declarations: WHICH columns carry the outcome, and of WHAT kind.

A single ``outcome_column: str`` can express "the label lives here" and
nothing else. That is enough for binary classification and structurally
excludes the two endpoint families clinical imaging research relies on most:

- **Survival / time-to-event** needs TWO columns (follow-up time and event
  indicator) that are only meaningful together. Splitting them across two
  independent fields would allow a table declaring a time without an event,
  which is not a survival endpoint at all.
- **Continuous regression** needs the same single column as binary
  classification but a DIFFERENT downstream treatment; a bare column name
  cannot tell a metric whether to compute AUC or R-squared, so every
  consumer would have to guess from the dtype.

Declaring the endpoint as an object rather than a column name makes the
research question explicit at the type level, and lets each consumer state
the endpoint families it supports instead of failing deep inside an
algorithm on data it never handled.

Extension rule: dispatch on the ``task`` STRING, never on ``isinstance``
against a closed union. A group adding, say, a competing-risks endpoint
declares ``task = "competing_risks"`` and every existing guard rejects it
with a precise "unsupported endpoint" message, rather than crashing or --
far worse -- silently treating it as one of the built-in four.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import pandas as pd

from habit.exceptions import HABITAPIError

__all__ = [
    "Outcome",
    "BinaryOutcome",
    "MulticlassOutcome",
    "ContinuousOutcome",
    "SurvivalOutcome",
    "outcome_to_dict",
    "outcome_from_dict",
]


def _validated_column(value: Any, *, field: str, owner: str) -> str:
    """
    Validate one column-name argument of an outcome declaration.

    Args:
        value: The candidate column name.
        field: Attribute name, used in the error message.
        owner: Outcome class name, used in the error message.

    Returns:
        The column name unchanged.

    Raises:
        HABITAPIError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise HABITAPIError(
            f"{owner}.{field} must be a non-empty column name; got {value!r}."
        )
    return value


@runtime_checkable
class Outcome(Protocol):
    """
    A declared study endpoint.

    Implementations are frozen value objects carrying only column names and
    the coding of those columns -- never data, so an endpoint declaration can
    be recorded in provenance, compared, and serialised.

    Attributes:
        task: Endpoint family identifier used for dispatch. Built-in values
            are ``"binary"``, ``"multiclass"``, ``"continuous"`` and
            ``"survival"``; third-party endpoints declare their own. Declared
            as a class-level constant: implementations are frozen value
            objects whose task never varies per instance.
    """

    task: ClassVar[str]

    @property
    def columns(self) -> Tuple[str, ...]:
        """Return every frame column this endpoint occupies."""


@dataclass(frozen=True)
class BinaryOutcome:
    """
    Two-class endpoint, e.g. treatment response or 2-year recurrence.

    Attributes:
        column: Column holding the class label.
        positive_label: Value denoting the POSITIVE class. Made explicit
            because sensitivity, PPV, ROC and decision-curve analysis are all
            defined relative to it, and inferring it from the data (largest
            label? most frequent? sorted last?) silently flips those metrics
            whenever the coding changes.
    """

    column: str
    positive_label: Any = 1
    task: ClassVar[str] = "binary"

    def __post_init__(self) -> None:
        """Validate the column name."""
        _validated_column(self.column, field="column", owner="BinaryOutcome")

    @property
    def columns(self) -> Tuple[str, ...]:
        """Return the single label column."""
        return (self.column,)

    def positive_mask(self, frame: pd.DataFrame) -> pd.Series:
        """
        Return a boolean mask marking the positive class.

        Args:
            frame: Frame carrying :attr:`column`.

        Returns:
            Boolean Series aligned to ``frame``, true where the label equals
            :attr:`positive_label`.
        """
        return frame[self.column] == self.positive_label


@dataclass(frozen=True)
class MulticlassOutcome:
    """
    Endpoint with three or more mutually exclusive classes.

    Attributes:
        column: Column holding the class label.
        classes: Optional declared class ORDER. Supplying it pins the column
            order of probability outputs and the row/column order of
            confusion matrices, so a class absent from one validation fold
            cannot shift the meaning of a column; ``None`` leaves the order
            to be derived from the observed labels.
    """

    column: str
    classes: Optional[Tuple[Any, ...]] = None
    task: ClassVar[str] = "multiclass"

    def __post_init__(self) -> None:
        """Validate the column name and the declared class order."""
        _validated_column(self.column, field="column", owner="MulticlassOutcome")
        if self.classes is None:
            return
        declared = tuple(self.classes)
        if len(declared) < 2:
            raise HABITAPIError(
                "MulticlassOutcome.classes must declare at least two classes; "
                f"got {declared!r}."
            )
        if len(set(declared)) != len(declared):
            raise HABITAPIError(
                f"MulticlassOutcome.classes contains duplicates: {declared!r}."
            )
        object.__setattr__(self, "classes", declared)

    @property
    def columns(self) -> Tuple[str, ...]:
        """Return the single label column."""
        return (self.column,)


@dataclass(frozen=True)
class ContinuousOutcome:
    """
    Continuous endpoint for regression, e.g. tumour volume change or a score.

    Attributes:
        column: Column holding the continuous response.
    """

    column: str
    task: ClassVar[str] = "continuous"

    def __post_init__(self) -> None:
        """Validate the column name."""
        _validated_column(self.column, field="column", owner="ContinuousOutcome")

    @property
    def columns(self) -> Tuple[str, ...]:
        """Return the single response column."""
        return (self.column,)


@dataclass(frozen=True)
class SurvivalOutcome:
    """
    Right-censored time-to-event endpoint, e.g. overall or progression-free
    survival.

    The two columns are declared TOGETHER because neither is interpretable
    alone: a follow-up time without an event indicator cannot distinguish a
    death at 10 months from a patient still alive at last contact, and every
    survival estimator, metric and plot needs both.

    Attributes:
        time_column: Follow-up duration until the event or until censoring.
            Units are the study's own (months, days); HABIT never converts
            them, it only requires that they be consistent within a table.
        event_column: Event indicator. Rows where it equals
            :attr:`event_value` experienced the event; all others are treated
            as right-censored.
        event_value: Value in :attr:`event_column` denoting an observed
            event. Explicit because both ``1``/``0`` and ``"Dead"``/``"Alive"``
            codings are common in clinical tables, and guessing wrong inverts
            the entire analysis.
    """

    time_column: str
    event_column: str
    event_value: Any = 1
    task: ClassVar[str] = "survival"

    def __post_init__(self) -> None:
        """Validate the column names and that they are distinct."""
        _validated_column(
            self.time_column, field="time_column", owner="SurvivalOutcome"
        )
        _validated_column(
            self.event_column, field="event_column", owner="SurvivalOutcome"
        )
        if self.time_column == self.event_column:
            raise HABITAPIError(
                "SurvivalOutcome.time_column and event_column must be "
                f"different columns; both are {self.time_column!r}."
            )

    @property
    def columns(self) -> Tuple[str, ...]:
        """Return the time and event columns, in that order."""
        return (self.time_column, self.event_column)

    def times(self, frame: pd.DataFrame) -> pd.Series:
        """
        Return the follow-up durations.

        Args:
            frame: Frame carrying :attr:`time_column`.

        Returns:
            The follow-up durations aligned to ``frame``.
        """
        return frame[self.time_column]

    def event_mask(self, frame: pd.DataFrame) -> pd.Series:
        """
        Return a boolean mask marking rows with an OBSERVED event.

        Args:
            frame: Frame carrying :attr:`event_column`.

        Returns:
            Boolean Series aligned to ``frame``, true where the event was
            observed and false where the row is right-censored.
        """
        return frame[self.event_column] == self.event_value


# ---------------------------------------------------------------------------
# Serialisation: YAML/JSON round-trip for endpoint declarations
# ---------------------------------------------------------------------------

#: task string -> the concrete dataclass that implements it. Typed as
#: callables rather than ``type[Outcome]``: the ``Outcome`` protocol carries
#: the instance attribute ``column``, which a class object does not have, so
#: ``type[BinaryOutcome]`` cannot satisfy ``type[Outcome]`` directly.
_TASK_TO_TYPE: Dict[str, Callable[..., Outcome]] = {
    "binary": BinaryOutcome,
    "multiclass": MulticlassOutcome,
    "continuous": ContinuousOutcome,
    "survival": SurvivalOutcome,
}


def outcome_to_dict(outcome: Outcome) -> dict:
    """
    Serialise an endpoint declaration to a plain mapping for YAML/JSON.

    The ``task`` field is the discriminator :func:`outcome_from_dict` reads;
    the remaining keys mirror the dataclass fields, so the mapping is both
    human-writable and losslessly reversible.

    Args:
        outcome: The endpoint declaration to serialise.

    Returns:
        A dict with a ``task`` key plus the endpoint's fields.

    Raises:
        HABITAPIError: If the outcome's task is not a built-in one.
    """
    task = outcome.task
    if task == "binary":
        assert isinstance(outcome, BinaryOutcome)
        return {
            "task": task,
            "column": outcome.column,
            "positive_label": outcome.positive_label,
        }
    if task == "multiclass":
        assert isinstance(outcome, MulticlassOutcome)
        payload: Dict[str, Any] = {"task": task, "column": outcome.column}
        if outcome.classes is not None:
            payload["classes"] = list(outcome.classes)
        return payload
    if task == "continuous":
        assert isinstance(outcome, ContinuousOutcome)
        return {"task": task, "column": outcome.column}
    if task == "survival":
        assert isinstance(outcome, SurvivalOutcome)
        return {
            "task": task,
            "time_column": outcome.time_column,
            "event_column": outcome.event_column,
            "event_value": outcome.event_value,
        }
    raise HABITAPIError(
        f"outcome_to_dict cannot serialise endpoint task {task!r}; only the "
        "built-in four are supported. Extend the outcome module."
    )


def outcome_from_dict(payload: dict) -> Outcome:
    """
    Rebuild an endpoint declaration from its serialised mapping.

    Args:
        payload: Mapping produced by :func:`outcome_to_dict` (or written by
            hand in a YAML document). Must carry a ``task`` key.

    Returns:
        The corresponding :class:`Outcome` implementation.

    Raises:
        HABITAPIError: If the task is missing or not a built-in one.
    """
    if not isinstance(payload, dict):
        raise HABITAPIError(
            f"outcome_from_dict expects a mapping, got {type(payload).__name__}."
        )
    task = payload.get("task")
    # A missing or non-string task takes the same error path as an unknown
    # one; the isinstance guard also narrows the ``dict.get`` key for mypy.
    cls = _TASK_TO_TYPE.get(task) if isinstance(task, str) else None
    if cls is None:
        raise HABITAPIError(
            f"outcome_from_dict got endpoint task {task!r}; expected one of "
            f"{sorted(_TASK_TO_TYPE)}. Declare the endpoint in YAML as, e.g., "
            "task: survival, time_column: os_months, event_column: os_event."
        )
    kwargs = {k: v for k, v in payload.items() if k != "task"}
    if task == "multiclass" and kwargs.get("classes") is not None:
        kwargs["classes"] = tuple(kwargs["classes"])
    return cls(**kwargs)
