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
"""Shared machinery for the built-in feature selectors."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.table import FeatureTable
from habit.spec.specs import Spec

__all__ = ["outcome_series", "restrict_table", "FittedSelectorBase"]


def outcome_series(table: FeatureTable, *, owner: str) -> pd.Series:
    """
    Return the table's outcome column as a Series.

    Args:
        table: Table expected to carry an outcome column.
        owner: Human-readable selector name for the error message.

    Returns:
        The outcome column aligned to the table's rows.

    Raises:
        HABITAPIError: If the table has no outcome column (supervised
            selectors cannot fit without one).
    """
    if table.outcome_column is None:
        raise HABITAPIError(
            f"{owner} is supervised and requires a table with an outcome "
            "column; the table passed declares none."
        )
    return table.frame[table.outcome_column]


def restrict_table(
    table: FeatureTable,
    columns: Tuple[str, ...],
    spec: Spec,
) -> FeatureTable:
    """
    Build a new table restricted to the selected feature columns.

    Identifier and outcome columns and the row order are inherited from
    ``table``; provenance chains back to the input so a selected table still
    traces to the raw images.

    Args:
        table: Source table.
        columns: Feature columns to keep (must exist in ``table``).
        spec: The selector's specification (provenance fingerprint).

    Returns:
        A new table whose feature block is exactly ``columns``.
    """
    non_feature = [
        column for column in table.frame.columns
        if column not in table.feature_columns
    ]
    frame = table.frame[non_feature + list(columns)].copy()
    if table.provenance is not None:
        provenance = table.provenance.derive(
            produced_by=f"feature_selector.{spec.name}",
            spec_fingerprint=spec.fingerprint(),
        )
    else:
        provenance = Provenance(
            produced_by=f"feature_selector.{spec.name}",
            spec_fingerprint=spec.fingerprint(),
            inputs=(),
            software=software_fingerprint(),
        )
    return FeatureTable(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=columns,
        outcome_column=table.outcome_column,
        provenance=provenance,
    )


class FittedSelectorBase:
    """
    Shared fitted-state bookkeeping for the built-in selectors.

    ``fit`` implementations compute the selected names and call
    :meth:`_remember_selection`; ``transform`` restricts any later table to
    exactly those columns, so prediction data is reduced with the TRAINING
    selection and never re-selected. Subclasses set ``_spec_name``.
    """

    _spec_name: str = ""

    def __init__(self) -> None:
        self._selected_columns: Optional[Tuple[str, ...]] = None

    @property
    def selected_columns_(self) -> Tuple[str, ...]:
        """Return the feature columns selected at fit time (sklearn-style)."""
        if self._selected_columns is None:
            raise HABITAPIError(
                f"feature_selector.{self._spec_name} has not been fitted."
            )
        return self._selected_columns

    def _remember_selection(
        self,
        table: FeatureTable,
        selected: Sequence[str],
    ) -> None:
        """
        Store the selection in the TABLE's column order.

        Selectors report names in ranked order; downstream consumers need a
        stable schema, so the stored order follows the fit table's declared
        feature columns. Names the selector produced that are not feature
        columns of the fit table are dropped (mirroring the v0.1 registry's
        candidate-restriction rule).
        """
        selected_set = set(selected)
        self._selected_columns = tuple(
            column for column in table.feature_columns if column in selected_set
        )

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Restrict a table to the fitted feature subset."""
        selected = self.selected_columns_
        missing = [
            column for column in selected if column not in table.feature_columns
        ]
        if missing:
            raise HABITAPIError(
                f"feature_selector.{self._spec_name} selected {list(selected)} "
                f"at fit time but the table to transform does not declare "
                f"{missing} as feature columns. Apply the same upstream "
                "pipeline steps as at fit time, or re-fit."
            )
        return restrict_table(table, selected, self.spec)

    @property
    def spec(self) -> Spec:  # pragma: no cover - subclasses override
        raise NotImplementedError
