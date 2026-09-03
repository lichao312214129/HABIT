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

from habit.exceptions import HABITAPIError
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.table import FeatureTable
from habit.pipeline.outcome_access import outcome_series
from habit.spec.specs import Spec
from habit.utils.estimator_utils import ComponentParamsMixin

# ``outcome_series`` now lives in habit.pipeline.outcome_access, next to the
# survival accessors, and is re-exported here because the selectors and the
# classifiers have always imported it from this module.
__all__ = ["outcome_series", "restrict_table", "FittedSelectorBase"]


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
        outcome=table.outcome,
        provenance=provenance,
    )


class FittedSelectorBase(ComponentParamsMixin):
    """
    Shared fitted-state bookkeeping for the built-in selectors.

    ``fit`` implementations compute the selected names and call
    :meth:`_remember_selection`; ``transform`` restricts any later table to
    exactly those columns, so prediction data is reduced with the TRAINING
    selection and never re-selected. Subclasses set ``_spec_name``.

    :class:`~habit.utils.estimator_utils.ComponentParamsMixin` adds the
    scikit-learn ``get_params``/``set_params``/``clone`` protocol. Selectors
    keep one private attribute per constructor parameter
    (``self._threshold`` for ``threshold``), which is the second resolution
    rule of the mixin, so a nested grid such as
    ``select__component__threshold`` reaches the value ``spec.params``
    publishes.
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
