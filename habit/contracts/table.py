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
"""Feature table contract with explicit column semantics.

v0.1 passed bare DataFrames whose column roles were conventions spread across
the codebase. Making the roles explicit removes a whole class of leakage
bugs, e.g. an identifier accidentally entering the model matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.provenance import Provenance

__all__ = ["FeatureTable"]


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """
    Feature table with explicit column semantics.

    Attributes:
        frame: The underlying table.
        id_columns: Columns identifying the unit of analysis, e.g.
            ``subject``.
        feature_columns: Columns usable as model inputs.
        outcome_column: Clinical-outcome column when present. Named
            ``outcome`` because that is the medical-research term for the
            predicted endpoint.
        provenance: How this table was produced.
    """

    frame: pd.DataFrame
    id_columns: Tuple[str, ...]
    feature_columns: Tuple[str, ...]
    outcome_column: Optional[str] = None
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        """Validate that every declared column exists in the frame."""
        missing = [
            column
            for column in (
                *self.id_columns,
                *self.feature_columns,
                *(() if self.outcome_column is None else (self.outcome_column,)),
            )
            if column not in self.frame.columns
        ]
        if missing:
            raise HABITAPIError(
                f"FeatureTable columns missing from frame: {missing}."
            )
        object.__setattr__(self, "id_columns", tuple(self.id_columns))
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))

    def feature_matrix(self) -> pd.DataFrame:
        """
        Return only the model-input columns, indexed by the id columns.

        Named ``feature_matrix`` rather than ``features`` so it cannot be
        confused with running feature extraction, and because it returns a
        matrix-like frame rather than a list of features.

        Returns:
            A frame with the id columns as (possibly multi-) index and only
            the declared feature columns as data.
        """
        return self.frame.set_index(list(self.id_columns))[
            list(self.feature_columns)
        ]

    def join(self, other: "FeatureTable") -> "FeatureTable":
        """
        Join another table on the shared id columns.

        Args:
            other: Table to merge; must share ``id_columns``.

        Returns:
            A new table whose provenance records both inputs.

        Raises:
            HABITAPIError: If the id columns do not match.
        """
        if tuple(self.id_columns) != tuple(other.id_columns):
            raise HABITAPIError(
                "FeatureTable.join requires identical id_columns; got "
                f"{self.id_columns} and {other.id_columns}."
            )
        overlap = set(self.feature_columns) & set(other.feature_columns)
        if overlap:
            raise HABITAPIError(
                f"FeatureTable.join would duplicate feature columns: "
                f"{sorted(overlap)}."
            )
        merged = self.frame.merge(
            other.frame,
            on=list(self.id_columns),
            how="inner",
            validate="one_to_one",
        )
        provenance: Optional[Provenance] = None
        if self.provenance is not None and other.provenance is not None:
            provenance = Provenance(
                produced_by="feature_table.join",
                spec_fingerprint="",
                inputs=(self.provenance, other.provenance),
                software=dict(self.provenance.software),
            )
        elif self.provenance is not None:
            provenance = self.provenance
        return FeatureTable(
            frame=merged,
            id_columns=self.id_columns,
            feature_columns=(*self.feature_columns, *other.feature_columns),
            outcome_column=self.outcome_column or other.outcome_column,
            provenance=provenance,
        )
