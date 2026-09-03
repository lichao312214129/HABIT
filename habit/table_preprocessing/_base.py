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
"""Shared machinery for the built-in table preprocessors."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.table import FeatureTable
from habit.spec.specs import Spec

__all__ = ["fit_feature_block", "replace_feature_values"]


def fit_feature_block(
    table: FeatureTable,
    fit_columns: Tuple[str, ...],
    *,
    owner: str,
) -> pd.DataFrame:
    """
    Return the table's feature block after schema-drift validation.

    A fitted preprocessor remembers the feature columns it learned its state
    on; transforming a table that does not declare exactly those columns as
    features is the silent-schema-drift bug this guard turns into an explicit
    error.

    Args:
        table: Table whose feature block is requested.
        fit_columns: Feature columns the component was fitted on.
        owner: Human-readable component name for the error message.

    Returns:
        The block ``table.frame[list(fit_columns)]`` (row order preserved).

    Raises:
        HABITAPIError: If the table does not declare every fitted column as
            a feature column.
    """
    missing = [
        column for column in fit_columns if column not in table.feature_columns
    ]
    if missing:
        raise HABITAPIError(
            f"{owner} was fitted on feature columns {list(fit_columns)} but the "
            f"table to transform does not declare {missing} as feature columns. "
            "Apply the same upstream pipeline steps as at fit time, or re-fit."
        )
    return table.frame[list(fit_columns)]


def replace_feature_values(
    table: FeatureTable,
    values: pd.DataFrame,
    feature_columns: Tuple[str, ...],
    spec: Spec,
) -> FeatureTable:
    """
    Build a new table with transformed feature values.

    Identifier and outcome columns, row order and row count are inherited
    from ``table``; only the feature block changes. Provenance chains back to
    the input table so a processed table still traces to the raw images.

    Args:
        table: Source table providing id/outcome columns and row order.
        values: Transformed feature block, row-aligned to ``table.frame`` and
            carrying exactly ``feature_columns``.
        feature_columns: Feature columns of the output table (a subset of the
            input's when the preprocessor filters columns).
        spec: The preprocessor's specification (provenance fingerprint).

    Returns:
        A new table sharing id/outcome columns with ``table``.

    Raises:
        HABITAPIError: If ``values`` is not row-aligned with ``table``.
    """
    if len(values) != len(table.frame):
        raise HABITAPIError(
            f"table_preprocessor.{spec.name} produced {len(values)} rows for a "
            f"{len(table.frame)}-row table; transformations must be row-preserving."
        )
    non_feature = [
        column for column in table.frame.columns
        if column not in table.feature_columns
    ]
    frame = table.frame[non_feature].copy()
    for column in feature_columns:
        frame[column] = values[column].to_numpy()
    if table.provenance is not None:
        provenance = table.provenance.derive(
            produced_by=f"table_preprocessor.{spec.name}",
            spec_fingerprint=spec.fingerprint(),
        )
    else:
        provenance = Provenance(
            produced_by=f"table_preprocessor.{spec.name}",
            spec_fingerprint=spec.fingerprint(),
            inputs=(),
            software=software_fingerprint(),
        )
    return FeatureTable(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=feature_columns,
        outcome=table.outcome,
        provenance=provenance,
    )
