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
"""Expression combiner: safe arithmetic over sibling block columns.

The evaluation engine (restricted AST, whitelisted functions) is shared
with the ``expression`` voxel extractor in
:mod:`habit.voxel_features.expression`; this combiner only changes
WHERE the names in a formula resolve: to the columns of the merged child
blocks instead of to subject image modalities.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.combiners._base import concat_blocks
from habit.combiners.registry import CombinerRegistry

# The restricted-AST evaluator is deliberately shared with the voxel-level
# expression extractor so both forms enforce the exact same syntax rules;
# importing the private helpers keeps one source of truth for what a
# "safe formula" is.
from habit.voxel_features.expression import (
    _BUILTINS,
    _SafeEvaluator,
    _compile_expression,
    _resolve_feature_table,
)
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["ExpressionCombiner"]


@CombinerRegistry.register("expression")
class ExpressionCombiner:
    """
    Features defined by restricted arithmetic over sibling block columns.

    Each formula may reference the COLUMN NAMES of the merged child blocks
    (e.g. ``T1``, ``wash_in_slope``), the injected constant ``eps``, and the
    whitelisted functions ``abs``, ``sqrt``, ``square``, ``log``, ``log10``,
    ``exp``, ``minimum``, ``maximum``, ``clip``. Power may be written as
    ``**`` or ``^``.

    Example::

        expression(
            raw("T1"), raw("T2"),
            features={"t1_over_t2_sq": "square(T1 / (T2 ** 3 + eps))"},
        )

    Args:
        features: Mapping of feature name to formula. Mutually exclusive
            with ``expressions``.
        expressions: Ordered formulas when names are not provided up front.
        feature_names: Names aligned with ``expressions``; defaults to
            ``expr_0``, ``expr_1``, ...
        eps: Value bound to the name ``eps`` inside every formula.
    """

    def __init__(
        self,
        features: Optional[Mapping[str, str]] = None,
        expressions: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        eps: float = 1e-8,
    ) -> None:
        if features and expressions:
            raise HABITAPIError(
                "expression: provide either 'features' or 'expressions', not both."
            )
        if not features and not expressions:
            raise HABITAPIError(
                "expression: requires 'features' (name->formula) or "
                "'expressions' (ordered formulas)."
            )
        self.features: Optional[Dict[str, str]] = (
            {str(key): str(value) for key, value in features.items()}
            if features is not None
            else None
        )
        self.expressions: Optional[Tuple[str, ...]] = (
            tuple(str(formula) for formula in expressions)
            if expressions is not None
            else None
        )
        self.feature_names: Optional[Tuple[str, ...]] = (
            tuple(str(name) for name in feature_names)
            if feature_names is not None
            else None
        )
        self.eps = float(eps)
        resolved_names, formulas = _resolve_feature_table(
            self.features, self.expressions, self.feature_names
        )
        self._resolved_feature_names: Tuple[str, ...] = resolved_names
        self._formulas: Tuple[str, ...] = formulas
        self._trees: Tuple[ast.Expression, ...] = tuple(
            _compile_expression(formula) for formula in formulas
        )

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="expression",
            params={
                "features": dict(
                    zip(self._resolved_feature_names, self._formulas)
                ),
                "eps": self.eps,
            },
        )

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate every formula on the merged child block.

        Args:
            blocks: Child blocks in child order; their merged column names
                are the identifiers formulas may reference.
            context: Unused by this combiner.

        Returns:
            One column per formula, one row per input row.

        Raises:
            HABITAPIError: If a formula references an unknown name or
                produces a non-broadcastable shape.
        """
        merged = concat_blocks(blocks, owner="expression")
        n_rows = len(merged)
        namespace: Dict[str, Any] = {
            **_BUILTINS,
            "eps": self.eps,
        }
        for column in merged.columns:
            namespace[str(column)] = merged[column].to_numpy(dtype=np.float64)

        columns: List[np.ndarray] = []
        for name, tree in zip(self._resolved_feature_names, self._trees):
            value = _SafeEvaluator(namespace).visit(tree)
            column = np.asarray(value, dtype=np.float64)
            if column.shape != (n_rows,):
                # Broadcast scalars (e.g. literal-only formulas) to every row.
                if column.ndim == 0:
                    column = np.full(n_rows, float(column), dtype=np.float64)
                else:
                    raise HABITAPIError(
                        f"expression: formula for {name!r} produced shape "
                        f"{column.shape}, expected {(n_rows,)}."
                    )
            columns.append(column)

        return pd.DataFrame(
            np.stack(columns, axis=1), columns=list(self._resolved_feature_names)
        )
