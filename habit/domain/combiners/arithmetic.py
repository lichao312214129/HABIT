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
"""Pairwise arithmetic combiners: ratio and difference of two blocks."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from habit.domain.combiners._base import block_sources, check_blocks
from habit.domain.combiners.registry import CombinerRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "RatioCombiner",
    "RatioCombinerParams",
    "DifferenceCombiner",
    "DifferenceCombinerParams",
]


def _require_two_children(
    blocks: Sequence[pd.DataFrame], *, owner: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Require exactly two child blocks and return them.

    Args:
        blocks: Child blocks in child order.
        owner: Combiner name used in error messages.

    Returns:
        The ``(first, second)`` block pair.

    Raises:
        HABITAPIError: If the child count differs from two.
    """
    check_blocks(blocks, owner=owner)
    if len(blocks) != 2:
        raise HABITAPIError(
            f"{owner}: requires exactly two child blocks (numerator/first "
            f"and denominator/second); got {len(blocks)}."
        )
    return blocks[0], blocks[1]


def _paired_columns(
    first: pd.DataFrame,
    second: pd.DataFrame,
    sources: Sequence[str],
    prefix: str,
    *,
    owner: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Pair two blocks column-wise and derive output column names.

    A one-column block broadcasts against every column of a wider partner
    (dividing three radiomics columns by one normaliser column); otherwise
    the column counts must match and columns pair positionally.

    Args:
        first: Numerator/minuend block.
        second: Denominator/subtrahend block.
        sources: Source label of each child, in child order.
        prefix: Output name prefix (``ratio`` or ``diff``).
        owner: Combiner name used in error messages.

    Returns:
        ``(left, right, names)`` arrays of shape ``(n_rows, n_pairs)`` plus
        one output name per pair.

    Raises:
        HABITAPIError: If neither broadcast nor positional pairing works.
    """
    left = first.to_numpy(dtype=np.float64)
    right = second.to_numpy(dtype=np.float64)
    first_names = [str(column) for column in first.columns]
    second_names = [str(column) for column in second.columns]

    if left.shape[1] == right.shape[1]:
        names = [
            f"{prefix}-{name_a}-{name_b}"
            for name_a, name_b in zip(first_names, second_names)
        ]
        return left, right, names
    if right.shape[1] == 1:
        names = [f"{prefix}-{name_a}-{sources[1]}" for name_a in first_names]
        return left, np.repeat(right, left.shape[1], axis=1), names
    if left.shape[1] == 1:
        names = [f"{prefix}-{sources[0]}-{name_b}" for name_b in second_names]
        return np.repeat(left, right.shape[1], axis=1), right, names
    raise HABITAPIError(
        f"{owner}: cannot pair {left.shape[1]} columns against "
        f"{right.shape[1]} columns; use equal counts or a one-column block "
        "for broadcasting."
    )


class RatioCombinerParams(BaseModel):
    """Constructor parameters for :class:`RatioCombiner`."""

    model_config = ConfigDict(extra="forbid")
    eps: float = Field(default=1e-8, ge=0.0)


@CombinerRegistry.register("ratio")
class RatioCombiner:
    """
    Element-wise ratio of two sibling blocks: ``first / (second + eps)``.

    Ratios of two modalities (e.g. PET over CT uptake, T1 post-contrast over
    pre-contrast) cancel subject-level intensity scale and are a classic
    habitat feature. The denominator is guarded by ``eps`` so a zero-valued
    voxel yields a large-but-finite ratio instead of an inf that would
    poison every downstream statistic.

    Args:
        eps: Constant added to the denominator before dividing.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(name="ratio", params={"eps": self.eps})

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Divide the first child block by the second.

        Args:
            blocks: Exactly two blocks: numerator first, denominator second.
            context: Carries the child source labels under ``"sources"``.

        Returns:
            One column per paired column, named
            ``ratio-{numerator}-{denominator}``.
        """
        first, second = _require_two_children(blocks, owner="ratio")
        sources = block_sources(blocks, context, owner="ratio")
        left, right, names = _paired_columns(
            first, second, sources, "ratio", owner="ratio"
        )
        values = left / (right + self.eps)
        return pd.DataFrame(values, columns=names)


class DifferenceCombinerParams(BaseModel):
    """Constructor parameters for :class:`DifferenceCombiner`."""

    model_config = ConfigDict(extra="forbid")


@CombinerRegistry.register("difference")
class DifferenceCombiner:
    """
    Element-wise difference of two sibling blocks: ``first - second``.

    Subtraction maps (post-contrast minus pre-contrast enhancement, two
    time points of the same sequence) turn absolute intensities into a
    change signal, which is often the actual biology of interest.
    """

    def __init__(self) -> None:
        pass

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(name="difference", params={})

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Subtract the second child block from the first.

        Args:
            blocks: Exactly two blocks: minuend first, subtrahend second.
            context: Carries the child source labels under ``"sources"``.

        Returns:
            One column per paired column, named ``diff-{first}-{second}``.
        """
        first, second = _require_two_children(blocks, owner="difference")
        sources = block_sources(blocks, context, owner="difference")
        left, right, names = _paired_columns(
            first, second, sources, "diff", owner="difference"
        )
        values = left - right
        return pd.DataFrame(values, columns=names)


CombinerRegistry.register_params_model("ratio", RatioCombinerParams)
CombinerRegistry.register_params_model("difference", DifferenceCombinerParams)
