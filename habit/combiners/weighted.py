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
"""Weighted combiners: scaled concatenation and (weighted) averaging."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from habit.combiners._base import (
    block_sources,
    check_blocks,
    concat_blocks,
)
from habit.combiners.registry import CombinerRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "WeightedConcatCombiner",
    "AverageCombiner",
]


def _resolve_weights(
    weights: Mapping[str, float],
    sources: Sequence[str],
    *,
    owner: str,
) -> np.ndarray:
    """
    Map child-keyed weights onto a per-child weight vector.

    Args:
        weights: Weight per child source label; children without an entry
            keep weight 1.0.
        sources: Source label of each child block, in child order.
        owner: Combiner name used in error messages.

    Returns:
        Array of shape ``(n_children,)`` with the resolved weights.

    Raises:
        HABITAPIError: If a weight key matches no child source label --
            a mistyped key must not silently fall back to 1.0.
    """
    unknown = sorted(set(weights) - set(sources))
    if unknown:
        raise HABITAPIError(
            f"{owner}: weights reference unknown children {unknown}; "
            f"the tree supplied source labels {list(sources)}. Key weights "
            "by each child's ``as_`` alias or modality name."
        )
    return np.array([float(weights.get(source, 1.0)) for source in sources])


@CombinerRegistry.register("weighted_concat")
class WeightedConcatCombiner:
    """
    Concatenate sibling blocks after scaling each by a child-specific weight.

    Modalities with different intensity scales (e.g. CT in Hounsfield units
    next to a normalised MR sequence) distort distance-based clustering:
    the louder modality dominates purely through units. Scaling each child
    block before the merge is the explicit, specifiable answer -- the weight
    is part of the specification and lands in the model fingerprint.

    Args:
        weights: Scale factor per child, keyed by the child's source label
            (its ``as_`` alias when set, else its modality). Children
            without an entry keep weight 1.0.
    """

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.weights: Dict[str, float] = {
            str(key): float(value)
            for key, value in dict(weights or {}).items()
        }

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(name="weighted_concat", params={"weights": dict(self.weights)})

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Scale each child block by its weight and concatenate.

        Args:
            blocks: Child blocks in child order.
            context: Carries the child source labels under ``"sources"``.

        Returns:
            The merged block of weighted child columns.
        """
        check_blocks(blocks, owner="weighted_concat")
        sources = block_sources(blocks, context, owner="weighted_concat")
        factors = _resolve_weights(self.weights, sources, owner="weighted_concat")
        scaled = [
            block.astype(np.float64) * factor
            for block, factor in zip(blocks, factors)
        ]
        return concat_blocks(scaled, owner="weighted_concat")


@CombinerRegistry.register("average")
class AverageCombiner:
    """
    Average sibling blocks element-wise, optionally with child weights.

    The averaging counterpart of :class:`WeightedConcatCombiner`: where
    concatenation keeps every child column, averaging collapses the children
    into a consensus signal -- e.g. the mean of two co-registered repeats of
    the same sequence. All children must have the same number of columns,
    paired positionally.

    Args:
        weights: Weight per child source label. Weights are normalised to
            sum to one; children without an entry keep weight 1.0 (before
            normalisation).
    """

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.weights: Dict[str, float] = {
            str(key): float(value)
            for key, value in dict(weights or {}).items()
        }

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(name="average", params={"weights": dict(self.weights)})

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute the (weighted) column-wise mean across child blocks.

        Args:
            blocks: Child blocks in child order, all with equal column
                counts.
            context: Carries the child source labels under ``"sources"``.

        Returns:
            One block with the averaged columns.

        Raises:
            HABITAPIError: If the children have different column counts.
        """
        check_blocks(blocks, owner="average")
        sources = block_sources(blocks, context, owner="average")
        n_columns = blocks[0].shape[1]
        for index, block in enumerate(blocks[1:], start=1):
            if block.shape[1] != n_columns:
                raise HABITAPIError(
                    f"average: child block {index} has {block.shape[1]} "
                    f"columns but child block 0 has {n_columns}; averaging "
                    "pairs columns positionally and needs equal counts."
                )
        factors = _resolve_weights(self.weights, sources, owner="average")
        total = float(factors.sum())
        if total <= 0:
            raise HABITAPIError(
                f"average: the resolved weights {factors.tolist()} sum to "
                f"{total}; a non-positive total cannot normalise an average."
            )
        factors = factors / total
        stacked = np.stack(
            [block.to_numpy(dtype=np.float64) for block in blocks], axis=0
        )
        values = np.tensordot(factors, stacked, axes=(0, 0))

        names = [str(column) for column in blocks[0].columns]
        identical = all(
            [str(column) for column in block.columns] == names for block in blocks[1:]
        )
        if not identical:
            joined = "-".join(sources)
            if n_columns == 1:
                names = [f"average-{joined}"]
            else:
                names = [f"average_{position}-{joined}" for position in range(n_columns)]
        return pd.DataFrame(values, columns=names)
