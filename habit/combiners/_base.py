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
"""Shared machinery for the built-in combiners.

Every combiner answers the same question -- "merge the column blocks my
sibling nodes produced" -- and therefore shares the same preconditions: at
least one block, a shared row count across siblings, and no duplicate output
columns (a duplicated name would make the merged block ambiguous to every
downstream consumer, from cohort preprocessing to model assignment). Keeping
that contract in one place is what lets new combiners stay one-screen
implementations.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from habit.exceptions import HABITAPIError

__all__ = [
    "check_blocks",
    "block_sources",
    "concat_blocks",
]


def check_blocks(blocks: Sequence[pd.DataFrame], *, owner: str) -> None:
    """
    Require at least one block and a shared row count across siblings.

    Args:
        blocks: Child blocks in child order.
        owner: Combiner name used in error messages.

    Raises:
        HABITAPIError: If no block was given or row counts differ.
    """
    if not blocks:
        raise HABITAPIError(
            f"{owner}: a combiner requires at least one child block; "
            "the feature tree gave it none."
        )
    rows = len(blocks[0])
    for index, block in enumerate(blocks[1:], start=1):
        if len(block) != rows:
            raise HABITAPIError(
                f"{owner}: child block {index} has {len(block)} rows but "
                f"child block 0 has {rows}. Siblings of a feature tree must "
                "describe the same units; a row mismatch means one child "
                "silently worked on a different population."
            )


def block_sources(
    blocks: Sequence[pd.DataFrame],
    context: Optional[Mapping[str, Any]],
    *,
    owner: str,
) -> Tuple[str, ...]:
    """
    Resolve the source label of each child block, in child order.

    The tree wrapper normally supplies the labels through
    ``context["sources"]`` (a leaf's ``as_`` alias when set, else its
    ``modality``, else the node name). When no context is available -- a
    combiner called directly in a notebook -- a single-column block falls
    back to its column name and anything else to ``block_{index}``.

    Args:
        blocks: Child blocks in child order.
        context: Evaluation context supplied by the tree wrapper, or None.
        owner: Combiner name used in error messages.

    Returns:
        One source label per child block.

    Raises:
        HABITAPIError: If a supplied ``sources`` list does not match the
            block count.
    """
    sources = (context or {}).get("sources")
    if sources is None:
        resolved = []
        for index, block in enumerate(blocks):
            if block.shape[1] == 1:
                resolved.append(str(block.columns[0]))
            else:
                resolved.append(f"block_{index}")
        return tuple(resolved)
    resolved = tuple(str(label) for label in sources)
    if len(resolved) != len(blocks):
        raise HABITAPIError(
            f"{owner}: the tree supplied {len(resolved)} source labels for "
            f"{len(blocks)} child blocks."
        )
    return resolved


def concat_blocks(blocks: Sequence[pd.DataFrame], *, owner: str) -> pd.DataFrame:
    """
    Concatenate child blocks column-wise with a duplicate-column guard.

    Args:
        blocks: Child blocks in child order.
        owner: Combiner name used in error messages.

    Returns:
        The merged block. A single child is returned as a copy, so a
        combiner node with one child behaves as a transparent rename point.

    Raises:
        HABITAPIError: On row mismatch or duplicated output column names.
    """
    check_blocks(blocks, owner=owner)
    if len(blocks) == 1:
        return blocks[0].copy()
    merged = pd.concat([block.reset_index(drop=True) for block in blocks], axis=1)
    duplicated = sorted(merged.columns[merged.columns.duplicated()].unique())
    if duplicated:
        raise HABITAPIError(
            f"{owner}: the merged block would contain duplicate columns "
            f"{duplicated}. Give one of the colliding children an ``as_`` "
            "alias so every output column keeps a unique name."
        )
    return merged
