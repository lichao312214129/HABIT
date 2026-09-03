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
"""Concat combiner: the plain column-wise merge of sibling blocks."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from habit.combiners._base import concat_blocks
from habit.combiners.registry import CombinerRegistry
from habit.spec.specs import Spec

__all__ = ["ConcatCombiner"]


@CombinerRegistry.register("concat")
class ConcatCombiner:
    """
    Merge sibling blocks by placing their columns side by side.

    This is the workhorse of multi-modality composition:
    ``concat(raw("T1"), raw("T2"))`` yields the two-modality voxel field,
    and ``concat(mean("T1"), std("T1"), mean("T2"))`` the mixed supervoxel
    description. Column names pass through unchanged, so the children own
    the naming (single-column leaves name their column after the source
    label; multi-column leaves suffix it).
    """

    def __init__(self) -> None:
        pass

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(name="concat", params={})

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Concatenate the child blocks column-wise.

        Args:
            blocks: Child blocks in child order.
            context: Unused by this combiner.

        Returns:
            The merged block with all child columns in child order.
        """
        return concat_blocks(blocks, owner="concat")
