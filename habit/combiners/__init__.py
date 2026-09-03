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
"""Built-in feature block combiners and their registry.

Combiners are the internal nodes of a feature composition tree: they merge
the column blocks their sibling nodes produced and never touch images,
subjects, or the filesystem. The same combiner implementations serve every
granularity (voxel, supervoxel, habitat), because the tree wrappers hand
them plain ``DataFrame`` blocks with positionally aligned rows.
"""

from __future__ import annotations

from habit.combiners._base import (
    block_sources,
    check_blocks,
    concat_blocks,
)
from habit.combiners.arithmetic import (
    DifferenceCombiner,
    RatioCombiner,
)
from habit.combiners.concat import ConcatCombiner
from habit.combiners.expression import ExpressionCombiner
from habit.combiners.kinetic import KineticCombiner
from habit.combiners.registry import CombinerRegistry
from habit.combiners.weighted import (
    AverageCombiner,
    WeightedConcatCombiner,
)

from habit._protocols import Combiner

__all__ = [
    "Combiner",
    "block_sources",
    "check_blocks",
    "concat_blocks",
    "AverageCombiner",
    "ConcatCombiner",
    "DifferenceCombiner",
    "ExpressionCombiner",
    "KineticCombiner",
    "RatioCombiner",
    "WeightedConcatCombiner",
    "CombinerRegistry",
]
