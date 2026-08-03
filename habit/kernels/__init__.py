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
"""L0 kernels: pure numerical computation -- no IO, no state, no logging.

Kernels are the independently reviewable mathematical core of HABIT. They
know nothing about subjects, specifications, or the filesystem, and are
usable standalone (e.g. to re-derive a published metric).
"""

from __future__ import annotations

from habit.kernels.habitat_metrics import (
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    msi_features_from_matrix,
    spatial_interaction_matrix,
)

__all__ = [
    "spatial_interaction_matrix",
    "msi_features_from_matrix",
    "habitat_volume_fractions",
    "habitat_region_stats",
    "ith_score",
]
