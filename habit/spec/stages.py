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
"""Named habitat dataflow stages (re-exports from :mod:`habit.spec.specs`)."""

from __future__ import annotations

from habit.spec.specs import (
    POOL_COMPONENT_NAME,
    ROLE_ASSIGN,
    ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    ROLE_EXTRACT_VOXEL_FEATURES,
    ROLE_FIT,
    ROLE_PARTITION,
    ROLE_POOL,
    ROLE_POSTPROCESS_HABITAT,
    ROLE_POSTPROCESS_SUPERVOXEL,
    ROLE_PREPROCESS,
    ROLE_QUANTIFY,
    Stage,
)

__all__ = [
    "Stage",
    "ROLE_EXTRACT_VOXEL_FEATURES",
    "ROLE_PREPROCESS",
    "ROLE_PARTITION",
    "ROLE_EXTRACT_SUPERVOXEL_FEATURES",
    "ROLE_POOL",
    "ROLE_FIT",
    "ROLE_ASSIGN",
    "ROLE_QUANTIFY",
    "ROLE_POSTPROCESS_SUPERVOXEL",
    "ROLE_POSTPROCESS_HABITAT",
    "POOL_COMPONENT_NAME",
]
