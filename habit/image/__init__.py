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
"""Convenience re-exports for image volumes and file I/O.

Canonical types live in :mod:`habit.contracts.image`. File I/O and resampling
live in :mod:`habit.adapters.volume_io`. Import this package from notebooks
or CLI helpers that need both in one place; L0-L3 code must import types from
``habit.contracts`` and must not import this package (it pulls adapters).
"""

from habit.adapters.volume_io import (
    ImageInput,
    MaskInput,
    align_image_mask,
    read_image,
    read_mask,
    validate_geometry,
)
from habit.contracts.image import (
    GeometryPolicy,
    GeometryReport,
    ImageMaskPair,
    ImageVolume,
    MaskVolume,
)

__all__ = [
    "GeometryPolicy",
    "GeometryReport",
    "ImageVolume",
    "MaskVolume",
    "ImageMaskPair",
    "ImageInput",
    "MaskInput",
    "read_image",
    "read_mask",
    "validate_geometry",
    "align_image_mask",
]
