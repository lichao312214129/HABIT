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
"""Stable image and mask data contracts.

This namespace is a public alias for the image contracts implemented by
``habit.api.image``.  Import from here when integrating HABIT into a
third-party imaging pipeline.
"""

from habit.api.image import (
    GeometryPolicy,
    GeometryReport,
    ImageInput,
    ImageMaskPair,
    ImageVolume,
    MaskInput,
    MaskVolume,
    align_image_mask,
    read_image,
    read_mask,
    validate_geometry,
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
