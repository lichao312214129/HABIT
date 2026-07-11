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
