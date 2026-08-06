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
"""Spatial geometry contract shared by every volumetric object of a subject.

This module is part of the L2 contracts layer. It must stay free of any
configuration, YAML, or filesystem concerns so that in-memory pipelines built
by third parties can rely on it without accepting HABIT conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["Geometry"]


@dataclass(frozen=True)
class Geometry:
    """
    Spatial definition shared by every volumetric object of one subject.

    Two volumetric objects may only be combined when their geometries are
    compatible. Making geometry an explicit, comparable value is what lets
    HABIT accept images produced by other tools (nnU-Net, MONAI, 3D Slicer)
    without a directory convention acting as the implicit contract.

    Axis-order convention follows the existing public ``habit.api.image``
    contract: ``shape`` is the NumPy array shape in ``(z, y, x)`` order,
    while ``spacing``, ``origin`` and ``direction`` keep the SimpleITK
    physical-space axis order ``(x, y, z)`` so that round-tripping through
    ``SimpleITK.Image`` never transposes metadata.

    Attributes:
        shape: Voxel grid size as the NumPy array shape ``(z, y, x)``.
        spacing: Physical voxel size in mm, SimpleITK axis order ``(x, y, z)``.
        origin: Physical coordinate of voxel ``(0, 0, 0)``, SimpleITK order.
        direction: Row-major direction cosine matrix, flattened (9 values for
            3D volumes).
        frame_of_reference: Optional identifier tying several series to the
            same physical space, used to detect silently mismatched
            registrations.
    """

    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    origin: Tuple[float, ...]
    direction: Tuple[float, ...]
    frame_of_reference: Optional[str] = None

    def is_compatible_with(
        self,
        other: "Geometry",
        *,
        tolerance: float = 1e-5,
        direction_tolerance: float = 1e-4,
    ) -> bool:
        """
        Report whether two geometries describe the same voxel grid.

        Spacing and origin use ``tolerance`` (default ``1e-5`` absolute).
        Direction cosines use the looser ``direction_tolerance`` (default
        ``1e-4`` absolute): DICOM / ITK round-trips routinely differ by
        ~1e-5 in individual cosine entries without any meaningful grid
        misalignment, while a true axis swap or oblique mismatch remains
        far above ``1e-4``.

        Args:
            other: Geometry to compare against.
            tolerance: Absolute tolerance for spacing and origin.
            direction_tolerance: Absolute tolerance for the flattened
                direction cosine matrix. Kept separate from ``tolerance``
                so spacing/origin stay strict while DICOM noise in
                direction is tolerated.

        Returns:
            ``True`` when the grids coincide within the stated tolerances.
        """
        if not isinstance(other, Geometry):
            return NotImplemented
        if tuple(self.shape) != tuple(other.shape):
            return False
        if self.frame_of_reference and other.frame_of_reference:
            if self.frame_of_reference != other.frame_of_reference:
                return False
        # rtol=0 keeps the documented absolute tolerances honest: DICOM noise
        # often lands on near-zero cosine entries where a non-zero rtol would
        # not enlarge the acceptance window (and would on the ~1.0 diagonals).
        return (
            bool(np.allclose(self.spacing, other.spacing, rtol=0.0, atol=tolerance))
            and bool(np.allclose(self.origin, other.origin, rtol=0.0, atol=tolerance))
            and bool(
                np.allclose(
                    self.direction,
                    other.direction,
                    rtol=0.0,
                    atol=direction_tolerance,
                )
            )
        )

    @classmethod
    def from_array(
        cls,
        shape: Tuple[int, ...],
        *,
        spacing: Optional[Tuple[float, ...]] = None,
        origin: Optional[Tuple[float, ...]] = None,
        direction: Optional[Tuple[float, ...]] = None,
        frame_of_reference: Optional[str] = None,
    ) -> "Geometry":
        """
        Build a geometry from an array shape with identity defaults.

        Args:
            shape: Voxel grid size as the NumPy array shape ``(z, y, x)``.
            spacing: Physical voxel size; defaults to 1 mm isotropic.
            origin: Physical origin; defaults to the zero vector.
            direction: Flattened direction cosine matrix; defaults to identity.
            frame_of_reference: Optional shared-space identifier.

        Returns:
            A geometry describing the grid.
        """
        ndim = len(shape)
        return cls(
            shape=tuple(int(v) for v in shape),
            spacing=spacing or tuple(1.0 for _ in range(ndim)),
            origin=origin or tuple(0.0 for _ in range(ndim)),
            direction=direction
            or tuple(float(v) for v in np.eye(ndim, dtype=float).ravel()),
            frame_of_reference=frame_of_reference,
        )
