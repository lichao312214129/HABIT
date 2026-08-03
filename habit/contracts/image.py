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
"""Lazy image references and materialised volume contracts.

This module is part of the L2 contracts layer. The lazy/eager split is the
single most important reason HABIT can serve both the notebook user with 30
subjects and the batch user with 3000: operators always receive an
:class:`ImageRef` and decide when to materialise it, so small cohorts can stay
fully in memory while large cohorts pass lightweight handles across process
boundaries.

The materialised :class:`ImageVolume` / :class:`MaskVolume` defined here reuse
the existing public classes from ``habit.api.image`` (per the v1.0 architecture
mapping) and add the two members that make them satisfy :class:`ImageRef`
structurally -- a ``geometry`` property and a ``load()`` method returning the
already-resident array. There is therefore ONE family of image types, not a
parallel eager/lazy pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from habit.api.image import ImageVolume as _PublicImageVolume
from habit.api.image import MaskVolume as _PublicMaskVolume
from habit.contracts.geometry import Geometry

__all__ = [
    "ImageRef",
    "ImageVolume",
    "MaskVolume",
    "ArrayImageRef",
]


@runtime_checkable
class ImageRef(Protocol):
    """
    Lazy handle to volumetric data.

    Operators always receive an ``ImageRef`` and decide when to materialise
    it, which means:

    - small cohorts can stay fully in memory and compose freely;
    - large cohorts pass lightweight handles across process boundaries;
    - third parties can back a subject with PACS, zarr, a torch tensor, or an
      in-memory array by implementing this protocol alone.

    ``ImageVolume`` / ``MaskVolume`` below are the already-materialised
    counterparts and satisfy this protocol structurally (``load()`` returning
    their own array).
    """

    @property
    def geometry(self) -> Geometry:
        """Return grid definition without materialising voxel data."""

    def load(self) -> np.ndarray:
        """Materialise and return the voxel array."""


class ImageVolume(_PublicImageVolume):
    """
    Materialised intensity volume bound to a geometry.

    Subclasses the stable public :class:`habit.api.image.ImageVolume` so any
    value produced by existing HABIT code can flow into the v1.0 contracts
    unchanged, and adds the :class:`ImageRef` surface (``geometry`` /
    ``load()``).

    Attributes:
        data: Voxel intensities, NumPy axis order ``(z, y, x)``.
        spacing: Physical voxel size, SimpleITK axis order ``(x, y, z)``.
        origin: Physical origin, SimpleITK axis order.
        direction: Flattened direction cosine matrix.
        modality: Modality or sequence label, e.g. ``"T1"``, ``"delay2"``.
    """

    @property
    def geometry(self) -> Geometry:
        """Return the spatial definition of this volume without copying data."""
        return Geometry(
            shape=tuple(int(v) for v in self.data.shape),
            spacing=tuple(self.spacing),
            origin=tuple(self.origin),
            direction=tuple(self.direction),
        )

    def load(self) -> np.ndarray:
        """Return the already-resident voxel array (ImageRef conformance)."""
        return self.data


class MaskVolume(_PublicMaskVolume):
    """
    Materialised label volume bound to a geometry.

    Attributes:
        data: Integer labels; ``0`` denotes background.
        spacing: Physical voxel size, SimpleITK axis order.
        origin: Physical origin, SimpleITK axis order.
        direction: Flattened direction cosine matrix.
        labels: Non-background label values present in the mask.
        label_names: Optional human-readable names per label value.
    """

    @property
    def geometry(self) -> Geometry:
        """Return the spatial definition of this mask without copying data."""
        return Geometry(
            shape=tuple(int(v) for v in self.data.shape),
            spacing=tuple(self.spacing),
            origin=tuple(self.origin),
            direction=tuple(self.direction),
        )

    def load(self) -> np.ndarray:
        """Return the already-resident label array (ImageRef conformance)."""
        return self.data

    @property
    def roi_name(self) -> Optional[str]:
        """Return the ROI name for this mask, mapped from ``modality``.

        The public base stores the region label in the shared ``modality``
        field; the contracts layer exposes it under the domain term used by
        the ``Subject.masks`` mapping.
        """
        return self.modality


@dataclass(frozen=True)
class ArrayImageRef:
    """
    In-memory :class:`ImageRef` backed by a NumPy array.

    This is the reference implementation for custom lazy references: it holds
    the array plus its geometry and materialises trivially. The ``geometry``
    field satisfies the :class:`ImageRef` property structurally, so third
    parties can implement the same two-member surface over PACS, zarr, or
    torch tensors.

    Attributes:
        array: Voxel values, NumPy axis order ``(z, y, x)``.
        geometry: Spatial definition of ``array``.
    """

    array: np.ndarray
    geometry: Geometry

    def load(self) -> np.ndarray:
        """Return the held array."""
        return self.array

    def load_volume(self, *, modality: Optional[str] = None) -> ImageVolume:
        """
        Materialise directly as an :class:`ImageVolume`.

        Args:
            modality: Optional modality label attached to the volume.

        Returns:
            The materialised volume bound to ``geometry``.
        """
        return ImageVolume(
            data=self.array,
            spacing=self.geometry.spacing,
            origin=self.geometry.origin,
            direction=self.geometry.direction,
            modality=modality,
        )
