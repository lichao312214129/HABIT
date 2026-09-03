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
"""File-backed lazy image references (L1).

Shared building block for filesystem ``DataSource`` implementations: a
reference that carries only a path plus lazily-read header metadata, so a
cohort of thousands of subjects can cross a process boundary without
carrying a single voxel. ``DirectoryDataSource`` (HABIT layout) builds on it,
and third-party sources may subclass it (e.g. to binarise multi-label files
at load time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.image import ImageVolume, MaskVolume

__all__ = ["FileImageRef"]


def _require_simpleitk() -> Any:
    """Import SimpleITK lazily so the adapter layer stays light to import."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency present in CI
        raise HABITAPIError(
            "SimpleITK is required to read image files from disk."
        ) from exc
    return sitk


class FileImageRef:
    """
    Lazy :class:`~habit.contracts.image.ImageRef` backed by one image file.

    Holds only the path plus lazily-read header metadata, so a cohort of
    thousands of subjects can cross a process boundary without carrying a
    single voxel. Header fields are cached after the first access; pixel
    data is only read by :meth:`load` / :meth:`load_volume`.

    Args:
        path: Image file readable by SimpleITK.
        is_mask: Whether the file holds a label mask (selects
            :class:`MaskVolume` materialisation and nearest-neighbour
            semantics downstream).
        role_name: Modality or ROI name attached to materialised volumes.
    """

    def __init__(self, path: Union[str, Path], *, is_mask: bool, role_name: str) -> None:
        self.path = Path(path)
        self.is_mask = is_mask
        self.role_name = role_name
        self._geometry: Optional[Geometry] = None

    @property
    def geometry(self) -> Geometry:
        """Return the grid definition, reading only the file header."""
        if self._geometry is None:
            sitk = _require_simpleitk()
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(self.path))
            reader.ReadImageInformation()
            size_xyz = tuple(int(v) for v in reader.GetSize())
            # SimpleITK reports size in (x, y, z); NumPy arrays are (z, y, x).
            shape = tuple(reversed(size_xyz))
            self._geometry = Geometry(
                shape=shape,
                spacing=tuple(float(v) for v in reader.GetSpacing()),
                origin=tuple(float(v) for v in reader.GetOrigin()),
                direction=tuple(float(v) for v in reader.GetDirection()),
            )
        return self._geometry

    def load(self) -> np.ndarray:
        """Materialise and return the voxel array."""
        sitk = _require_simpleitk()
        # ``_require_simpleitk`` returns ``Any`` (lazy optional import), but
        # ``GetArrayFromImage`` is guaranteed to produce a NumPy array.
        return cast(np.ndarray, sitk.GetArrayFromImage(sitk.ReadImage(str(self.path))))

    def load_volume(self) -> Union[ImageVolume, MaskVolume]:
        """
        Materialise with full physical metadata in one read.

        Returns:
            An :class:`ImageVolume`, or a :class:`MaskVolume` when the
            reference was created for a mask file.
        """
        sitk = _require_simpleitk()
        image = sitk.ReadImage(str(self.path))
        array = sitk.GetArrayFromImage(image)
        geometry = self.geometry
        if self.is_mask:
            return MaskVolume(
                data=array,
                spacing=geometry.spacing,
                origin=geometry.origin,
                direction=geometry.direction,
                modality=self.role_name,
            )
        return ImageVolume(
            data=array,
            spacing=geometry.spacing,
            origin=geometry.origin,
            direction=geometry.direction,
            modality=self.role_name,
        )
