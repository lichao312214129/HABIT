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
"""Directory-convention data source (L1).

Migrates the v0.1 directory convention parsed by
``habit.utils.io_utils.get_image_and_mask_paths`` into a first-class
:class:`~habit.contracts.ops.DataSource`. The layout is::

    <root>/images/<subject>/<modality>/<image file>
    <root>/masks/<subject>/<roi>/<mask file>

Subjects are returned in sorted-id order so that cohort order -- part of the
reproducibility contract for population-level clustering -- is deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from habit.api.exceptions import DataFormatError, HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.image import ImageVolume, MaskVolume
from habit.contracts.subject import Cohort, Subject

__all__ = ["DirectoryDataSource"]


def _require_simpleitk() -> Any:
    """Import SimpleITK lazily so the adapter layer stays light to import."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency present in CI
        raise HABITAPIError(
            "SimpleITK is required to read image files from disk."
        ) from exc
    return sitk


class _FileImageRef:
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
        return sitk.GetArrayFromImage(sitk.ReadImage(str(self.path)))

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


def _first_file_in(directory: Path) -> Optional[Path]:
    """Return the first non-hidden file inside a convention subdirectory."""
    if not directory.is_dir():
        return None
    files = sorted(
        entry for entry in directory.iterdir() if entry.is_file() and not entry.name.startswith(".")
    )
    if not files:
        return None
    if len(files) > 1:
        # Matches the v0.1 behaviour: warn and take the first file.
        print(f"Warning: Multiple files in {directory}; using {files[0].name}")
    return files[0]


class DirectoryDataSource:
    """
    Build a :class:`Cohort` from HABIT's conventional directory layout.

    Args:
        root: Directory root holding the images and masks subdirectories.
        modalities: Modality keys to include, in analysis order. A subject
            missing any requested modality is skipped with a warning, mirroring
            the v0.1 scan behaviour.
        roi: Mask key identifying the region of interest.
        images_folder: Name of the images subdirectory under ``root``.
        masks_folder: Name of the masks subdirectory under ``root``.
        name: Human-readable cohort name used in reports.
        metadata: Optional cohort-level attributes (centre, scanner, study).
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        modalities: Sequence[str],
        roi: str,
        images_folder: str = "images",
        masks_folder: str = "masks",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.modalities: Tuple[str, ...] = tuple(modalities)
        self.roi = roi
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.name = name
        self.metadata = dict(metadata or {})

    def _subject_dirs(self, kind_folder: str) -> List[Path]:
        """List subject subdirectories under one convention folder, sorted."""
        base = self.root / kind_folder
        if not base.is_dir():
            raise DataFormatError(
                f"Convention folder not found: {base}. Expected layout "
                "<root>/images|masks/<subject>/<modality>/<file>."
            )
        return sorted(
            (entry for entry in base.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
            key=lambda entry: entry.name,
        )

    def load(self) -> Cohort:
        """
        Build the cohort described by this source.

        Returns:
            A cohort with a defined, reproducible subject order (sorted
            subject ids), whose images and masks are lazy file references.

        Raises:
            DataFormatError: If the convention folders are missing.
        """
        subjects: List[Subject] = []
        for subject_dir in self._subject_dirs(self.images_folder):
            images: Dict[str, Any] = {}
            missing: List[str] = []
            for modality in self.modalities:
                file_path = _first_file_in(subject_dir / modality)
                if file_path is None:
                    missing.append(modality)
                    continue
                images[modality] = _FileImageRef(
                    file_path, is_mask=False, role_name=modality
                )
            if missing:
                print(
                    f"Warning: Subject {subject_dir.name} misses modalities "
                    f"{missing}; skipped."
                )
                continue
            masks: Dict[str, Any] = {}
            mask_path = _first_file_in(
                self.root / self.masks_folder / subject_dir.name / self.roi
            )
            if mask_path is None:
                print(
                    f"Warning: Subject {subject_dir.name} misses ROI "
                    f"{self.roi!r}; skipped."
                )
                continue
            masks[self.roi] = _FileImageRef(mask_path, is_mask=True, role_name=self.roi)
            subjects.append(
                Subject(
                    subject_id=subject_dir.name,
                    images=images,
                    masks=masks,
                )
            )
        if not subjects:
            raise DataFormatError(
                f"No complete subjects found under {self.root} for modalities "
                f"{list(self.modalities)} and roi {self.roi!r}."
            )
        return Cohort(subjects, name=self.name, metadata=self.metadata)
