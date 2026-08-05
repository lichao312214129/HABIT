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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from habit.adapters.image_refs import FileImageRef
from habit.exceptions import DataFormatError
from habit.contracts.subject import Cohort, Subject

__all__ = ["DirectoryDataSource"]


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
                images[modality] = FileImageRef(
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
            masks[self.roi] = FileImageRef(mask_path, is_mask=True, role_name=self.roi)
            subjects.append(
                Subject(
                    subject_id=subject_dir.name,
                    images=images,
                    masks=masks,
                )
            )
        if not subjects:
            raise DataFormatError(self._incomplete_cohort_message())

        return Cohort(subjects, name=self.name, metadata=self.metadata)

    def _discovered_modalities(self) -> Tuple[Set[str], List[str]]:
        """
        Scan the images tree for modality folders that contain a file.

        Returns:
            Tuple of (modalities with at least one file somewhere, subject ids scanned).
        """
        found: Set[str] = set()
        scanned: List[str] = []
        for subject_dir in self._subject_dirs(self.images_folder):
            scanned.append(subject_dir.name)
            for entry in subject_dir.iterdir():
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                if _first_file_in(entry) is not None:
                    found.add(entry.name)
        return found, scanned

    def _incomplete_cohort_message(self) -> str:
        """
        Build an actionable error when no subject satisfies the spec.

        Returns:
            Multi-line message describing where HABIT looked and what it found.
        """
        found_modalities, scanned_subjects = self._discovered_modalities()
        requested = set(self.modalities)
        missing_from_tree = sorted(requested - found_modalities)
        images_root = self.root / self.images_folder
        masks_root = self.root / self.masks_folder

        lines = [
            (
                f"No complete subjects found under {self.root} for modalities "
                f"{list(self.modalities)} and roi {self.roi!r}."
            ),
            "",
            f"Looked under: {images_root}/<subject>/<modality>/",
            f"             {masks_root}/<subject>/{self.roi}/",
        ]
        if scanned_subjects:
            lines.append(
                "Subjects scanned: " + ", ".join(sorted(scanned_subjects))
            )
        else:
            lines.append(f"No subject folders found under {images_root}.")
        if found_modalities:
            lines.append(
                "Modalities present in the data tree: "
                + ", ".join(sorted(found_modalities))
            )
        if missing_from_tree:
            lines.append(
                "Modalities configured but not found in the data tree: "
                + ", ".join(missing_from_tree)
            )
        lines.append(
            "Each subject must provide every configured modality and an ROI "
            f"mask at {masks_root}/<subject>/{self.roi}/."
        )
        return "\n".join(lines)
