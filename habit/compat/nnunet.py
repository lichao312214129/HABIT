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
"""nnU-Net dataset interop: read ``imagesTr`` / ``labelsTr`` + ``dataset.json``.

Segmentation is usually produced by a dedicated tool, so requiring users to
re-arrange those outputs into HABIT's own folder layout is friction with no
scientific purpose. :class:`NnUNetDataSource` reads an nnU-Net raw dataset
directly and turns it into a :class:`~habit.contracts.subject.Cohort`::

    from habit.compat.nnunet import NnUNetDataSource

    cohort = NnUNetDataSource("Dataset001_Tumor", roi_label=1).load()

Expected layout (nnU-Net v2; the v1 ``"modality"`` key is also honoured)::

    Dataset001_Tumor/
      dataset.json      # channel_names, labels, training entries
      imagesTr/         # <case>_0000.nii.gz (channel 0), <case>_0001.nii.gz, ...
      labelsTr/         # <case>.nii.gz (multi-label segmentation)

The multi-label files are binarised AT LOAD TIME against ``roi_label`` -- the
HABIT mask contract is a binary ROI, so ``roi_label=1`` yields a mask that is
1 exactly where the nnU-Net label equals 1. Everything stays lazy: only file
paths cross process boundaries until a voxel is actually needed.

No part of the ``nnunet`` package is required; this module only follows the
dataset convention.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from habit.adapters.image_refs import FileImageRef
from habit.exceptions import DataFormatError
from habit.contracts.image import MaskVolume
from habit.contracts.subject import Cohort, Subject

__all__ = ["NnUNetDataSource"]

#: Suffix nnU-Net uses for channel files: ``<case>_0000.nii.gz`` etc.
_CHANNEL_SUFFIX = re.compile(r"_(\d{4})$")

#: Compound and simple image extensions understood when stripping file names.
_KNOWN_EXTENSIONS = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd",
    ".nhdr",
    ".png",
    ".tif",
    ".tiff",
)


def _strip_extension(filename: str) -> str:
    """Remove a known (possibly compound) image extension from a file name."""
    for extension in _KNOWN_EXTENSIONS:
        if filename.lower().endswith(extension):
            return filename[: -len(extension)]
    return Path(filename).stem


def _case_id_from_image_name(filename: str) -> Optional[str]:
    """Derive the case id from a channel file name, or ``None`` if not one."""
    stem = _strip_extension(filename)
    match = _CHANNEL_SUFFIX.search(stem)
    if match is None:
        return None
    return stem[: match.start()]


def _channel_index_from_name(filename: str) -> Optional[int]:
    """Extract the 4-digit channel index from a channel file name."""
    stem = _strip_extension(filename)
    match = _CHANNEL_SUFFIX.search(stem)
    return int(match.group(1)) if match is not None else None


def _read_dataset_json(path: Path) -> Optional[Mapping[str, Any]]:
    """Read ``dataset.json`` when present; ``None`` when absent."""
    dataset_json = path / "dataset.json"
    if not dataset_json.is_file():
        return None
    try:
        payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"Invalid dataset.json at {dataset_json}: {exc}.") from exc
    if not isinstance(payload, Mapping):
        raise DataFormatError(f"dataset.json at {dataset_json} is not a JSON object.")
    return payload


def _channel_names(payload: Optional[Mapping[str, Any]]) -> Dict[int, str]:
    """
    Extract channel index -> modality name from a dataset.json payload.

    nnU-Net v2 writes ``channel_names``; v1 wrote ``modality`` with either
    plain strings or ``{"name": ...}`` mappings per channel. Both are
    honoured so older datasets keep working.
    """
    if payload is None:
        return {}
    raw = payload.get("channel_names", payload.get("modality", {}))
    names: Dict[int, str] = {}
    if isinstance(raw, Mapping):
        for index, value in raw.items():
            try:
                channel = int(index)
            except (TypeError, ValueError):
                continue
            if isinstance(value, Mapping):
                value = value.get("name")
            if value is not None:
                names[channel] = str(value)
    return names


def _labels(payload: Optional[Mapping[str, Any]]) -> Dict[str, Union[int, Tuple[int, ...]]]:
    """
    Extract label name -> value(s) from a dataset.json payload.

    Values are integers, or tuples of integers for nnU-Net's overlapping
    labels (e.g. ``"tumor": [1, 2]``); both binarise correctly via ``isin``.
    """
    if payload is None:
        return {}
    raw = payload.get("labels", {})
    labels: Dict[str, Union[int, Tuple[int, ...]]] = {}
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if isinstance(value, (list, tuple)):
                labels[str(name)] = tuple(int(v) for v in value)
            else:
                try:
                    labels[str(name)] = int(value)
                except (TypeError, ValueError):
                    continue
    return labels


class _BinarizedLabelRef(FileImageRef):
    """
    File reference that binarises a multi-label nnU-Net label file at load.

    The HABIT mask contract is a binary ROI while nnU-Net labels are
    multi-label, so the conversion happens here, at the boundary, and stays
    lazy like every other file reference.

    Args:
        path: Label file readable by SimpleITK.
        label_values: Label values that become foreground (``1``).
        role_name: ROI name attached to the materialised mask.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        label_values: Sequence[int],
        role_name: str,
    ) -> None:
        super().__init__(path, is_mask=True, role_name=role_name)
        self.label_values: Tuple[int, ...] = tuple(int(v) for v in label_values)

    def load(self) -> np.ndarray:
        """Materialise the binarised mask array (1 = selected label(s))."""
        labels = super().load()
        return np.isin(labels, list(self.label_values)).astype(np.int32)

    def load_volume(self) -> MaskVolume:
        """Materialise the binarised mask with full physical metadata."""
        geometry = self.geometry
        return MaskVolume(
            data=self.load(),
            spacing=geometry.spacing,
            origin=geometry.origin,
            direction=geometry.direction,
            modality=self.role_name,
        )


class NnUNetDataSource:
    """
    Build a :class:`Cohort` from an nnU-Net raw dataset.

    Args:
        root: Dataset directory (e.g. ``Dataset001_Tumor``) holding
            ``imagesTr`` / ``labelsTr`` and optionally ``dataset.json``.
        roi_label: Foreground definition -- an integer label value, a label
            NAME resolved through ``dataset.json``'s ``labels`` mapping, or
            a sequence of integer values (union). Binarisation happens at
            load time.
        roi_name: Mask key inside each subject. Defaults to the label name
            when ``roi_label`` resolves to one, else ``"roi"``.
        name: Cohort name used in reports. Defaults to the dataset
            directory name.
        metadata: Optional cohort-level attributes (centre, scanner, study).

    Raises:
        DataFormatError: If the convention folders are missing or a named
            ``roi_label`` is unknown to ``dataset.json``.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        roi_label: Union[int, str, Sequence[int]] = 1,
        roi_name: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.roi_label = roi_label
        self.roi_name = roi_name
        self.name = name
        self.metadata = dict(metadata or {})

    # ------------------------------------------------------------------
    # dataset.json interpretation
    # ------------------------------------------------------------------

    def _resolve_roi(
        self, labels: Mapping[str, Union[int, Tuple[int, ...]]]
    ) -> Tuple[Tuple[int, ...], str]:
        """
        Resolve ``roi_label`` into foreground values plus an ROI name.

        Args:
            labels: Label name -> value(s) from dataset.json (possibly empty).

        Returns:
            ``(label_values, roi_name)``.

        Raises:
            DataFormatError: If a named label is unknown to dataset.json.
        """
        roi_label = self.roi_label
        if isinstance(roi_label, str):
            if roi_label not in labels:
                raise DataFormatError(
                    f"ROI label {roi_label!r} is not declared in dataset.json; "
                    f"available labels: {sorted(labels)}."
                )
            value = labels[roi_label]
            values = (value,) if isinstance(value, int) else tuple(value)
            name = self.roi_name or roi_label
            return tuple(int(v) for v in values), name
        if isinstance(roi_label, (list, tuple)):
            values = tuple(int(v) for v in roi_label)
            default_name = "_".join(str(v) for v in values)
            return values, self.roi_name or f"roi_{default_name}"
        values = (int(roi_label),)
        if self.roi_name is not None:
            return values, self.roi_name
        for label_name, value in labels.items():
            if label_name.lower() == "background":
                continue
            if value == values[0] or (
                isinstance(value, tuple) and values[0] in value
            ):
                return values, label_name
        return values, f"roi_{values[0]}"

    def _training_cases(
        self, payload: Optional[Mapping[str, Any]]
    ) -> Optional[List[str]]:
        """List case ids from dataset.json's ``training`` entries, when present."""
        if payload is None:
            return None
        training = payload.get("training")
        if not isinstance(training, list) or not training:
            return None
        cases: List[str] = []
        for entry in training:
            if not isinstance(entry, Mapping):
                continue
            label_path = entry.get("label")
            image_path = entry.get("image")
            if isinstance(label_path, str) and label_path:
                cases.append(_strip_extension(Path(label_path).name))
            elif isinstance(image_path, str) and image_path:
                case = _case_id_from_image_name(Path(image_path).name)
                if case is not None:
                    cases.append(case)
        return sorted(set(cases)) if cases else None

    def _scan_cases(self, images_tr: Path) -> List[str]:
        """Discover case ids by scanning ``imagesTr`` for channel files."""
        cases = {
            case
            for entry in images_tr.iterdir()
            if entry.is_file()
            and not entry.name.startswith(".")
            and (case := _case_id_from_image_name(entry.name)) is not None
        }
        return sorted(cases)

    def _channel_files(self, images_tr: Path, case: str) -> Dict[int, Path]:
        """Map channel index -> image file for one case."""
        channels: Dict[int, Path] = {}
        for entry in sorted(images_tr.iterdir()):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            if _case_id_from_image_name(entry.name) != case:
                continue
            index = _channel_index_from_name(entry.name)
            if index is not None:
                channels[index] = entry
        return channels

    def _label_file(self, labels_tr: Path, case: str) -> Optional[Path]:
        """Locate the multi-label file for one case."""
        candidates = sorted(
            entry
            for entry in labels_tr.iterdir()
            if entry.is_file()
            and not entry.name.startswith(".")
            and _strip_extension(entry.name) == case
        )
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # DataSource API
    # ------------------------------------------------------------------

    def load(self) -> Cohort:
        """
        Build the cohort described by this source.

        Returns:
            A cohort in sorted case-id order whose images are lazy channel
            references and whose single mask binarises ``roi_label`` at load
            time. Cases without a label file are skipped with a warning,
            mirroring ``DirectoryDataSource``.

        Raises:
            DataFormatError: If ``imagesTr`` / ``labelsTr`` are missing or no
                complete cases are found.
        """
        images_tr = self.root / "imagesTr"
        labels_tr = self.root / "labelsTr"
        for folder in (images_tr, labels_tr):
            if not folder.is_dir():
                raise DataFormatError(
                    f"nnU-Net folder not found: {folder}. Expected layout "
                    "<dataset>/imagesTr|labelsTr + dataset.json."
                )
        payload = _read_dataset_json(self.root)
        names = _channel_names(payload)
        labels = _labels(payload)
        label_values, roi_name = self._resolve_roi(labels)
        cases = self._training_cases(payload)
        if cases is None:
            cases = self._scan_cases(images_tr)

        subjects: List[Subject] = []
        for case in cases:
            channel_files = self._channel_files(images_tr, case)
            if not channel_files:
                print(f"Warning: nnU-Net case {case} has no channel files; skipped.")
                continue
            label_file = self._label_file(labels_tr, case)
            if label_file is None:
                print(
                    f"Warning: nnU-Net case {case} misses its label file; skipped."
                )
                continue
            images = {
                names.get(index, f"channel_{index:04d}"): FileImageRef(
                    path, is_mask=False, role_name=names.get(index, f"channel_{index:04d}")
                )
                for index, path in sorted(channel_files.items())
            }
            masks = {
                roi_name: _BinarizedLabelRef(
                    label_file, label_values=label_values, role_name=roi_name
                )
            }
            subjects.append(Subject(subject_id=case, images=images, masks=masks))
        if not subjects:
            raise DataFormatError(
                f"No complete nnU-Net cases found under {self.root} "
                f"(imagesTr + labelsTr)."
            )
        return Cohort(
            subjects,
            name=self.name or self.root.name,
            metadata=self.metadata,
        )
