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
"""Shared eye-check helper for habitat example scripts.

Opens the first subject's anatomy + habitat labels in napari so users can
confirm the run succeeded visually. Set ``HABIT_NO_VIEW=1`` to skip (CI /
docs smoke). For fuller 3D review, also load the volumes in ITK-SNAP /
3D Slicer / a SimpleITK-based viewer.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

__all__ = ["eye_check_habitats", "eye_check_study"]


def eye_check_habitats(
    subject: Any,
    habitat_map: Any,
    *,
    modality: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Open napari with one anatomy volume and one habitat label map.

    Args:
        subject: A :class:`~habit.contracts.Subject` (or compatible) with images.
        habitat_map: Object with ``label_array`` (and optional ``subject_id``).
        modality: Image key to display; ``None`` uses the first image.
        title: Optional viewer window title.
    """
    flag = os.environ.get("HABIT_NO_VIEW", "").strip().lower()
    if flag in {"1", "true", "yes", "y"}:
        print("HABIT_NO_VIEW set: skipped napari eye-check "
              "(load image + *_habitats in ITK-SNAP / 3D Slicer / SimpleITK for 3D).")
        return

    from habit.viz import view_habitat_napari

    images = subject.images
    if modality is None:
        modality = next(iter(images))
    volume = subject.image(modality)
    sid = getattr(habitat_map, "subject_id", None) or getattr(subject, "subject_id", "?")
    window_title = title or f"{sid} habitats on {modality}"
    print(f"Eye-check: opening napari for {window_title} "
          "(close the window to continue; HABIT_NO_VIEW=1 to skip).")
    print("Tip: for 3D review, also load the image + habitat map in "
          "ITK-SNAP / 3D Slicer / a SimpleITK-based viewer.")
    # Pass volume objects, not direction=volume.direction (image header can
    # flip coronal/sagittal vs the ROI / HabitatMap).
    view_habitat_napari(volume, habitat_map, title=window_title)


def eye_check_study(
    cohort: Sequence[Any],
    result: Any,
    *,
    modality: Optional[str] = None,
    map_index: int = 0,
) -> None:
    """
    Eye-check the first (or chosen) habitat map from a StudyResult-like object.

    Args:
        cohort: Cohort used for the run (for anatomy).
        result: Object with ``habitat_maps`` sequence.
        modality: Image key; ``None`` uses the first image on that subject.
        map_index: Which habitat map to show.
    """
    maps: Union[Sequence[Any], Any] = result.habitat_maps
    if not maps:
        print("No habitat maps on result; skipped eye-check.")
        return
    habitat_map = maps[map_index]
    subject_id = getattr(habitat_map, "subject_id", None)
    subject = None
    if subject_id is not None:
        for item in cohort:
            if getattr(item, "subject_id", None) == subject_id:
                subject = item
                break
    if subject is None:
        subject = cohort[0]
    eye_check_habitats(subject, habitat_map, modality=modality)
