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
"""Habitat stability under perturbation: matched Dice scores.

When habitats are computed independently on the original and on a perturbed
image, the cluster labels are not comparable -- cluster 1 of the second fit
may correspond to cluster 3 of the first. The reference implementation
(Prior et al., Radiol Artif Intell 2024;6(2):e230118) matches the clusters
by maximal overlap (Hungarian assignment, their ``munkres`` step) and then
reports the Dice similarity of every matched pair. This module implements
exactly that on :class:`~habit.contracts.habitat.HabitatMap` objects; WHO
produced the maps (per-subject GMM, cohort model, any recipe) is not this
layer's concern.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from habit.contracts.habitat import HabitatMap
from habit.exceptions import HABITAPIError

__all__ = ["habitat_stability"]


def _present_labels(label_array: np.ndarray) -> np.ndarray:
    """Return the sorted non-background labels of a habitat label image."""
    labels = np.unique(label_array)
    return labels[labels != 0]


def habitat_stability(
    reference: HabitatMap,
    perturbed: Sequence[HabitatMap],
) -> pd.DataFrame:
    """
    Score habitat stability between a reference map and perturbed maps.

    Clusters of each perturbed map are matched to the reference clusters by
    maximal voxel overlap (Hungarian assignment), then the Dice similarity
    is computed per matched pair. Reference habitats left unmatched (the
    perturbed map has fewer clusters) score Dice 0 -- a vanished habitat is
    a stability failure, not missing data.

    Args:
        reference: Habitat map of the original subject.
        perturbed: Habitat maps computed independently on perturbed copies,
            each on the same voxel grid as ``reference``.

    Returns:
        Long-format DataFrame with one row per perturbation per reference
        habitat: ``perturbation`` (positional index), ``habitat_id``,
        ``matched_id`` (NA when unmatched), ``dice``, ``n_reference`` and
        ``n_matched`` voxel counts.

    Raises:
        HABITAPIError: If no perturbed map is given or the grids differ.
    """
    if not perturbed:
        raise HABITAPIError("habitat_stability: at least one perturbed map is required.")
    reference_labels = np.asarray(reference.label_array)
    reference_ids = _present_labels(reference_labels)
    records = []
    for index, moved in enumerate(perturbed):
        moved_labels = np.asarray(moved.label_array)
        if moved_labels.shape != reference_labels.shape:
            raise HABITAPIError(
                f"habitat_stability: perturbed map {index} has shape "
                f"{moved_labels.shape}, expected {reference_labels.shape}."
            )
        moved_ids = _present_labels(moved_labels)
        # Overlap matrix: rows = reference habitats, columns = perturbed.
        overlap = np.zeros((reference_ids.size, moved_ids.size), dtype=np.int64)
        for row, habitat_id in enumerate(reference_ids):
            selector = reference_labels == habitat_id
            overlap[row] = [
                int(np.count_nonzero(selector & (moved_labels == moved_id)))
                for moved_id in moved_ids
            ]
        if overlap.size:
            # Hungarian assignment on the negative overlap maximises the
            # total matched overlap (scipy minimises).
            rows, columns = linear_sum_assignment(-overlap)
            matched = dict(zip(rows.tolist(), columns.tolist()))
        else:
            matched = {}
        for row, habitat_id in enumerate(reference_ids):
            n_reference = int(np.count_nonzero(reference_labels == habitat_id))
            if row in matched:
                column = matched[row]
                moved_id = int(moved_ids[column])
                n_moved = int(np.count_nonzero(moved_labels == moved_id))
                intersection = int(overlap[row, column])
                dice = (
                    2.0 * intersection / (n_reference + n_moved)
                    if n_reference + n_moved > 0
                    else 0.0
                )
                records.append(
                    (index, int(habitat_id), moved_id, dice, n_reference, n_moved)
                )
            else:
                records.append(
                    (index, int(habitat_id), None, 0.0, n_reference, 0)
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "perturbation",
            "habitat_id",
            "matched_id",
            "dice",
            "n_reference",
            "n_matched",
        ],
    )
