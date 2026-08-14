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
"""Tests for habitat label matching / remapping kernels."""

from __future__ import annotations

import numpy as np
import pytest

from habit.kernels.habitat_label_match import (
    align_label_array,
    habitat_intensity_centroids,
    match_labels_by_centroid,
    match_labels_by_overlap,
    remap_label_array,
)


def _two_block_labels() -> np.ndarray:
    """2x2x2 habitat 1 then 2x2x2 habitat 2 on a 4x4x4 grid."""
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels[2:4, 0:2, 0:2] = 2
    return labels


def _swap_ids(labels: np.ndarray) -> np.ndarray:
    """Permute 1 <-> 2 without changing spatial support."""
    swapped = np.asarray(labels).copy()
    swapped[labels == 1] = 9
    swapped[labels == 2] = 1
    swapped[swapped == 9] = 2
    return swapped


def test_overlap_recovers_swapped_ids() -> None:
    """Hungarian overlap maps a pure id permutation back to the reference."""
    reference = _two_block_labels()
    mapping = match_labels_by_overlap(reference, _swap_ids(reference))
    assert mapping == {1: 2, 2: 1}


def test_centroid_recovers_swapped_ids() -> None:
    """Distinct intensity centroids recover a swapped labelling."""
    reference = _two_block_labels()
    image = np.zeros((4, 4, 4), dtype=np.float64)
    image[0:2, 0:2, 0:2] = 1.0
    image[2:4, 0:2, 0:2] = 10.0
    aligned = align_label_array(
        reference, _swap_ids(reference), image=image, method="centroid"
    )
    assert np.array_equal(aligned, reference)


def test_explicit_centroids_swap() -> None:
    """Cluster-centre matrices (row i = habitat i+1) drive the assignment."""
    mapping = match_labels_by_centroid(
        np.array([1, 2]),
        np.array([[0.0], [10.0]]),
        np.array([1, 2]),
        np.array([[10.0], [0.0]]),
    )
    assert mapping == {1: 2, 2: 1}


def test_remap_swap_is_collision_safe() -> None:
    """A 1<->2 swap must not leave both ids as 2."""
    labels = _two_block_labels()
    remapped = remap_label_array(labels, {1: 2, 2: 1})
    assert np.array_equal(remapped, _swap_ids(labels))


def test_align_identity_when_already_matched() -> None:
    """Already-aligned maps stay unchanged under both matchers."""
    reference = _two_block_labels()
    image = np.zeros((4, 4, 4), dtype=np.float64)
    image[reference == 1] = 1.0
    image[reference == 2] = 10.0
    by_centroid = align_label_array(reference, reference, image=image)
    by_overlap = align_label_array(reference, reference, method="overlap")
    assert np.array_equal(by_centroid, reference)
    assert np.array_equal(by_overlap, reference)


def test_intensity_centroids_are_means() -> None:
    """Per-habitat centroid is the mean intensity of that habitat."""
    labels = _two_block_labels()
    image = np.zeros((4, 4, 4), dtype=np.float64)
    image[labels == 1] = 2.0
    image[labels == 2] = 8.0
    ids, centroids = habitat_intensity_centroids(image, labels)
    assert np.array_equal(ids, np.array([1, 2]))
    assert centroids[0, 0] == pytest.approx(2.0)
    assert centroids[1, 0] == pytest.approx(8.0)


def test_unknown_method_raises() -> None:
    """Only centroid and overlap are accepted."""
    labels = _two_block_labels()
    with pytest.raises(ValueError, match="method"):
        align_label_array(labels, labels, method="hungarian")
