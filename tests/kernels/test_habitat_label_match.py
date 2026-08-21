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
    match_label_ids,
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


def test_match_label_ids_centroid_uses_mean_intensity() -> None:
    """Mean intensity, not overlap, drives the Hungarian pairing."""
    reference = _two_block_labels()
    moving = _swap_ids(reference)
    image = np.zeros(reference.shape, dtype=np.float64)
    image[reference == 1] = 1.0
    image[reference == 2] = 10.0
    mapping = match_label_ids(
        reference, moving, method="centroid", image=image
    )
    assert mapping == {1: 2, 2: 1}
    aligned = align_label_array(reference, moving, method="centroid", image=image)
    assert np.array_equal(aligned, reference)


def _three_block_labels() -> np.ndarray:
    """Habitats 1 / 2 / 3 as three adjacent 2x2x2 blocks on a 6x4x4 grid."""
    labels = np.zeros((6, 4, 4), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels[2:4, 0:2, 0:2] = 2
    labels[4:6, 0:2, 0:2] = 3
    return labels


def _swap_ids_2_3(labels: np.ndarray) -> np.ndarray:
    """Permute 2 <-> 3 on the moving map only; habitat 1 stays put."""
    swapped = np.asarray(labels).copy()
    swapped[labels == 2] = 9
    swapped[labels == 3] = 2
    swapped[swapped == 9] = 3
    return swapped


def test_overlap_recovers_habitat_2_3_swap_on_moving_only() -> None:
    """A 2<->3 permutation on moving remaps those ids; reference is untouched."""
    reference = _three_block_labels()
    moving = _swap_ids_2_3(reference)
    mapping = match_labels_by_overlap(reference, moving)
    assert mapping == {1: 1, 2: 3, 3: 2}
    aligned = align_label_array(reference, moving, method="overlap")
    assert np.array_equal(aligned, reference)
    assert np.array_equal(reference, _three_block_labels())
    raw_disagree = (moving != reference) & ((moving > 0) | (reference > 0))
    aligned_disagree = (aligned != reference) & ((aligned > 0) | (reference > 0))
    assert int(np.count_nonzero(raw_disagree)) > 0
    assert int(np.count_nonzero(aligned_disagree)) == 0


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


def test_remap_unmatched_does_not_reuse_target_id() -> None:
    """A leftover moving id must not keep a number already used as a target.

    Moving habitats 1 / 2 / 3, mapping 3->1 and 2->2, leftover 1. Keeping
    1 would merge leftover 1 with remapped 3. Leftover 1 must become 3
    (max reserved reference id 2, then +1).
    """
    labels = np.zeros((6, 4, 4), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels[2:4, 0:2, 0:2] = 2
    labels[4:6, 0:2, 0:2] = 3
    remapped = remap_label_array(labels, {3: 1, 2: 2}, reserved_ids=(1, 2))
    assert set(np.unique(remapped[remapped > 0]).tolist()) == {1, 2, 3}
    assert np.all(remapped[labels == 3] == 1)
    assert np.all(remapped[labels == 2] == 2)
    assert np.all(remapped[labels == 1] == 3)
    assert int(np.count_nonzero(remapped == 0)) == int(np.count_nonzero(labels == 0))


def test_align_extra_moving_habitat_gets_fresh_id() -> None:
    """An extra moving cluster is rewritten to max(reference)+1, not kept."""
    reference = _two_block_labels()
    moving = np.zeros((4, 4, 4), dtype=np.int32)
    moving[0:2, 0:2, 0:2] = 1
    moving[2:4, 0:2, 0:2] = 2
    moving[0:2, 2:4, 0:2] = 3
    aligned = align_label_array(reference, moving, method="overlap")
    # Overlap pairs moving 1->1 and 2->2; leftover 3 becomes 3
    # (max(reference ids)=2, then +1). Ids 1 and 2 stay spatially distinct.
    assert np.all(aligned[moving == 1] == 1)
    assert np.all(aligned[moving == 2] == 2)
    assert np.all(aligned[moving == 3] == 3)
    assert set(np.unique(aligned[aligned > 0]).tolist()) == {1, 2, 3}


def test_align_leftover_original_id_does_not_merge() -> None:
    """Leftover moving 1 must not share a color with a habitat remapped to 1."""
    reference = _two_block_labels()
    moving = np.zeros((4, 4, 4), dtype=np.int32)
    # Habitat 1 is far from both reference blocks so it is the leftover;
    # 2 and 3 sit on reference 1 and 2.
    moving[0:2, 2:4, 2:4] = 1
    moving[0:2, 0:2, 0:2] = 2
    moving[2:4, 0:2, 0:2] = 3
    aligned = align_label_array(reference, moving, method="overlap")
    assert np.all(aligned[moving == 2] == 1)
    assert np.all(aligned[moving == 3] == 2)
    leftover_id = int(aligned[moving == 1][0])
    assert leftover_id == 3
    assert leftover_id not in (1, 2)
    assert np.all(aligned[moving == 1] == leftover_id)


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
