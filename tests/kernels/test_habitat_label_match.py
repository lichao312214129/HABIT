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
    adjusted_rand_index,
    align_label_array,
    fit_feature_match_scale,
    habitat_dice_from_mapping,
    habitat_intensity_centroids,
    match_label_ids,
    match_labels_by_centroid,
    match_labels_by_features,
    match_labels_by_overlap,
    overlap_count_table,
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
    """Only centroid, features, and overlap are accepted."""
    labels = _two_block_labels()
    with pytest.raises(ValueError, match="method"):
        align_label_array(labels, labels, method="hungarian")


def test_feature_match_zscore_recovers_energy_shift() -> None:
    """Unscaled habitat means + cohort z-score survive a global Energy shift.

    Patient A: habitat 1 = (8e6, 0.12), habitat 2 = (2e7, 0.03).
    Patient B is the same pair with Energy += 1.5e7 and ids swapped.
    A global Energy offset makes raw Euclidean costs nearly tied
    (1-D translation invariance). After column z-score, Coarseness
    has equal weight and the swapped ids are recovered uniquely.
    """
    reference_ids = np.array([1, 2], dtype=np.int64)
    moving_ids = np.array([1, 2], dtype=np.int64)
    # Rows: habitat 1, habitat 2. Columns: Energy, Coarseness.
    reference = np.array([[8.0e6, 0.12], [2.0e7, 0.03]], dtype=np.float64)
    # Moving id 1 is the bright habitat, id 2 is the dark one, plus shift.
    moving = np.array([[2.0e7 + 1.5e7, 0.03], [8.0e6 + 1.5e7, 0.12]], dtype=np.float64)
    named = match_labels_by_features(reference_ids, reference, moving_ids, moving)
    assert named == {1: 2, 2: 1}


def test_feature_match_not_per_tumor_minmax() -> None:
    """Per-tumour MinMax moves a habitat when that tumour's own range changes.

    Unscaled means of the dark habitat stay close (8e6 vs 9e6). After
    each tumour MinMax, a wider Energy max on B pulls the same biology
    from 0.26 down to 0.11. Cohort z-score on the unscaled rows still
    pairs dark-to-dark / bright-to-bright.
    """
    reference_ids = np.array([1, 2], dtype=np.int64)
    moving_ids = np.array([1, 2], dtype=np.int64)
    reference_raw = np.array([[8.0e6, 0.12], [2.0e7, 0.03]], dtype=np.float64)
    moving_raw = np.array([[2.2e7, 0.04], [9.0e6, 0.11]], dtype=np.float64)
    a_energy = (8.0e6 - 2.0e6) / (2.5e7 - 2.0e6)
    b_energy_wide = (9.0e6 - 5.0e6) / (4.0e7 - 5.0e6)
    b_energy_narrow = (9.0e6 - 5.0e6) / (2.8e7 - 5.0e6)
    assert abs(a_energy - b_energy_wide) > abs(a_energy - b_energy_narrow)
    named = match_labels_by_features(
        reference_ids, reference_raw, moving_ids, moving_raw
    )
    assert named == {1: 2, 2: 1}


def test_feature_match_volume_is_tiebreak_only() -> None:
    """Equal feature rows: volume fraction breaks the remaining tie."""
    reference_ids = np.array([1, 2], dtype=np.int64)
    moving_ids = np.array([1, 2], dtype=np.int64)
    # Identical feature profiles; Hungarian needs a secondary key.
    features = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    mapping = match_labels_by_features(
        reference_ids,
        features,
        moving_ids,
        features,
        standardize="none",
        reference_volumes=np.array([0.8, 0.2]),
        moving_volumes=np.array([0.25, 0.75]),
    )
    assert mapping == {1: 2, 2: 1}


def test_feature_match_pearson_after_zscore() -> None:
    """Pearson cost after cohort z-score still recovers a swapped pair."""
    reference_ids = np.array([1, 2], dtype=np.int64)
    moving_ids = np.array([1, 2], dtype=np.int64)
    reference = np.array(
        [[8.0e6, 0.12, 1.0], [2.0e7, 0.03, 8.0]], dtype=np.float64
    )
    moving = np.array(
        [[2.0e7, 0.03, 8.0], [8.0e6, 0.12, 1.0]], dtype=np.float64
    )
    mapping = match_labels_by_features(
        reference_ids, reference, moving_ids, moving, metric="pearson"
    )
    assert mapping == {1: 2, 2: 1}


def test_feature_match_hungarian_is_one_to_one() -> None:
    """Two moving habitats cannot claim the same reference id."""
    reference_ids = np.array([1, 2], dtype=np.int64)
    moving_ids = np.array([1, 2], dtype=np.int64)
    # Both moving rows are closer to reference 1 than to reference 2.
    reference = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    moving = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float64)
    mapping = match_labels_by_features(
        reference_ids, reference, moving_ids, moving, standardize="none"
    )
    assert mapping == {1: 1, 2: 2}
    assert len(set(mapping.values())) == 2


def test_locked_cohort_scaler_matches_pairwise_zscore() -> None:
    """A scaler fit on stacked rows equals the pairwise default."""
    reference = np.array([[8.0e6, 0.12], [2.0e7, 0.03]], dtype=np.float64)
    moving = np.array([[9.0e6, 0.11], [2.2e7, 0.04]], dtype=np.float64)
    ids = np.array([1, 2], dtype=np.int64)
    location, scale = fit_feature_match_scale((reference, moving))
    locked = match_labels_by_features(
        ids, reference, ids, moving, location=location, scale=scale
    )
    pairwise = match_labels_by_features(ids, reference, ids, moving)
    assert locked == pairwise == {1: 1, 2: 2}


def test_median_reduction_is_robust_to_one_outlier_voxel() -> None:
    """Median summaries ignore a single extreme voxel; mean does not."""
    labels = _two_block_labels()
    image = np.zeros(labels.shape + (2,), dtype=np.float64)
    image[labels == 1] = (8.0e6, 0.12)
    image[labels == 2] = (2.0e7, 0.03)
    image[0, 0, 0] = (8.0e8, 0.12)
    _ids_mean, mean_cent = habitat_intensity_centroids(image, labels, reduction="mean")
    _ids_med, med_cent = habitat_intensity_centroids(image, labels, reduction="median")
    assert mean_cent[0, 0] > med_cent[0, 0]
    assert med_cent[0, 0] == pytest.approx(8.0e6)


def _overlap_table_nested_loop(
    reference: np.ndarray, moving: np.ndarray
) -> np.ndarray:
    """Pre-speedup overlap table: one full-volume scan per id pair."""
    from habit.kernels.habitat_label_match import present_habitat_ids

    ref_ids = present_habitat_ids(reference)
    mov_ids = present_habitat_ids(moving)
    overlap = np.zeros((mov_ids.size, ref_ids.size), dtype=np.int64)
    for column, ref_id in enumerate(ref_ids):
        selector = reference == ref_id
        overlap[:, column] = [
            int(np.count_nonzero(selector & (moving == mov_id)))
            for mov_id in mov_ids
        ]
    return overlap


def test_overlap_bincount_matches_nested_loop() -> None:
    """bincount contingency must be bit-identical to the old nested loop."""
    rng = np.random.default_rng(7)
    # Sparse labels on a large lattice: the case that used to cost O(K^2 N).
    volume = np.zeros((64, 80, 80), dtype=np.int32)
    volume[8:24, 10:40, 12:44] = rng.integers(0, 5, size=(16, 30, 32))
    moving = volume.copy()
    remap = {1: 3, 2: 1, 3: 4, 4: 2}
    for old_id, new_id in remap.items():
        moving[volume == old_id] = new_id
    moving[20:22, 10:16, 12:18] = 2
    _ref_ids, _mov_ids, table = overlap_count_table(volume, moving)
    assert np.array_equal(table, _overlap_table_nested_loop(volume, moving))
    mapping = match_labels_by_overlap(volume, moving)
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-_overlap_table_nested_loop(volume, moving))
    from habit.kernels.habitat_label_match import present_habitat_ids

    ref_ids = present_habitat_ids(volume)
    mov_ids = present_habitat_ids(moving)
    expected = {
        int(mov_ids[row]): int(ref_ids[column])
        for row, column in zip(rows.tolist(), columns.tolist())
    }
    assert mapping == expected


def test_dice_from_mapping_matches_count_nonzero() -> None:
    """Dice helper must match a full-volume count_nonzero scan."""
    reference = _two_block_labels()
    moving = _swap_ids(reference)
    moving[1, 0, 0] = 0
    mapping = match_labels_by_overlap(reference, moving)
    rows = habitat_dice_from_mapping(reference, moving, mapping)
    matched_moving = {ref_id: mov_id for mov_id, ref_id in mapping.items()}
    by_id = {int(hid): (mid, dice, n_ref, n_mov) for hid, mid, dice, n_ref, n_mov in rows}
    for hid in (1, 2):
        mid = int(matched_moving[hid])
        n_ref = int(np.count_nonzero(reference == hid))
        n_mov = int(np.count_nonzero(moving == mid))
        inter = int(np.count_nonzero((reference == hid) & (moving == mid)))
        dice = 2.0 * inter / (n_ref + n_mov)
        assert by_id[hid][0] == mid
        assert by_id[hid][2] == n_ref
        assert by_id[hid][3] == n_mov
        assert by_id[hid][1] == pytest.approx(dice)


def test_adjusted_rand_index_permutation_invariant() -> None:
    """Swapping habitat ids must leave ARI at 1 on an otherwise identical map."""
    reference = _two_block_labels()
    moving = _swap_ids(reference)
    assert adjusted_rand_index(reference, moving) == pytest.approx(1.0)
    assert adjusted_rand_index(reference, reference) == pytest.approx(1.0)


def test_adjusted_rand_index_matches_sklearn_and_ignores_background() -> None:
    """Kernel ARI must match sklearn on jointly labelled voxels only."""
    from sklearn.metrics import adjusted_rand_score

    reference = _two_block_labels()
    moving = reference.copy()
    moving[0:2, 0:2, 0:2] = 2
    moving[2:4, 0:2, 0:2] = 1
    moving[0, 3, 3] = 1
    both = (reference > 0) & (moving > 0)
    expected = float(adjusted_rand_score(reference[both], moving[both]))
    assert adjusted_rand_index(reference, moving) == pytest.approx(expected)
    mask = np.zeros(reference.shape, dtype=bool)
    mask[0:2] = True
    both_m = mask & (reference > 0) & (moving > 0)
    if int(both_m.sum()) >= 2:
        expected_m = float(adjusted_rand_score(reference[both_m], moving[both_m]))
        assert adjusted_rand_index(reference, moving, mask=mask) == pytest.approx(expected_m)


def test_adjusted_rand_index_empty_is_nan() -> None:
    """Fewer than two jointly labelled voxels yield NaN, not a fake 1."""
    empty = np.zeros((3, 3, 3), dtype=np.int32)
    assert np.isnan(adjusted_rand_index(empty, empty))
