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
    fit_feature_match_scale,
    habitat_dice_from_mapping,
    habitat_intensity_centroids,
    match_labels_by_overlap,
    match_rows_to_prototypes,
    overlap_count_table,
    present_habitat_ids,
    remap_label_array,
)


def _align(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Overlap pairing + collision-safe remap (what align_habitat_map does)."""
    return remap_label_array(
        moving,
        match_labels_by_overlap(reference, moving),
        reserved_ids=present_habitat_ids(reference).tolist(),
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
    aligned = _align(reference, moving)
    assert np.array_equal(aligned, reference)
    assert np.array_equal(reference, _three_block_labels())
    raw_disagree = (moving != reference) & ((moving > 0) | (reference > 0))
    aligned_disagree = (aligned != reference) & ((aligned > 0) | (reference > 0))
    assert int(np.count_nonzero(raw_disagree)) > 0
    assert int(np.count_nonzero(aligned_disagree)) == 0


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
    aligned = _align(reference, moving)
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
    aligned = _align(reference, moving)
    assert np.all(aligned[moving == 2] == 1)
    assert np.all(aligned[moving == 3] == 2)
    leftover_id = int(aligned[moving == 1][0])
    assert leftover_id == 3
    assert leftover_id not in (1, 2)
    assert np.all(aligned[moving == 1] == leftover_id)


def test_align_identity_when_already_matched() -> None:
    """Already-aligned maps stay unchanged."""
    reference = _two_block_labels()
    assert np.array_equal(_align(reference, reference), reference)


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


def test_cohort_scaler_is_pooled_zscore() -> None:
    """fit_feature_match_scale = mean / population std of the stacked rows."""
    reference = np.array([[8.0e6, 0.12], [2.0e7, 0.03]], dtype=np.float64)
    moving = np.array([[9.0e6, 0.11], [2.2e7, 0.04]], dtype=np.float64)
    location, scale = fit_feature_match_scale((reference, moving))
    stacked = np.vstack((reference, moving))
    np.testing.assert_allclose(location, stacked.mean(axis=0))
    np.testing.assert_allclose(scale, stacked.std(axis=0, ddof=0))


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


def _brute_force_prototype_objective(blocks) -> float:
    """Exhaustive minimum of the within-group sum of squares (block 0 fixed)."""
    import itertools

    k = blocks[0].shape[0]
    best = np.inf
    for perms in itertools.product(
        itertools.permutations(range(k)), repeat=len(blocks) - 1
    ):
        groups = [[blocks[0][j]] for j in range(k)]
        for block, perm in zip(blocks[1:], perms):
            for row, proto in enumerate(perm):
                groups[proto].append(block[row])
        cost = sum(
            float(np.sum((np.array(g) - np.mean(g, axis=0)) ** 2)) for g in groups
        )
        best = min(best, cost)
    return best


def test_prototype_matching_reaches_brute_force_optimum() -> None:
    """Well-separated cohorts: multi-start search equals exhaustive search."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        centres = rng.normal(size=(3, 2)) * 4
        blocks = [
            (centres + rng.normal(scale=0.5, size=(3, 2)))[rng.permutation(3)]
            for _ in range(4)
        ]
        found = match_rows_to_prototypes(blocks)
        assert found.converged
        assert found.objective == pytest.approx(
            _brute_force_prototype_objective(blocks), rel=1e-9
        )


def test_prototype_matching_one_row_per_prototype_per_block() -> None:
    """Cannot-link: a block never uses the same prototype twice (any metric)."""
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(3, 4)) + 5.0 for _ in range(6)]
    for metric in ("sqeuclidean", "manhattan", "cosine", "correlation"):
        found = match_rows_to_prototypes(blocks, metric=metric)
        assert found.metric == metric
        assert found.converged
        for assignment in found.assignments:
            matched = assignment[assignment >= 0]
            assert matched.size == np.unique(matched).size == 3


def test_two_subjects_equal_pairwise_squared_hungarian() -> None:
    """Two blocks: prototype grouping = pairwise Hungarian on squared distance."""
    import itertools

    rng = np.random.default_rng(11)
    for _ in range(30):
        a = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 3))
        # Exhaustive pairwise optimum on squared Euclidean distance.
        best_perm = min(
            itertools.permutations(range(4)),
            key=lambda perm: float(np.sum((a - b[list(perm)]) ** 2)),
        )
        found = match_rows_to_prototypes([a, b])
        pairs = {
            int(found.assignments[0][i]): int(found.assignments[1][best_perm[i]])
            for i in range(4)
        }
        assert all(key == value for key, value in pairs.items())


def test_objective_decreases_for_every_metric() -> None:
    """The fitted objective is never above the seed-only assignment cost."""
    rng = np.random.default_rng(5)
    blocks = [rng.normal(size=(3, 5)) + 3.0 for _ in range(8)]
    for metric in ("sqeuclidean", "manhattan", "cosine", "correlation"):
        fitted = match_rows_to_prototypes(blocks, metric=metric)
        seed = blocks[fitted.init_block]
        frozen = match_rows_to_prototypes(blocks, metric=metric, prototypes=seed)
        assert fitted.objective <= frozen.objective + 1e-12


def test_manhattan_prototype_is_median() -> None:
    """k-medians update: one outlying habitat does not drag the prototype."""
    blocks = [np.array([[0.0], [10.0]]) + shift for shift in (0.0, 0.1, 0.2)]
    blocks.append(np.array([[0.3], [100.0]]))
    found = match_rows_to_prototypes(blocks, metric="manhattan")
    np.testing.assert_allclose(found.prototypes.ravel(), [0.15, 10.15])


def test_cosine_ignores_magnitude() -> None:
    """Cosine pairs by direction: (0.5, 0.6) and (2.0, 2.4) are one habitat.

    Euclidean pairs the two weak habitats (0.5, 0.6) / (0.4, 0.05) instead,
    because it compares values, not ratios.
    """
    a = np.array([[0.5, 0.6], [2.0, 0.2]])
    b = np.array([[0.4, 0.05], [2.0, 2.4]])
    cos = match_rows_to_prototypes([a, b], metric="cosine")
    assert cos.assignments[0][0] == cos.assignments[1][1]
    assert cos.distances[1][1] == pytest.approx(0.0, abs=1e-12)
    euc = match_rows_to_prototypes([a, b])
    assert euc.assignments[0][0] == euc.assignments[1][0]


def test_correlation_rejects_constant_rows() -> None:
    """Pearson r is undefined for one feature / a flat profile."""
    with pytest.raises(ValueError, match="constant across features"):
        match_rows_to_prototypes(
            [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])],
            metric="correlation",
        )
    with pytest.raises(ValueError, match="all zeros"):
        match_rows_to_prototypes(
            [np.array([[0.0, 0.0], [1.0, 2.0]])], metric="cosine"
        )


def test_frozen_prototypes_keep_order_and_leave_extras_unmatched() -> None:
    """Frozen prototypes: no refit, given order, extra habitats unnamed."""
    trained = np.array([[5.0], [1.0]])
    new = [np.array([[0.9], [5.2], [20.0]]), np.array([[1.1]])]
    found = match_rows_to_prototypes(new, prototypes=trained)
    np.testing.assert_array_equal(found.prototypes, trained)
    assert found.init_block == -1
    # Hungarian on 3 rows vs 2 prototypes leaves the farthest row out.
    assert found.assignments[0].tolist() == [1, 0, -1]
    assert found.assignments[1].tolist() == [1]
    assert np.isnan(found.distances[0][2])


def test_unknown_prototype_metric_raises() -> None:
    with pytest.raises(ValueError, match="metric"):
        match_rows_to_prototypes([np.zeros((2, 2))], metric="chebyshev")  # type: ignore[arg-type]
