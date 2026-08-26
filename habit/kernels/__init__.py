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
"""L0 kernels: pure numerical computation -- no IO, no state, no logging.

Kernels are the independently reviewable mathematical core of HABIT. They
know nothing about subjects, specifications, or the filesystem, and are
usable standalone (e.g. to re-derive a published metric).
"""

from __future__ import annotations

from habit.kernels.cluster_selection import (
    KNEE,
    MAXIMIZE,
    MINIMIZE,
    SCORE_DIRECTIONS,
    best_index,
    gap_statistic,
    knee_index,
    score_direction,
    vote_best_index,
)
from habit.kernels.habitat_graph import (
    GraphNullModelOptions,
    GraphNullModelResult,
    HabitatGraphFeatureOptions,
    build_adjacency_graph,
    build_centroid_distance_graph,
    build_min_distance_graph,
    compare_graph_to_degree_preserving_null,
    extract_graph_features,
    extract_graph_features_for_labels,
    extract_habitat_nodes,
    pair_count,
)
from habit.kernels.habitat_label_match import adjusted_rand_index
from habit.kernels.habitat_metrics import (
    habitat_ith_dispersion,
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    msi_features_from_matrix,
    spatial_interaction_matrix,
)
from habit.kernels.icc import icc2_1, icc3_1, two_way_mean_squares
from habit.kernels.image_perturbation import (
    add_gaussian_noise,
    boundary_band_mask,
    boundary_weighted_perturbation,
    estimate_noise_sigma,
    morphological_grow_shrink,
    rigid_transform_image,
    rotate_image,
    slice_extent_perturbation,
    translate_image,
)
from habit.kernels.label_postprocess import remove_small_connected_components
from habit.kernels.voxel_icc import ICCEstimate, icc2_1_estimate, icc3a_1, icc3c_1
from habit.kernels.voxel_texture import local_entropy_map
from habit.kernels.statistics import (
    compute_midrank,
    delong_roc_ci,
    delong_roc_test,
    delong_roc_variance,
    fast_delong,
    hosmer_lemeshow_test,
    spiegelhalter_z_test,
)

__all__ = [
    "SCORE_DIRECTIONS",
    "MAXIMIZE",
    "MINIMIZE",
    "KNEE",
    "score_direction",
    "knee_index",
    "best_index",
    "vote_best_index",
    "gap_statistic",
    "local_entropy_map",
    "spatial_interaction_matrix",
    "msi_features_from_matrix",
    "habitat_volume_fractions",
    "habitat_region_stats",
    "habitat_ith_dispersion",
    "ith_score",
    "adjusted_rand_index",
    "HabitatGraphFeatureOptions",
    "GraphNullModelOptions",
    "GraphNullModelResult",
    "extract_graph_features",
    "extract_graph_features_for_labels",
    "extract_habitat_nodes",
    "build_centroid_distance_graph",
    "build_min_distance_graph",
    "build_adjacency_graph",
    "pair_count",
    "compare_graph_to_degree_preserving_null",
    "compute_midrank",
    "fast_delong",
    "delong_roc_variance",
    "delong_roc_test",
    "delong_roc_ci",
    "hosmer_lemeshow_test",
    "spiegelhalter_z_test",
    "two_way_mean_squares",
    "icc3_1",
    "icc2_1",
    "estimate_noise_sigma",
    "add_gaussian_noise",
    "translate_image",
    "rotate_image",
    "rigid_transform_image",
    "morphological_grow_shrink",
    "boundary_band_mask",
    "boundary_weighted_perturbation",
    "slice_extent_perturbation",
    "ICCEstimate",
    "icc3a_1",
    "icc3c_1",
    "icc2_1_estimate",
    "remove_small_connected_components",
]
