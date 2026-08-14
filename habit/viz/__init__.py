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
"""Publication figures and image overlays for HABIT results.

This package is the home for HABIT visualization: habitat-core figures
(overlays, partition triptych, auto-K curves, volume/MSI/ITH, label compare),
graph / voxel-texture panels, greyscale intensity slices for image
preprocessing (:func:`plot_intensity_slice`), habitat-clustering PCA,
optional napari (:func:`view_habitat_napari`), plus ROC / survival /
regression helpers.
New drawing / viewing code should land here rather than in CLI or
``habit.api`` helpers.

Most functions in this package are PURE matplotlib helpers: they take contract
objects (or plain arrays), draw on a ``Figure``, and return that figure.
Nothing here touches the filesystem -- there is no ``savefig``, no output-
directory parameter. Where a figure ends up is entirely the caller's
decision. The optional napari helper is the one exception that may open a Qt
window; it still takes arrays only and never reads or writes image files.

Two consequences of that rule:

- a figure's geometry, typography and palette come from a STYLE PRESET
  (:func:`use_style`), never from a per-plot hard-coding, so one figure can be
  re-rendered for a different journal without touching the plotting code;
- every piece of text drawn on a figure is guaranteed ASCII via
  :func:`~habit.viz.labels.sanitize_label`, because data-driven labels (a
  feature or group name) can otherwise leak non-ASCII characters onto an
  axis that a journal will reject.
"""

from __future__ import annotations

from habit.viz.classification import (
    net_benefit,
    plot_calibration,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_permutation_importance,
    plot_precision_recall,
    plot_roc,
    plot_shap_bar,
    plot_shap_decision,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_heatmap,
    plot_shap_summary,
    plot_shap_violin,
    plot_shap_waterfall,
    rank_shap_feature_indices,
    select_representative_sample_indices,
)
from habit.viz.habitat_features import (
    plot_habitat_feature_bars,
    plot_habitat_feature_components,
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
)
from habit.viz.labels import sanitize_label
from habit.viz.regression import (
    plot_bland_altman,
    plot_coefficient_forest,
    plot_predicted_vs_observed,
    plot_residual_qq,
    plot_residuals,
)
from habit.viz.style import (
    StyleSpec,
    available_styles,
    get_style,
    register_style,
    use_style,
)
from habit.viz.habitat_clustering import (
    plot_habitat_clustering_pca_2d,
    plot_habitat_clustering_pca_3d,
    plot_habitat_clustering_pca_3d_interactive,
)
from habit.viz.habitat_graph import (
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
    render_habitat_graph_network_3d,
    render_habitat_graph_surface_3d,
)
from habit.viz.habitat_core import (
    plot_cluster_validation_curves,
    plot_cluster_validation_from_report,
    plot_habitat_label_compare,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)
from habit.viz.habitat_overlay import plot_habitat_overlay
from habit.viz.habitat_napari import view_habitat_napari
from habit.viz.intensity import plot_intensity_slice
from habit.viz.voxel_texture import dense_voxel_feature_map, plot_voxel_texture_slice
from habit.viz.survival import (
    plot_brier_curve,
    plot_cox_forest,
    plot_kaplan_meier,
    plot_risk_triptych,
    plot_survival_calibration,
    plot_time_dependent_auc,
)

__all__ = [
    "StyleSpec",
    "use_style",
    "get_style",
    "register_style",
    "available_styles",
    "sanitize_label",
    # survival
    "plot_kaplan_meier",
    "plot_risk_triptych",
    "plot_time_dependent_auc",
    "plot_survival_calibration",
    "plot_brier_curve",
    "plot_cox_forest",
    # regression
    "plot_predicted_vs_observed",
    "plot_residuals",
    "plot_residual_qq",
    "plot_bland_altman",
    "plot_coefficient_forest",
    # classification
    "plot_roc",
    "plot_precision_recall",
    "plot_calibration",
    "plot_decision_curve",
    "plot_confusion_matrix",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_violin",
    "plot_shap_heatmap",
    "plot_shap_decision",
    "plot_shap_force",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "plot_permutation_importance",
    "rank_shap_feature_indices",
    "select_representative_sample_indices",
    "net_benefit",
    # habitat clustering
    "plot_habitat_clustering_pca_2d",
    "plot_habitat_clustering_pca_3d",
    "plot_habitat_clustering_pca_3d_interactive",
    # habitat overlay on source image
    "plot_habitat_overlay",
    # habitat core analysis figures (validation / map features / compare)
    "plot_cluster_validation_curves",
    "plot_cluster_validation_from_report",
    "plot_habitat_volume_fractions",
    "plot_msi_matrix",
    "plot_ith_summary",
    "plot_habitat_label_compare",
    "plot_partition_triptych",
    "plot_habitat_feature_heatmap",
    "plot_habitat_feature_effect",
    "plot_habitat_feature_components",
    "plot_habitat_feature_violin",
    "plot_habitat_feature_bars",
    # habitat graph topology figures (2D: [viz] extra; 3D: [view]+[slic] extras)
    "plot_habitat_graph_slice",
    "plot_habitat_graph_network_2d",
    "render_habitat_graph_surface_3d",
    "render_habitat_graph_network_3d",
    # greyscale anatomy / intensity slices (image preprocessing, not texture)
    "plot_intensity_slice",
    # voxel texture / feature-map slices (local entropy, voxel radiomics, ...)
    "dense_voxel_feature_map",
    "plot_voxel_texture_slice",
    # optional interactive napari viewer (requires [view] extra)
    "view_habitat_napari",
]
