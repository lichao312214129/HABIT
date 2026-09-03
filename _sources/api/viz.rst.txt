.. _api-viz:

:mod:`habit.viz`: publication overlays
======================================

.. automodule:: habit.viz
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.viz

**User guide:** Habitat Guide
:doc:`../auto_examples/05_quantify/plot_01_volume_fractions`.

``matplotlib`` is imported lazily inside each function, so importing
``habit`` never pulls a plotting backend. All figure labels are
English-only.

Interactive overlay (optional ``[view]`` extra):
:func:`~habit.viz.view_habitat_napari`. Static PNG:
:func:`~habit.viz.plot_habitat_overlay` (pass an ``ImageVolume`` so
coronal/sagittal keep superior up).

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   StyleSpec

Functions
---------

Style helpers
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   use_style
   get_style
   register_style
   available_styles
   sanitize_label

Habitat clustering and overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_habitat_clustering_pca_2d
   plot_habitat_clustering_pca_3d
   plot_habitat_clustering_pca_3d_interactive
   plot_habitat_overlay
   plot_cluster_validation_curves
   plot_cluster_validation_from_report
   plot_habitat_volume_fractions
   plot_msi_matrix
   plot_ith_summary
   plot_habitat_label_compare
   plot_partition_triptych
   plot_precision_icc
   plot_habitat_feature_heatmap
   plot_habitat_feature_effect
   plot_habitat_feature_components
   plot_habitat_feature_violin
   plot_habitat_feature_bars
   plot_habitat_graph_slice
   plot_habitat_graph_network_2d
   plot_graph_feature_heatmap
   render_habitat_graph_surface_3d
   render_habitat_graph_network_3d
   dense_voxel_feature_map
   plot_intensity_slice
   plot_voxel_texture_slice
   view_habitat_napari

Supporting plots (survival / classification / SHAP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bookmarks for table-ML figures. Not the habitat core.

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_kaplan_meier
   plot_risk_triptych
   plot_time_dependent_auc
   plot_survival_calibration
   plot_brier_curve
   plot_cox_forest
   plot_predicted_vs_observed
   plot_residuals
   plot_residual_qq
   plot_bland_altman
   plot_coefficient_forest
   plot_roc
   plot_precision_recall
   plot_calibration
   plot_decision_curve
   plot_confusion_matrix
   plot_shap_summary
   plot_shap_bar
   plot_shap_violin
   plot_shap_heatmap
   plot_shap_dependence
   plot_shap_waterfall
   plot_shap_decision
   plot_shap_force
   plot_permutation_importance
   rank_shap_feature_indices
   select_representative_sample_indices
   net_benefit
