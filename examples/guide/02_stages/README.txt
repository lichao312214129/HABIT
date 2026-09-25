2. Each stage
=============

**Background.** A two-step habitat analysis is a chain of stages:
extract voxel features, preprocess them, partition each tumour into
supervoxels, pool the cohort, fit one habitat model, assign labels.

**Purpose.** Each page runs one stage on demo data so you can see its
input, its output, and the parameters that matter.

These pages open the stage list of the
:doc:`complete analysis </auto_examples/00_full_pipeline/plot_01_full_pipeline>`.
``Stage``'s first argument is a label. ``Spec`` names the component.

* **extract** — :doc:`/auto_examples/02_stages/plot_06_voxel_intensities`,
  :doc:`/auto_examples/02_stages/plot_01_feature_routes`,
  :doc:`/auto_examples/02_stages/plot_02_expression`,
  :doc:`/auto_examples/02_stages/plot_02_custom_features`,
  :doc:`/auto_examples/02_stages/plot_03_voxel_texture`,
  :doc:`/auto_examples/02_stages/plot_07_voxel_texture`,
  :doc:`/auto_examples/02_stages/plot_08_derived_map`,
  :doc:`/auto_examples/02_stages/plot_09_texture_habitats`.
* **preprocess** — :doc:`/auto_examples/02_stages/plot_04_feature_preprocessing`,
  :doc:`/auto_examples/02_stages/plot_10_preprocess_features`,
  :doc:`/auto_examples/02_stages/plot_11_preprocess_compare`,
  :doc:`/auto_examples/02_stages/plot_06_texture_preprocessing`.
* **partition** — :doc:`/auto_examples/02_stages/plot_12_supervoxels`,
  :doc:`/auto_examples/02_stages/plot_05_supervoxel_features`.
* **fit** — :doc:`/auto_examples/02_stages/plot_13_fit_model`.
* **assign** — :doc:`/auto_examples/02_stages/plot_14_assign_labels`.
