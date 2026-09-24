4. Habitat Mapping
==================

A habitat map is a label volume. Integer ids are arbitrary until two
maps share one fitted model, or until labels are matched after the fact.

Use at least two subjects for two-step and for pooling. One subject has
no shared cohort to fit. Applying a saved model to a validation cohort
keeps the training definition. Fitting that cohort again creates a
different definition. Do not compare colours across the three design
pages until labels have been matched.

Choosing how habitats are defined
----------------------------------

Same modalities (``LAP``), fixed ``n_habitats=3``, one overlay and one
volume table on each page.

* **Defining habitats in two steps** (partition, then a shared model) —
  :doc:`/auto_examples/04_habitat_maps/plot_01_two_step`.
* **Defining habitats inside each subject** (ids do not match across people) —
  :doc:`/auto_examples/04_habitat_maps/plot_02_inside_each_subject`.
* **Pooling voxels across the cohort** (no supervoxels) —
  :doc:`/auto_examples/04_habitat_maps/plot_03_pool_voxels`.
* **Applying a saved habitat model** —
  :doc:`/auto_examples/04_habitat_maps/plot_04_apply_saved_model`.
* **Matching habitat labels across subjects** (shared or frozen
  prototypes) — :doc:`/auto_examples/04_habitat_maps/plot_05_match_labels`.
  Which matcher to use and why: :doc:`/reference/habitat_matching`;
  worked, visual pages: :doc:`/auto_examples/07_habitat_matching/index`.

Building a habitat map step by step
------------------------------------

``VoxelFeatureField`` → ``Supervoxelization`` → ``HabitatModel`` →
``HabitatMap`` → ``FeatureTable``. Each page names its input, its output,
and the :class:`~habit.spec.HabitatSpec` stage.

* **Extracting voxel intensities** —
  :doc:`/auto_examples/04_habitat_maps/plot_06_voxel_intensities`.
* **Extracting a voxel texture** —
  :doc:`/auto_examples/04_habitat_maps/plot_07_voxel_texture`.
* **Clustering habitats from a derived map** —
  :doc:`/auto_examples/04_habitat_maps/plot_08_derived_map`.
* **Clustering habitats from a texture field** —
  :doc:`/auto_examples/04_habitat_maps/plot_09_texture_habitats`.
* **Preprocessing features before clustering** —
  :doc:`/auto_examples/04_habitat_maps/plot_10_preprocess_features`.
* **Comparing habitat maps with and without preprocessing** —
  :doc:`/auto_examples/04_habitat_maps/plot_11_preprocess_compare`.
* **Partitioning a ROI into supervoxels** —
  :doc:`/auto_examples/04_habitat_maps/plot_12_supervoxels`.
* **Fitting a cohort habitat model** —
  :doc:`/auto_examples/04_habitat_maps/plot_13_fit_model`.
* **Assigning habitat labels** —
  :doc:`/auto_examples/04_habitat_maps/plot_14_assign_labels`.
