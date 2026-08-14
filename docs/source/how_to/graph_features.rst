Graph topology features
=======================

Goal: extract built-in habitat **graph topology** features (nodes / edges /
metrics) from habitat label maps, optionally with 2D/3D figures.

Need habitat maps first (:doc:`segment_habitat`). Reviewer-grade formulas
(nodes, edges, metrics, VOI normalization):
:doc:`../reference/features/graph`. Short end-to-end gallery:
:doc:`../examples/graph_features` (one-step with **fixed** ``n_habitats=10``,
then graph features + overlay / 2D network).

``graph`` is a **built-in** light family under
:doc:`../reference/features/index` (same tier as ``volume`` / ``msi`` /
``ith_score``). Prefer the domain / public API; ``habit.compat`` graph shims
are deprecated transitional loaders.

CLI / YAML
----------

List ``graph`` under ``feature_types`` and optionally tune a top-level
``graph:`` block (validated as
:class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`). One physical
line per shell command (Windows / conda PowerShell)::

   habit extract --config path/to/your_extract_with_graph.yaml

Minimal YAML fragment::

   feature_types:
     - volume
     - graph

   graph:
     edge_method: adjacency
     adjacency_connectivity: face
     adjacency_min_voxels: 10
     erosion_radius: 0
     subdivide_region_voxels: 1000
     include_single_habitat_graph: true
     include_pairwise_habitat_graph: true
     include_extended_metrics: true
     visualize: false

Outputs:

* ``habitat_graph_features.csv`` under ``out_dir``
* when ``graph.visualize: true``, optional figures under
  ``out_dir/visualizations/graph/`` (2D needs ``[viz]``; 3D also needs
  ``[view]``)

By default there is **no morphological erosion** (``erosion_radius: 0``).
An **edge exists when two regions are adjacent** (face-sharing voxels;
``adjacency_connectivity: face``) **and** the contact (shared-boundary)
voxel count is **>= 10** (``adjacency_min_voxels: 10``), measured on the
habitat labels as drawn. Set ``erosion_radius`` to a positive integer if
you want to shrink each habitat before building edges. Use
``edge_method: centroid_distance`` plus ``distance_threshold`` if you
want the older centroid-proximity rule instead.

Parameter reference: :doc:`../configuration/feature_extraction`.

Python API
----------

The figure below is written by the graph gallery
(:doc:`../examples/graph_features`) — one-step with **fixed**
``n_habitats=10``, then the same plot call. It is **not** from the YAML
fragment above. Reproduce it::

   python docs/source/examples/scripts/graph_features_demo.py

Or paste the same code the gallery shows::

   from habit import cohort_from_directory, extract_graph_features, one_step_habitat
   from habit.viz import plot_habitat_graph_network_2d

   DATA = "demo_data/preprocessed"
   MODALITIES = ("LAP",)
   ROI = "LAP"
   cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
   result = one_step_habitat(
       modalities=MODALITIES, n_habitats=10, random_seed=0, roi=ROI
   ).fit_predict(cohort)
   labels = result.habitat_maps[0].label_array
   feats = extract_graph_features(labels)
   fig = plot_habitat_graph_network_2d(labels)

Optional: ``HabitatGraphFeatureOptions(adjacency_min_voxels=...)``, registry
``HabitatFeatureExtractorRegistry.create("graph", ...)``, and 3D
:func:`~habit.viz.render_habitat_graph_network_3d` /
:func:`~habit.viz.render_habitat_graph_surface_3d` (needs ``[view]``).

.. figure:: ../_static/images/examples/graph_habitat_network_2d.png
   :alt: Habitat graph network on a 2D slice
   :width: 520

   Same file the gallery script writes to ``out/graph_habitat_network_2d.png``.

Also see
--------

* General extract how-to: :doc:`extract_features`
* Examples gallery: :doc:`../examples/graph_features`
* Feature columns: :doc:`../reference/features/graph`
