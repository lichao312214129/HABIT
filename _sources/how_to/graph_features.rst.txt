Graph topology features
=======================

Goal: extract built-in habitat **graph topology** features (nodes / edges /
metrics) from habitat label maps, optionally with 2D/3D figures.

Need habitat maps first (:doc:`segment_habitat`). Full column definitions:
:doc:`../reference/features/graph`. Short end-to-end gallery (one subject →
one-step → graph plot): :doc:`../examples/graph_features`.

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
     edge_method: centroid_distance
     distance_threshold: 5.0
     erosion_radius: 1
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

Parameter reference: :doc:`../configuration/feature_extraction`.

Python API
----------

Kernel helper (arrays only)::

   from habit import HabitatGraphFeatureOptions, extract_graph_features

   feats = extract_graph_features(
       label_array,
       options=HabitatGraphFeatureOptions(distance_threshold=5.0),
       expected_labels=(1, 2, 3),
   )

Registry / domain::

   import habit.domain
   from habit.domain import HabitatFeatureExtractorRegistry

   table = HabitatFeatureExtractorRegistry.create(
       "graph",
       edge_method="centroid_distance",
       distance_threshold=5.0,
   )(subject, habitat_map)

2D plot: :func:`~habit.viz.plot_habitat_graph_network_2d`. 3D (optional)::

   from habit.viz import render_habitat_graph_network_3d, render_habitat_graph_surface_3d

Also see
--------

* General extract how-to: :doc:`extract_features`
* Examples gallery: :doc:`../examples/graph_features`
* Feature columns: :doc:`../reference/features/graph`
