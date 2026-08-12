Graph topology features
=======================

Goal: extract built-in habitat **graph topology** features (nodes / edges /
metrics) from ``*_habitats.nrrd``, optionally with 2D/3D figures.

Need habitat maps first (:doc:`segment_habitat`). Full column definitions:
:doc:`../reference/features/graph`. Runnable gallery (demo_data figures):
:doc:`../examples/graph_features`.

``graph`` is a **built-in** light family (same tier as ``volume`` / ``msi`` /
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
     # visualization_format: both
     # visualization_dpi: 600

Outputs:

* ``habitat_graph_features.csv`` under ``out_dir``
* when ``graph.visualize: true``, optional figures under
  ``out_dir/visualizations/graph/`` (2D needs ``[viz]``; 3D also needs
  ``[view]``)

Parameter reference: :doc:`../configuration/feature_extraction`.

Python API
----------

Registry / domain (preferred for notebooks and third-party pipelines)::

   import habit.domain  # registers built-ins
   from habit.domain import HabitatFeatureExtractorRegistry

   graph_fx = HabitatFeatureExtractorRegistry.create(
       "graph",
       edge_method="centroid_distance",
       distance_threshold=5.0,
   )
   table = graph_fx(subject, habitat_map)  # FeatureTable

Kernel helper (arrays only; same numeric definitions)::

   from habit import HabitatGraphFeatureOptions, extract_graph_features

   feats = extract_graph_features(
       label_array,
       options=HabitatGraphFeatureOptions(distance_threshold=5.0),
       expected_labels=(1, 2, 3),
   )

Class form: :class:`~habit.domain.GraphHabitatFeatures`.

Also see
--------

* General extract how-to: :doc:`extract_features`
* Examples gallery page: :doc:`../examples/graph_features`
* Feature columns: :doc:`../reference/features/graph`
