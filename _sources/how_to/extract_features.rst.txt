Feature extraction
==================

Goal: habitat-level feature CSVs from images + ``*_habitats.nrrd``.

Need habitat maps first (:doc:`segment_habitat`).

Run the demo
------------

::

   habit check-config --config config/feature_extraction/config_extract_features_demo.yaml
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Default ``feature_types``: light families (``volume``, ``msi``, ``ith_score``,
``non_radiomics``). Heavy radiomics lines are commented in that YAML — uncomment
when needed. The built-in ``graph`` topology family is opt-in (add it to
``feature_types``; see below).

Your data
---------

★ Edit ``raw_img_folder``, ``habitats_map_folder``, ``out_dir``. Then
``habit check-config`` + ``habit extract``.

Success: CSVs under ``out_dir``.

Graph topology features (built-in)
----------------------------------

``graph`` is a **built-in** light habitat feature family (not a private
HABIT-v2-only plugin). Prefer the domain / public API path; ``habit.compat``
graph shims are deprecated transitional loaders.

CLI / YAML
~~~~~~~~~~

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

Full parameter list: :doc:`../configuration/feature_extraction` and
:doc:`../reference/features/graph`.

Python API
~~~~~~~~~~

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

Class form: :class:`~habit.domain.GraphHabitatFeatures`. Synthetic runnable
snippet: :doc:`../examples/graph_features`.

Next: :doc:`train_model`.
