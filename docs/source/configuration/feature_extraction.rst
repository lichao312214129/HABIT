Feature Extraction Configuration
==================================

This section documents **feature extraction** configuration. CLI: ``habit extract -c <yaml>``. Demo example: ``config/feature_extraction/config_extract_features_demo.yaml``.

**Example configuration file:**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./demo_data/preprocessed
   habitats_map_folder: ./demo_data/results/habitat_two_step
   out_dir: ./demo_data/results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - volume
     - msi
     - ith_score
     - non_radiomics
     - graph
     # Heavy PyRadiomics families (opt-in; require pyradiomics):
     # - traditional
     # - whole_habitat
     # - each_habitat

   # Optional settings when feature_types includes graph (stripped before
   # FeatureExtractionConfig validation; see GraphFeatureBlock):
   # graph:
   #   edge_method: min_distance
   #   node_method: uniform_grid
   #   block_size: 8
   #   visualize: false

   n_habitats:

   debug: false

**params_file_of_non_habitat**: parameter file for features extracted from raw images

- **Type**: string
- **Required**: no
- **Default**: ``null`` (bundled ``roi`` preset → ``habit/resources/radiomics/parameter.yaml``)
- **Description**: PyRadiomics parameter file for traditional / each_habitat radiomics on raw images
- **Example**: ``./parameter.yaml``

**params_file_of_habitat**: parameter file for features extracted from habitat maps

- **Type**: string
- **Required**: no
- **Default**: ``null`` (bundled ``habitat`` preset → ``habit/resources/radiomics/parameter_habitat.yaml``)
- **Description**: PyRadiomics parameter file for whole_habitat radiomics on the label map
- **Example**: ``./parameter_habitat.yaml``

**raw_img_folder**: root directory of raw images

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: contains preprocessed images
- **Example**: ``./demo_data/preprocessed``

**habitats_map_folder**: root directory of habitat maps

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: contains generated habitat maps
- **Example**: ``./results/habitat``

**out_dir**: output directory

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: feature files are saved here
- **Example**: ``./results/features``

**debug** (``FeatureExtractionConfig``)

- **Type**: boolean
- **Default**: ``false``

**n_processes**: number of parallel processes

- **Type**: integer
- **Required**: no
- **Default**: ``4`` (built-in default for feature extraction config)
- **Description**: number of processes for parallel processing.
  When ``n_processes > 1``, Windows scripts that call the extract recipe
  from Python must use ``if __name__ == "__main__":`` (same spawn rule as
  habitat :class:`~habit.execution.ProcessPoolBackend`; see
  :doc:`../api/execution`).
- **Example**: ``3``

**habitat_pattern**: habitat file glob pattern

- **Type**: string
- **Required**: no
- **Default**: ``'*_habitats.nrrd'``
- **Description**: pattern to match habitat map files; supports wildcards (``*``)
- **Example**: ``*_habitats.nrrd``

**feature_types**: list of feature types

- **Type**: list
- **Required**: yes
- **Default**: none (required; at least one item)
- **Description**: types not in the list are not extracted
- **Allowed values**: ``volume``, ``msi``, ``ith_score``, ``non_radiomics``, ``graph``, ``traditional``, ``whole_habitat``, ``each_habitat``
- **Example**: ``[volume, msi, ith_score, non_radiomics, graph]`` (the shipped default light set; add heavy radiomics when needed)
- **Meanings and references per type**: see :doc:`../reference/features/index`

**graph**: optional top-level block for the built-in graph topology family

- **Type**: mapping (validated as ``GraphFeatureBlock``)
- **Required**: no (defaults apply when ``graph`` is listed in ``feature_types`` without a block)
- **Description**: extraction options mirror
  :class:`~habit.domain.GraphHabitatFeaturesParams`; visualization fields are
  consumed by the extract recipe only. ``graph`` is a **built-in** family — not
  a private plugin. Prefer domain / API paths; ``habit.compat.graph_plugin`` is
  deprecated.
- **Key extraction fields** (defaults in parentheses):

  - ``include_single_habitat_graph`` (``true``) / ``include_pairwise_habitat_graph`` (``true``)
  - ``edge_method``: ``min_distance`` (default), ``adjacency``, or ``centroid_distance``
  - ``distance_threshold`` (``5.0``, voxel-index units) — used by ``centroid_distance`` and ``min_distance``. With default 8-voxel cubes, face-adjacent cubes connect; one empty lattice cell (closest-voxel distance about 8) stays disconnected.
  - ``adjacency_connectivity`` (``corner``: 8-conn in 2D / 26-conn in 3D; ``face`` = 4/6 remains available) / ``adjacency_min_voxels`` (``10``) — used by ``adjacency``. An edge exists when two regions are adjacent and the contact voxel count is >= 10, measured on the habitat labels as drawn (default ``erosion_radius`` is ``0``).
  - ``edge_weight``: ``none`` | ``distance`` | ``inverse_distance`` | ``contact_voxels``
  - ``min_region_voxels`` (``1``), ``connectivity`` (default ``full``: 8-conn in 2D / 26-conn in 3D; ``face`` = 4/6 remains available)
  - ``node_method`` (``uniform_grid`` default; ``component`` for connected-component nodes)
  - ``erosion_radius`` (``0`` / off; set ``>= 1`` to shrink habitats before edges), ``subdivide_region_voxels`` (``1000``; used only by ``component``)
  - ``block_size`` (``8`` voxels, not millimetres), ``block_min_coverage`` (``0.2``)
  - ``pairwise_include_intra_edges`` (``true``)
  - ``include_extended_metrics`` (``false``; set ``true`` to opt in), ``extended_min_nodes`` (``10``)
  - ``graph_null_sampler`` (``analytic``): one ``small_world_sigma`` column. ``analytic`` is Humphries *S* vs an Erdős–Rényi graph (same *n*, *m*; closed-form :math:`C_{rand}`, :math:`L_{rand}`). ``config`` is the configuration model; ``rewire`` is Maslov–Sneppen (NetworkX ``sigma``). The last two **replace** the analytic value with a degree-preserving ensemble (``small_world_nrand`` / ``small_world_niter``, default ``100``). See :doc:`../reference/features/graph`.
  - ``rich_club_q`` (``100``): mixing floor for ``rewire``; analytic rich-club uses one configuration-model graph
  - ``graph_null_device`` (``auto``)
  - ``graph_metric_backend`` (``networkx``; optional ``igraph`` / ``auto`` after ``pip install habitat-analysis[igraph]``)

- **Visualization fields** (recipe hook; not part of the extractor ``Spec``):

  - ``visualize`` (``false``) → writes ``<out_dir>/visualizations/graph/``
  - ``visualization_format``: ``png`` | ``pdf`` | ``both`` (default)
  - ``visualization_dpi`` (``600``)
  - ``visualization_show_background`` (``true``)
  - ``visualization_show_grid`` (``true``)
  - ``visualization_block_size`` (``null`` → extraction ``block_size``, default 8 voxels)
  - ``visualization_grid_linestyle`` (``--`` dashed)
  - ``visualization_save_3d`` (``true``; 3D needs optional ``[view]`` stack)

- **Legacy keys** ``enabled`` / ``n_workers``: accepted, ignored (activation is
  ``feature_types``; figures run serially in the main process)
- **Output CSV**: ``habitat_graph_features.csv``
- **Reference**: :doc:`../reference/features/graph`

**n_habitats**: number of habitats

- **Type**: integer or null
- **Required**: no
- **Default**: ``null`` (auto-detect)
- **Description**: can manually specify habitat count
- **Example**: ``null``
