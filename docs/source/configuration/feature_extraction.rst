Feature Extraction Configuration
==================================

This section documents **feature extraction** configuration. CLI: ``habit extract -c <yaml>``. Demo example: ``config/feature_extraction/config_extract_features_demo.yaml``.

**Example configuration file:**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./demo_data/preprocessed/processed_images
   habitats_map_folder: ./demo_data/results/habitat_two_step
   out_dir: ./demo_data/results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - traditional
     - non_radiomics
     - whole_habitat
     - each_habitat
     - msi
     - ith_score
     - graph

   n_habitats:

   graph:
     include_single_habitat_graph: true
     include_pairwise_habitat_graph: true
     edge_method: centroid_distance
     distance_threshold: 5
     adjacency_connectivity: face
     adjacency_min_voxels: 1
     edge_weight: none
     min_region_voxels: 1
     connectivity: face
     erosion_radius: 1
     subdivide_region_voxels: 1000
     block_size: 5
     block_min_coverage: 0.5
     pairwise_include_intra_edges: true
     include_extended_metrics: true
     extended_min_nodes: 10
     visualize: false
     visualization_format: both
     visualization_dpi: 600
     visualization_show_background: true

   debug: false

**params_file_of_non_habitat**: parameter file for features extracted from raw images

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: PyRadiomics parameter file for traditional radiomics features from raw images
- **Example**: ``./parameter.yaml``

**params_file_of_habitat**: parameter file for features extracted from habitat maps

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: PyRadiomics parameter file for features from habitat maps
- **Example**: ``./parameter_habitat.yaml``

**raw_img_folder**: root directory of raw images

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: contains preprocessed images
- **Example**: ``./preprocessed/processed_images``

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
- **Description**: number of processes for parallel processing
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
- **Allowed values**: ``traditional``, ``non_radiomics``, ``whole_habitat``, ``each_habitat``, ``msi``, ``ith_score``, ``graph``
- **Example**: ``[traditional, non_radiomics, whole_habitat]``
- **Meanings and references per type**: see :doc:`../reference/features/index`

**n_habitats**: number of habitats

- **Type**: integer or null
- **Required**: no
- **Default**: ``null`` (auto-detect)
- **Description**: can manually specify habitat count
- **Example**: ``null``

**graph**: graph topology feature parameters (``GraphFeatureConfig``; active only when ``feature_types`` includes ``graph``)

Builds single-tissue graphs (within each habitat) and multi-tissue graphs (between habitat pairs) on habitat label maps, extracts graph topology metrics, outputs ``habitat_graph_features.csv``. Metric and column meanings: :doc:`../reference/features/graph/index`.

- **include_single_habitat_graph**: boolean, default ``true``; compute single-tissue (within-habitat) graphs.
- **include_pairwise_habitat_graph**: boolean, default ``true``; compute multi-tissue (pairwise habitat) graphs.
- **edge_method**: string, default ``centroid_distance``; edge construction, ``centroid_distance`` / ``adjacency``.
- **distance_threshold**: float, default ``5``; centroid distance threshold (pixel/voxel units); connect when :math:`\le` threshold (when ``edge_method=centroid_distance``).
- **adjacency_connectivity**: string, default ``face``; voxel adjacency rule for adjacency edges (when ``edge_method=adjacency``); ``face`` (3D face-adjacent / 6-connectivity), ``edge`` (18-connectivity), ``corner`` (26-connectivity).
- **adjacency_min_voxels**: integer, default ``1``; minimum adjacent voxel pairs required for an adjacency edge (when ``edge_method=adjacency``).
- **edge_weight**: string, default ``none``; edge weight source: ``none`` / ``distance`` / ``inverse_distance`` / ``contact_voxels`` (``contact_voxels`` uses contact voxel pair count as weight in ``adjacency`` mode).
- **min_region_voxels**: integer, default ``1``; connected components smaller than this are ignored (not nodes).
- **connectivity**: string, default ``face``; connected-component adjacency rule: ``face`` (face-adjacent) or ``full`` (includes diagonals).
- **erosion_radius**: integer, default ``1``; binary erosion iterations before labeling connected components; reduces boundary noise; ``0`` disables.
- **subdivide_region_voxels**: integer, default ``1000``; subdivide connected components larger than this into grid blocks; ``0`` disables. **Key knob** — strongly modality/resolution dependent; for 2D or small VOI, try 200–500.
- **block_size**: integer, default ``5``; grid block edge length (voxels); suggest near ``distance_threshold`` for connected grid nodes.
- **block_min_coverage**: float [0,1], default ``0.5``; minimum tissue coverage for a grid block; boundary blocks below this are discarded.
- **pairwise_include_intra_edges**: boolean, default ``true``; include same-class edges in multi-tissue graphs (affects global metrics like modularity/assortativity/betweenness; not interface metrics).
- **include_extended_metrics**: boolean, default ``true``; add global/local efficiency, small-world sigma, rich-club coefficient, and node-distribution summaries (betweenness max/std with ``*_norm``, degree skewness, local-efficiency min/std). See :doc:`../reference/features/graph/extended_metrics`.
- **extended_min_nodes**: integer, default ``10``, minimum ``3``; minimum node count in the analysis subgraph required to compute small-world sigma (smaller graphs return 0).
- **visualize**: boolean, default ``false``; render graph topology per subject (max cross-section habitat map, intra/cross-habitat networks on slices, 3D habitat map, 3D network) to ``out_dir/visualizations/graph``. Nodes are connected regions; edges are Delaunay spatial neighborhoods (same-class gray / cross-class purple).
- **visualization_format**: string, default ``both``; ``pdf`` (vector, preferred for publication) / ``png`` (raster) / ``both``.
- **visualization_dpi**: integer, default ``600``; PNG resolution (DPI).
- **visualization_show_background**: boolean, default ``true``; draw faint gray tissue outline under the network as spatial reference (similar to original Figure 4 style).
