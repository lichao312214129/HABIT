Graph topology features
=======================

Output
------

``habitat_graph_features.csv``

``graph`` is a **built-in** light habitat feature family (registered as
``habitat_feature_extractor`` name ``graph``). It is not a private HABIT-v2-only
plugin. Prefer the domain / public API path below; ``habit.compat.graph_plugin``
and related loaders are deprecated transitional shims.

Definition
----------

Each connected habitat region (optionally eroded and subdivided into grid
blocks) becomes a graph node. Edges connect region centroids within a distance
threshold (``edge_method: centroid_distance``) or region pairs that share enough
face / edge / corner-adjacent voxels (``edge_method: adjacency``). NetworkX-derived
topology metrics are reported:

* per habitat: ``single_h{label}_*`` columns
* per unordered habitat pair: ``pair_h{a}_h{b}_*`` columns

Absent labels still emit zero-valued empty-graph metrics so cohort tables keep
stable columns (same contract as :doc:`ith_score`). Size-dependent features also
carry VOI-normalized companions (``*_norm`` / ``*_per_habitat_volume`` /
``*_fraction``).

Numeric definitions live in L0 :mod:`habit.kernels.habitat_graph` and are
identical for the domain extractor, the kernel helper
:func:`~habit.kernels.extract_graph_features`, and the optional
:mod:`habit.viz` graph figures (same node / edge construction).

Graph construction options
--------------------------

These fields are shared by:

* YAML top-level ``graph:`` block (:class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`)
* Domain params (:class:`~habit.domain.GraphHabitatFeaturesParams`)
* Kernel options (:class:`~habit.kernels.HabitatGraphFeatureOptions`)

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Meaning (defaults match source)
   * - ``include_single_habitat_graph``
     - Within-habitat region graphs (default ``true``)
   * - ``include_pairwise_habitat_graph``
     - Pairwise inter-habitat graphs (default ``true``)
   * - ``edge_method``
     - ``centroid_distance`` (default) or ``adjacency``
   * - ``distance_threshold``
     - Centroid distance threshold in **pixel / voxel** units (default ``5.0``)
   * - ``adjacency_connectivity``
     - For ``adjacency``: ``face`` (6-conn), ``edge`` (18), ``corner`` (26); default ``face``
   * - ``adjacency_min_voxels``
     - Minimum adjacent voxel-pair count to create an adjacency edge (default ``1``)
   * - ``edge_weight``
     - ``none`` (default), ``distance``, ``inverse_distance``, or ``contact_voxels``
   * - ``min_region_voxels``
     - Drop connected regions smaller than this (default ``1``)
   * - ``connectivity``
     - Connected-component rule: ``face`` or ``full`` (default ``face``)
   * - ``erosion_radius``
     - Binary erosion iterations before labeling (default ``1``; ``0`` disables)
   * - ``subdivide_region_voxels``
     - Split components larger than this into grid blocks (default ``1000``; ``0`` disables)
   * - ``block_size``
     - Subdivision block edge length in voxels (default ``5``; keep near ``distance_threshold``)
   * - ``block_min_coverage``
     - Minimum fraction of a block that must be occupied (default ``0.5``)
   * - ``pairwise_include_intra_edges``
     - Add same-habitat proximity edges in pairwise graphs (default ``true``); interface metrics still use inter-class edges only
   * - ``include_extended_metrics``
     - Efficiency, small-world sigma, rich-club, node-distribution summaries (default ``true``)
   * - ``extended_min_nodes``
     - Minimum analysis-subgraph node count for small-world sigma (default ``10``; smaller graphs return ``0`` for that metric)

YAML-only visualization fields (recipe hook; **not** part of the extractor
``Spec`` fingerprint):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Meaning
   * - ``visualize``
     - When ``true``, write figures under ``<out_dir>/visualizations/graph/`` (default ``false``)
   * - ``visualization_format``
     - ``png``, ``pdf``, or ``both`` (default) for 2D figures; 3D renders are PNG only
   * - ``visualization_dpi``
     - Raster DPI (default ``600``)
   * - ``visualization_show_background``
     - Draw faint habitat partitions behind 2D networks (default ``true``)
   * - ``visualization_save_3d``
     - Also render 3D surface / network views when deps allow (default ``true``; needs ``[view]`` extras for PyVista / skimage)
   * - ``enabled`` / ``n_workers``
     - Legacy v0.1 keys accepted for compatibility; **no effect** (activation is ``feature_types``, figures run serially)

Normalization companions
------------------------

After base metrics are computed, size-dependent keys receive VOI-normalized
companions (tumor VOI = non-background voxel count ``V``):

* length-like suffixes (``*_avg_edge_distance``, ``*_std_edge_distance``,
  ``*_spatial_dispersion``) → ``*_norm`` by tumor bounding-box diagonal
* contact suffixes (``*_contact_voxels_*``) → ``*_norm`` by
  ``V**((ndim-1)/ndim)``
* count / voxel suffixes (``*_n_nodes*``, ``*_n_edges``, ``*_avg_node_voxels``,
  ``*_std_node_voxels``) → ``*_norm`` by ``V``, plus selected
  ``*_per_habitat_volume`` / ``*_fraction`` helpers

Output columns
--------------

Column prefixes follow habitat label ids present in the map's
``habitat_ids`` (or the explicit ``expected_labels`` argument of the kernel).

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Column pattern
     - Description
   * - ``single_h{k}_n_nodes``, ``_n_edges``, ``_edge_density``
     - Region-graph size and density for habitat ``k``
   * - ``single_h{k}_connected_components*``, ``_largest_component_ratio``
     - Component structure
   * - ``single_h{k}_avg_degree*``, ``_degree_cv``, ``_degree_entropy``
     - Degree statistics (some already hop-normalized)
   * - ``single_h{k}_avg_edge_distance*``, ``_spatial_dispersion*``
     - Geometry of edges / node spread
   * - ``single_h{k}_avg_clustering``, ``_avg_path_length*``, ``_diameter*``
     - Clustering and path metrics on the largest component
   * - ``single_h{k}_avg_betweenness``, ``_avg_closeness``, ``_modularity``
     - Centrality / community structure
   * - ``single_h{k}_nearest_neighbor_ratio``, node-voxel stats
     - Spatial packing / region size summaries
   * - ``single_h{k}_global_efficiency``, ``_local_efficiency``, ``_small_world_sigma``, rich-club / betweenness distribution (when extended)
     - Extended NetworkX metrics
   * - ``pair_h{a}_h{b}_*``
     - Analogous pairwise interface / bipartite-aware metrics (contact voxels, cross-degree ratios, assortativity, …)

Exact key sets depend on ``include_single_habitat_graph``,
``include_pairwise_habitat_graph``, and ``include_extended_metrics``.

Implementation
--------------

* Domain: ``habit/domain/habitat_features/graph.py``
  (``GraphHabitatFeatures`` / ``GraphHabitatFeaturesParams``)
* Kernels: ``habit/kernels/habitat_graph/``
* YAML block: ``GraphFeatureBlock`` in ``habit/schemas/workflows/habitat.py``
* Recipe + CSV name: ``habit/recipes/features.py``,
  ``habit/adapters/extract_io.py`` (stem ``habitat_graph_features``)
* Figures: ``habit/viz/habitat_graph.py`` (optional ``[viz]`` / ``[view]``)
* Deprecated shims: ``habit/compat/graph_plugin.py`` (prefer domain / API)

See also
--------

* How-to: :doc:`../../how_to/extract_features`
* Configuration: :doc:`../../configuration/feature_extraction`
* Example: :doc:`../../examples/graph_features`
