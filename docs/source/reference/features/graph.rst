Graph topology features
=======================

Output
------

``habitat_graph_features.csv``

``graph`` is a **built-in** light habitat feature family (registered as
``habitat_feature_extractor`` name ``graph``). Prefer the domain / public API
path; ``habit.compat.graph_plugin`` and related loaders are deprecated
transitional shims.

Numeric definitions live in L0 :mod:`habit.kernels.habitat_graph` and are
identical for the domain extractor, the kernel helper
:func:`~habit.kernels.extract_graph_features`, and the optional
:mod:`habit.viz` graph figures (same node / edge construction). This page
matches that source. It does not invent a second theory.

Notation and domains
--------------------

Let :math:`L:\Omega\to\{0,1,\ldots,K\}` be an integer habitat label map on a
2-D or 3-D voxel lattice :math:`\Omega\subset\mathbb{Z}^{d}` with
:math:`d\in\{2,3\}`.

* **Background.** Label :math:`0` is background and is **excluded** from node
  extraction, graph construction, and the tumour VOI measure
  :math:`V` below. It is not a graph node.
* **Habitat labels.** :math:`\mathcal{H}=\{k\in\mathbb{Z}:k>0,\;
  \exists\mathbf{x}\in\Omega,\; L(\mathbf{x})=k\}`. When the caller supplies
  ``expected_labels``, every listed id still emits columns even if it is
  absent from this map (empty graph → all metrics ``0``), so cohort tables
  keep a stable schema (same contract as :doc:`ith_score`).
* **Coordinates.** Centroids and distances are in **index / voxel units**,
  not millimetres. Spacing is not applied inside the kernel.
* **Undefined arithmetic.** Empty sequences, zero denominators, non-finite
  NetworkX outputs, and graphs below a metric's minimum size return
  ``0.0`` (never NaN in the exported table).

Tumour VOI (used only for size-normalized companions):

.. math::

   V = \#\{\mathbf{x}\in\Omega: L(\mathbf{x})\ne 0\},
   \qquad
   \delta_{\mathrm{bbox}}
   = \bigl\|(\mathbf{u}_{\max}-\mathbf{u}_{\min}+\mathbf{1})\bigr\|_{2}

where :math:`\mathbf{u}_{\min},\mathbf{u}_{\max}` are the inclusive index
bounds of non-background voxels. If :math:`V=0`, no ``*_norm`` companions
are written.

Graph construction
------------------

Nodes (connected regions)
~~~~~~~~~~~~~~~~~~~~~~~~~

For each habitat :math:`k\in\mathcal{H}`:

1. Build the binary mask :math:`M_k=\mathbf{1}[L=k]`.
2. If ``erosion_radius`` :math:`r\ge 1`, replace :math:`M_k` by
   :math:`r` iterations of binary erosion with structuring element
   :math:`S` and ``border_value=0``. Default in
   :class:`~habit.kernels.HabitatGraphFeatureOptions` is :math:`r=1`
   (one-voxel shell removed to suppress segmentation-edge noise).
   :math:`r=0` disables erosion.
3. Label connected components of the (possibly eroded) mask with
   ``scipy.ndimage.label`` and structure :math:`S`:

   * ``connectivity='face'``: 4-neighbourhood in 2-D / 6-neighbourhood in
     3-D (``generate_binary_structure(rank=d, connectivity=1)``);
   * ``connectivity='full'``: diagonal-inclusive
     (``connectivity=d``).
4. Drop a component :math:`C` if :math:`|C| <` ``min_region_voxels``
   (default ``1``).
5. If ``subdivide_region_voxels`` :math:`s>0` and :math:`|C|>s`, split
   :math:`C` into axis-aligned grid blocks of edge length ``block_size``
   :math:`b` (default :math:`s=1000`, :math:`b=5`). A block with voxel
   set :math:`B` is **kept** only when

   .. math::

      |B| / b^{d} \;>\; \texttt{block\_min\_coverage}

   (strict inequality; default coverage ``0.5``). Each kept block becomes
   its own node. If no block passes, or subdivision is off, the whole
   component is one node.

Each kept region (component or block) is a node

.. math::

   v=(k,\;\mathrm{id},\;\mathbf{c}_{v},\;n_{v},\;\mathrm{bbox}_{v})

with centroid :math:`\mathbf{c}_{v}` = mean of the region's voxel
**indices** (float), :math:`n_{v}=|B|` or :math:`|C|`, and a half-open
bounding box. Node ids are stable strings ``h{k}_c{id}``.

Edges: centroid distance
~~~~~~~~~~~~~~~~~~~~~~~~

``edge_method='centroid_distance'`` (not the default). Let :math:`\tau=`
``distance_threshold`` (default ``5.0`` voxel units). An undirected edge
exists between distinct nodes :math:`u,v` when

.. math::

   d(u,v)=\|\mathbf{c}_{u}-\mathbf{c}_{v}\|_{2} \le \tau

Implemented with a KD-tree (``scipy.spatial.cKDTree.query_pairs`` /
``query_ball_point``).

* **Single-habitat graph** for label :math:`k`: nodes of habitat :math:`k`
  only; every pair with :math:`d\le\tau` is an edge of type
  ``centroid_distance``.
* **Pairwise graph** for unordered labels :math:`(a,b)`, :math:`a<b`:
  **inter** edges between a node of :math:`a` and a node of :math:`b`
  with :math:`d\le\tau`. If ``pairwise_include_intra_edges`` is true
  (default), **intra** edges are also added within each label under the
  same :math:`\tau`. Interface metrics below use **inter edges only**;
  whole-graph metrics (modularity, assortativity, betweenness, components,
  extended efficiency) use the **full** graph (inter + optional intra).

Edge weight ``w`` from ``edge_weight``:

.. math::

   w =
   \begin{cases}
   1 & \text{``none'' (default)}\\
   d(u,v) & \text{``distance''}\\
   1/(d(u,v)+10^{-6}) & \text{``inverse_distance''}
   \end{cases}

``contact_voxels`` is unused for this method (stored as missing).

Edges: voxel adjacency (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``edge_method='adjacency'`` (default). Paint every node onto an integer id array
(background and non-requested labels stay ``0``). For each unique
neighbour offset in a **half-space** (first non-zero component :math:`=+1`,
so each unordered voxel pair is counted once):

* ``face``: one non-zero offset (4-conn / 6-conn);
* ``edge``: up to two non-zero offsets (8-conn / 18-conn);
* ``corner``: up to :math:`d` non-zero offsets (8-conn in 2-D, 26-conn in
  3-D). In 2-D, ``edge`` and ``corner`` coincide.

Let :math:`N_{\mathrm{adj}}(u,v)` be the number of neighbouring voxel
pairs whose ids are :math:`(u,v)`. An edge exists iff

.. math::

   N_{\mathrm{adj}}(u,v) \ge \texttt{adjacency\_min\_voxels}

(default ``10``). An edge exists when two regions are adjacent and the
contact (shared-boundary) voxel count is >= 10. ``contact_voxels``
:math:`=N_{\mathrm{adj}}`. Distance on
the edge is still the centroid Euclidean distance (used by distance
summaries). Weight is ``1`` unless ``edge_weight='contact_voxels'``, in
which case :math:`w=N_{\mathrm{adj}}`. Pairwise intra / inter filtering
matches the centroid-distance case.

Single-habitat metrics
----------------------

Prefix ``single_h{k}_``. Let :math:`G=(V_G,E_G)` be the undirected
NetworkX graph of habitat :math:`k` (:math:`n=|V_G|`, :math:`m=|E_G|`).
Degree of node :math:`v` is :math:`\deg(v)` (unweighted). Let
:math:`G_{\mathrm{LCC}}` be a copy of the largest connected component
(by node count).

.. math::

   \texttt{n\_nodes} &= n \\
   \texttt{n\_edges} &= m \\
   \texttt{edge\_density} &=
      \begin{cases}
      m\big/\binom{n}{2} & n\ge 2\\
      0 & \text{otherwise}
      \end{cases} \\
   \texttt{connected\_components} &=
      c(G) \quad\text{(0 if }n=0\text{)} \\
   \texttt{connected\_components\_ratio} &= c(G)/n \quad(0\text{ if }n=0) \\
   \texttt{largest\_component\_ratio} &= |V(G_{\mathrm{LCC}})|/n
      \quad(0\text{ if }n=0)

Degree statistics (``avg`` / ``max`` / ``min`` over :math:`\{\deg(v)\}`;
``0`` if :math:`n=0`). Hop-normalized companions divide by
:math:`n-1` (``0`` if :math:`n\le 1`):

.. math::

   \texttt{avg\_degree\_norm}
   = \overline{\deg}/(n-1),\quad
   \texttt{degree\_cv}
   = \mathrm{sd}(\deg)/\overline{\deg}

(``degree_cv`` is ``0`` if the mean is :math:`<10^{-12}`). Shannon
entropy of the **empirical degree histogram**, in bits:

.. math::

   \texttt{degree\_entropy}
   = -\sum_{t} p_{t}\log_{2}(p_{t}+10^{-12}),
   \qquad
   p_{t} = \#\{v:\deg(v)=t\}/n

Edge-length summaries use the list of centroid distances on **all** edges
of :math:`G` (population mean / standard deviation; ``0`` if no edges).

Node-size summaries use :math:`\{n_{v}:v\in V_G\}` (mean, sd, CV).

**Spatial dispersion.** Mean of the per-axis standard deviations of
centroids; ``0`` if fewer than two nodes:

.. math::

   \texttt{spatial\_dispersion}
   = \frac{1}{d}\sum_{\alpha=1}^{d}
     \mathrm{sd}\bigl(\{c_{v,\alpha}\}_{v\in V_G}\bigr)

**Nearest-neighbour ratio** (Clark–Evans :math:`R`). Let :math:`d_{\mathrm{NN}}`
be the mean Euclidean distance from each centroid to its nearest **other**
centroid (KD-tree, :math:`k=2`). The study-region measure is the sum of
node voxel counts :math:`A=\sum_{v}n_{v}` (not the centroid bounding box;
absolute values are therefore **not** comparable to PathPrism's 2-D
bounding-box form). Density :math:`\rho=n/A`. Expected CSR nearest-neighbour
distance:

.. math::

   d_{\mathrm{CSR}}
   =
   \begin{cases}
   1\big/(2\sqrt{\rho}) & d=2\\
   0.5539602785\big/\rho^{1/3} & d=3
   \end{cases}

The constant :math:`0.5539602785` is :math:`\Gamma(4/3)/(4\pi/3)^{1/3}`.
Then :math:`R=d_{\mathrm{NN}}/d_{\mathrm{CSR}}` (:math:`R<1` clustering,
:math:`R=1` CSR, :math:`R>1` regularity). Returns ``0`` if :math:`n<2` or
:math:`A\le 0`.

**Clustering coefficient.** ``nx.average_clustering(G)`` (unweighted) when
:math:`n>0`, else ``0``.

**Path metrics** are computed on :math:`G_{\mathrm{LCC}}` only, and only
when that component has at least two nodes. They are **hop counts**, not
physical millimetres:

.. math::

   \texttt{avg\_path\_length}
   &= \text{mean shortest-path length on }G_{\mathrm{LCC}} \\
   \texttt{diameter}
   &= \mathrm{diam}(G_{\mathrm{LCC}}) \\
   \texttt{avg\_path\_length\_norm}
   &= \texttt{avg\_path\_length}/(|V_{\mathrm{LCC}}|-1) \\
   \texttt{diameter\_norm}
   &= \texttt{diameter}/(|V_{\mathrm{LCC}}|-1)

**Centrality** (same LCC, :math:`|V_{\mathrm{LCC}}|>1`): mean of
``betweenness_centrality`` and ``closeness_centrality`` (NetworkX defaults:
betweenness is normalized). **Degree assortativity**:
``degree_assortativity_coefficient(G)``; non-finite or exception → ``0``.

**Modularity.** If :math:`m=0`, ``0``. Otherwise Louvain communities
(``louvain_communities(G, weight='weight', seed=0)``) then
``modularity(G, communities, weight='weight')``. Typical range roughly
:math:`[-0.5,1]`.

Pairwise metrics
----------------

Prefix ``pair_h{a}_h{b}_`` with :math:`a<b`. Let :math:`V_a,V_b` be the
node sets of the two labels, :math:`n_1=|V_a|`, :math:`n_2=|V_b|`, and
:math:`E_{\mathrm{inter}}` the inter-class edges only
(``edge_type != 'intra'``).

.. math::

   \texttt{n\_nodes\_1}=n_1,\quad
   \texttt{n\_nodes\_2}=n_2,\quad
   \texttt{n\_edges}=|E_{\mathrm{inter}}|

.. math::

   \texttt{edge\_density}
   =
   \begin{cases}
   |E_{\mathrm{inter}}|/(n_1 n_2) & n_1 n_2>0\\
   0 & \text{otherwise}
   \end{cases}

Distance summaries (mean / sd) use centroid distances on
:math:`E_{\mathrm{inter}}` only.

**Contact voxels** (adjacency method; ``0`` under centroid-distance):

.. math::

   \texttt{contact\_voxels\_sum}
   &= \sum_{e\in E_{\mathrm{inter}}} N_{\mathrm{adj}}(e) \\
   \texttt{contact\_voxels\_mean}
   &= \text{mean of those counts} \\
   \texttt{contact\_voxels\_max}
   &= \text{max, or 0 if none}

**Cross degree** of a node in :math:`V_a` is the number of neighbours in
:math:`V_b` (other-class only). Isolated ratio:

.. math::

   \texttt{isolated\_ratio\_1}
   = \#\{v\in V_a:\deg_{\mathrm{cross}}(v)=0\}/n_1

(and symmetrically ``_2``). **R21 / R12** (mean other-class neighbours):

.. math::

   \texttt{avg\_h\{b\}\_per\_h\{a\}}
   = \overline{\deg_{\mathrm{cross}}}(V_a),
   \qquad
   \texttt{avg\_h\{b\}\_per\_h\{a\}\_norm}
   = \overline{\deg_{\mathrm{cross}}}(V_a)/n_2

**AD / CV / EN** use **total** degree (intra + inter) so the triple shares
one basis: ``avg_degree_1``, ``degree_cv_1``, ``degree_entropy_1`` (and
``_2``). Hop-normalized AD divides by :math:`n_1+n_2-1`.

**Connected components** and modularity use the **full** pairwise graph.
``connected_components_norm`` divides component count by
:math:`n_1+n_2`. Mean betweenness is reported per class. Habitat-label
assortativity is ``attribute_assortativity_coefficient(G, 'habitat_label')``
when the graph has at least two nodes and one edge; else ``0``.

Extended metrics
----------------

Enabled by ``include_extended_metrics`` (default ``true``). Integration
metrics run on the **analysis subgraph**: the input graph if it is
connected and has edges; otherwise the largest connected component (empty
/ edgeless graphs stay as they are).

Let :math:`G^{\star}` be that analysis subgraph.

* ``global_efficiency`` / ``local_efficiency``:
  ``nx.global_efficiency`` / ``nx.local_efficiency`` on :math:`G^{\star}`
  when it has at least two nodes and one edge; else ``0``.
* ``small_world_sigma``: ``networkx.algorithms.smallworld.sigma`` with
  ``niter=5``, ``nrand=3``, ``seed=0``, **only if**
  :math:`|V(G^{\star})|\ge` ``extended_min_nodes`` (default ``10``), the
  graph is connected, and it has at least one edge; else ``0``. This is a
  Monte-Carlo estimate relative to random graphs; treat it as a
  **descriptor**, not a hypothesis test.
* ``rich_club_coefficient``: mean of the **finite** values of
  ``nx.rich_club_coefficient(G^*, normalized=True)``; ``0`` on failure or
  no edges.
* Betweenness distribution: ``betweenness_max`` / ``betweenness_std`` of
  ``betweenness_centrality(G^*)``. Normalized companions divide by the
  theoretical maximum betweenness of an undirected graph,
  :math:`(n-1)(n-2)/2` (``0`` if :math:`n<3`).
* ``degree_skewness``: unbiased sample skewness (``scipy.stats.skew(...,
  bias=False)``) of degrees on :math:`G^{\star}`; ``0`` if fewer than
  three nodes. Pairwise graphs report ``degree_skewness_1`` /
  ``degree_skewness_2`` per class instead of a single skewness.
* Per-node local efficiency: for each node, global efficiency of the
  subgraph induced by its neighbours (``0`` if fewer than two neighbours
  or no edges among them). Summaries: ``node_local_efficiency_min`` /
  ``node_local_efficiency_std``.

Pairwise extended metrics reuse the whole-graph efficiency / small-world /
rich-club block, then add per-class ``betweenness_max_{1,2}`` (and
``_norm`` using the **full-graph** :math:`n` in the betweenness scale).

VOI-normalized companions
-------------------------

After base metrics, size-dependent keys receive extra columns. Original
keys are **kept**. Let :math:`V` and :math:`\delta_{\mathrm{bbox}}` be as
above, and :math:`V_k=\#\{\mathbf{x}:L(\mathbf{x})=k\}`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Matching suffix
     - Companion
   * - ``*_avg_edge_distance``, ``*_std_edge_distance``, ``*_spatial_dispersion``
     - ``*_norm`` :math:`= x/\delta_{\mathrm{bbox}}` (length / bbox diagonal)
   * - ``*_contact_voxels_sum``, ``*_mean``, ``*_max``
     - ``*_norm`` :math:`= x\big/V^{(d-1)/d}` (interface ~ area scaling)
   * - ``*_avg_node_voxels``, ``*_std_node_voxels``
     - ``*_norm`` :math:`= x/V`
   * - ``*_n_nodes``, ``*_n_nodes_1``, ``*_n_nodes_2``, ``*_n_edges``, ``graph_num_nodes_total``
     - ``*_norm`` :math:`= x/V` (counts as densities)
   * - ``single_h{k}_n_nodes``
     - ``single_h{k}_n_nodes_per_habitat_volume`` :math:`= n/V_k`
   * - ``single_h{k}_avg_node_voxels``, ``*_std_node_voxels``
     - ``*_fraction`` :math:`= x/V_k`
   * - ``pair_*_n_nodes_1`` / ``_n_nodes_2``
     - ``*_per_habitat_volume`` using :math:`V_a` or :math:`V_b`

Hop-normalized path / degree companions (``*_avg_path_length_norm``,
``*_diameter_norm``, ``*_avg_degree_norm``, …) are **already** scaled by
graph size and are **not** divided by :math:`V`.

Subject-level extras (always written): ``graph_num_habitats`` = number of
habitats **present** in this map; ``graph_num_nodes_total`` = total nodes
across present habitats (independent of ``expected_labels``).

Construction options
--------------------

These fields are shared by the YAML ``graph:`` block
(:class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`), domain
params (:class:`~habit.domain.GraphHabitatFeaturesParams`), and kernel
options (:class:`~habit.kernels.HabitatGraphFeatureOptions`). Defaults
below match the kernel dataclass.

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
     - ``adjacency`` (default) or ``centroid_distance``
   * - ``distance_threshold``
     - Centroid distance threshold in **voxel** units (default ``5.0``)
   * - ``adjacency_connectivity``
     - For ``adjacency``: ``face`` (6-conn), ``edge`` (18), ``corner`` (26); default ``face``
   * - ``adjacency_min_voxels``
     - Minimum adjacent voxel-pair count to create an adjacency edge (default ``10``). An edge exists when two regions are adjacent and the contact voxel count is >= 10.
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
     - Minimum **strict** fraction of a block that must be occupied (default ``0.5``)
   * - ``pairwise_include_intra_edges``
     - Add same-habitat proximity edges in pairwise graphs (default ``true``); interface metrics still use inter-class edges only
   * - ``include_extended_metrics``
     - Efficiency, small-world sigma, rich-club, node-distribution summaries (default ``true``)
   * - ``extended_min_nodes``
     - Minimum analysis-subgraph node count for small-world sigma (default ``10``; smaller graphs return ``0``)

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
     - Also render 3D surface / network views when deps allow (default ``true``; needs ``[view]`` extras)
   * - ``enabled`` / ``n_workers``
     - Legacy v0.1 keys accepted for compatibility; **no effect** (activation is ``feature_types``, figures run serially)

What this family does not claim
-------------------------------

* Distances are **voxel hops in index space**, not physical millimetres.
  Do not interpret ``avg_edge_distance`` as a millimetre length unless the
  map is isotropic with 1 mm spacing.
* Small-world :math:`\sigma` is a NetworkX Monte-Carlo ratio on a possibly
  tiny habitat graph. It is not Watts–Strogatz inference for a brain
  connectome, and it is forced to ``0`` below ``extended_min_nodes``.
* Empty or missing habitats emit **zeros**, not missing values. A cohort
  mean of a ``single_h3_*`` column therefore mixes true structure with
  absent-label zeros unless you filter on ``graph_num_habitats`` / presence.
* Graph features describe the **partition geometry**. They do not identify
  a biological cell type or prove that a habitat is a clonal region.

Implementation
--------------

* Domain: ``habit/domain/habitat_features/graph.py``
  (``GraphHabitatFeatures`` / ``GraphHabitatFeaturesParams``)
* Kernels: ``habit/kernels/habitat_graph/``
  (``nodes.py``, ``edges.py``, ``metrics.py``, ``extended_metrics.py``,
  ``features.py``)
* YAML block: ``GraphFeatureBlock`` in ``habit/schemas/workflows/habitat.py``
* Recipe + CSV name: ``habit/recipes/features.py``,
  ``habit/adapters/extract_io.py`` (stem ``habitat_graph_features``)
* Figures: ``habit/viz/habitat_graph.py`` (optional ``[viz]`` / ``[view]``)
* Deprecated shims: ``habit/compat/graph_plugin.py`` (prefer domain / API)

See also
--------

* How-to: :doc:`../../how_to/graph_features`
* Configuration: :doc:`../../configuration/feature_extraction`
* Example: :doc:`../../examples/graph_features`
