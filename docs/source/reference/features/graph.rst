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
matches that source. It does not invent a second theory. The metric
family is adapted from PathPrism (Liang et al., *Cancer Cell* 2026),
a histopathology tissue-graph method; HABIT applies it to 2-D / 3-D
habitat maps with the voxel-index and VOI refinements documented below.

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
  not millimetres. Spacing is not applied inside the kernel. The size
  companions below are physically comparable across subjects only after all
  label maps have been resampled to the same isotropic spacing. They do not
  correct a change of acquisition resolution.
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
   :class:`~habit.kernels.HabitatGraphFeatureOptions` is :math:`r=0`
   (off): adjacency and contact are measured on the habitat labels as
   drawn. Pass a positive :math:`r` to shrink each habitat before
   labeling and edges.
3. Label connected components of the (possibly eroded) mask with
   ``scipy.ndimage.label`` and structure :math:`S`:

   * ``connectivity='full'`` (default): diagonal-inclusive 8-neighbourhood
     in 2-D / 26-neighbourhood in 3-D (``generate_binary_structure(rank=d,
     connectivity=d)``);
   * ``connectivity='face'``: 4-neighbourhood in 2-D / 6-neighbourhood in
     3-D (``generate_binary_structure(rank=d, connectivity=1)``). Face
     remains available when 4/6-connectivity is required.
4. Drop a component :math:`C` if :math:`|C| <` ``min_region_voxels``
   (default ``1``).
5. **Default** ``node_method='uniform_grid'`` skips per-component
   splitting. Instead the whole tumour VOI is painted with a **global**
   axis-aligned lattice of cubes of edge ``block_size`` :math:`b`
   (default :math:`b=8` **voxels**, not millimetres) whose origin is the
   VOI bounding-box minimum.
   A cube is kept when its occupied fraction exceeds
   ``block_min_coverage`` (default ``0.2``; cell-level filter).
   Inside each kept cube, every connected component of every habitat
   becomes its own node at that subregion's voxel-index centroid, so one
   cube may contribute several nodes. In-cell fragments smaller than
   ``min_region_voxels`` are dropped. Pass ``node_method='component'``
   for the older rule: if ``subdivide_region_voxels`` :math:`s>0` and
   :math:`|C|>s`, split :math:`C` into axis-aligned grid blocks of edge
   :math:`b` (default :math:`s=1000`). A block with voxel set :math:`B`
   is **kept** only when

   .. math::

      |B| / b^{d} \;>\; \mathrm{block\_min\_coverage}

   (strict inequality; default coverage ``0.2``). Each kept block becomes
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

``edge_method='centroid_distance'`` (opt-in). Let :math:`\tau=`
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
   1 & \text{if none (default)} \\
   d(u,v) & \text{if distance} \\
   1/(d(u,v)+10^{-6}) & \text{if inverse-distance}
   \end{cases}

``contact_voxels`` is unused for this method (stored as missing).

Edges: minimum voxel distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``edge_method='min_distance'`` (**default**). Reuses
``distance_threshold`` :math:`\tau` (default ``5.0`` voxel-index units).
With the default 8-voxel cubes, face-adjacent cubes connect (closest
voxels are one hop apart). One empty lattice cell between cubes is
closest-voxel distance about 8, which is greater than :math:`\tau=5`, so those
stay disconnected.
An undirected edge exists between distinct nodes :math:`u,v` when the
closest-voxel (set-separation) distance between their voxels satisfies

.. math::

   d_{\min}(u,v)=\min_{p\in B_{u},\,q\in B_{v}}\|p-q\|_{2} \le \tau

This is the minimum pairwise distance between the two voxel sets, **not**
the Hausdorff distance (which is a max-of-mins) and **not** centroid
distance: two large regions can have nearby boundaries while their
centroids are far apart. Implemented with a KD-tree over voxel indices
(``scipy.spatial.cKDTree``). Single-habitat and pairwise (inter /
optional intra) graphs follow the same structure as
``centroid_distance``. The stored edge ``distance`` is this
:math:`d_{\min}` (so ``avg_edge_distance`` / ``std_edge_distance``
summarize closest-voxel length, not centroid length). Edge weight uses
:math:`d_{\min}` when ``edge_weight`` is ``distance`` or
``inverse_distance``. ``contact_voxels`` is unused (stored as missing).

Edges: voxel adjacency (opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``edge_method='adjacency'`` (opt-in). Paint every node onto an integer id array
(background and non-requested labels stay ``0``). For each unique
neighbour offset in a **half-space** (first non-zero component :math:`=+1`,
so each unordered voxel pair is counted once):

* ``corner`` (default): up to :math:`d` non-zero offsets (8-conn in 2-D,
  26-conn in 3-D). In 2-D, ``edge`` and ``corner`` coincide;
* ``edge``: up to two non-zero offsets (8-conn / 18-conn);
* ``face``: one non-zero offset (4-conn / 6-conn). Face remains available
  when 4/6-connectivity is required.

Let :math:`N_{\mathrm{adj}}(u,v)` be the number of neighbouring voxel
pairs whose ids are :math:`(u,v)`. An edge exists iff

.. math::

   N_{\mathrm{adj}}(u,v) \ge \mathrm{adjacency\_min\_voxels}

(default ``10``). An edge exists when two regions are adjacent and the
contact (shared-boundary) voxel count is >= 10, measured on the
(possibly eroded) masks. Default ``erosion_radius=0`` means the habitat
labels as drawn. ``contact_voxels``
:math:`=N_{\mathrm{adj}}`. Distance on
the edge is still the centroid Euclidean distance (used by distance
summaries). Weight is ``1`` unless ``edge_weight='contact_voxels'``, in
which case :math:`w=N_{\mathrm{adj}}`. Pairwise intra / inter filtering
matches the centroid-distance case.

Single-habitat metrics
----------------------

Prefix ``single_h{k}_``. Let :math:`G=(V_G,E_G)` be the undirected
NetworkX graph of habitat :math:`k`, with :math:`n=|V_G|` and
:math:`m=|E_G|`. Degree :math:`\deg(v)` is unweighted. Let
:math:`G_{\mathrm{LCC}}` be the largest connected component (by node
count), :math:`n_{\mathrm{LCC}}=|V(G_{\mathrm{LCC}})|`. Population
standard deviation :math:`\mathrm{sd}` is ``numpy.std`` (``ddof=0``).
Empty / undefined cases return :math:`0`.

Size and components:

.. math::

   \begin{aligned}
   \mathrm{n\_nodes} &= n \\
   \mathrm{n\_edges} &= m \\
   \mathrm{edge\_density} &=
      \begin{cases}
      m\big/\binom{n}{2} & n\ge 2 \\
      0 & n<2
      \end{cases} \\
   \mathrm{connected\_components} &=
      \begin{cases}
      c(G) & n>0 \\
      0 & n=0
      \end{cases} \\
   \mathrm{connected\_components\_ratio} &=
      \begin{cases}
      c(G)/n & n>0 \\
      0 & n=0
      \end{cases} \\
   \mathrm{largest\_component\_ratio} &=
      \begin{cases}
      n_{\mathrm{LCC}}/n & n>0 \\
      0 & n=0
      \end{cases}
   \end{aligned}

Degree statistics. Let :math:`\{\deg(v)\}_{v\in V_G}` be empty when
:math:`n=0`. Hop-normalized companions divide by :math:`n-1`
(:math:`0` if :math:`n\le 1`). ``degree_cv`` is :math:`0` if
:math:`\overline{\deg}<10^{-12}`:

.. math::

   \begin{aligned}
   \mathrm{avg\_degree} &= \overline{\deg} \\
   \mathrm{max\_degree} &= \max_v \deg(v) \\
   \mathrm{min\_degree} &= \min_v \deg(v) \\
   \mathrm{avg\_degree\_norm} &= \overline{\deg}/(n-1) \\
   \mathrm{max\_degree\_norm} &= \max_v \deg(v)/(n-1) \\
   \mathrm{min\_degree\_norm} &= \min_v \deg(v)/(n-1) \\
   \mathrm{degree\_cv} &= \mathrm{sd}(\deg)/\overline{\deg} \\
   \mathrm{degree\_entropy}
   &= -\sum_{t} p_{t}\log_{2}(p_{t}+10^{-12}),
   \qquad
   p_{t}=\#\{v:\deg(v)=t\}/n
   \end{aligned}

Edge length uses the stored edge ``distance`` :math:`d_e` on **all**
edges of :math:`G` (:math:`0` if :math:`m=0`). Default
``min_distance`` stores closest-voxel :math:`d_{\min}`;
``centroid_distance`` and ``adjacency`` store centroid Euclidean
distance. Node size uses voxel counts :math:`\{n_v\}`:

.. math::

   \begin{aligned}
   \mathrm{avg\_edge\_distance} &= \overline{\{d_e:e\in E_G\}} \\
   \mathrm{std\_edge\_distance} &= \mathrm{sd}(\{d_e:e\in E_G\}) \\
   \mathrm{avg\_node\_voxels} &= \overline{\{n_v:v\in V_G\}} \\
   \mathrm{std\_node\_voxels} &= \mathrm{sd}(\{n_v:v\in V_G\}) \\
   \mathrm{node\_voxels\_cv} &= \mathrm{sd}(\{n_v\})/\overline{\{n_v\}}
   \end{aligned}

Spatial dispersion is the mean per-axis standard deviation of
centroids (:math:`0` if :math:`n<2`):

.. math::

   \mathrm{spatial\_dispersion}
   = \frac{1}{d}\sum_{\alpha=1}^{d}
     \mathrm{sd}\bigl(\{c_{v,\alpha}\}_{v\in V_G}\bigr)

Nearest-neighbour ratio (Clark–Evans :math:`R`). Let
:math:`d_{\mathrm{NN}}` be the mean Euclidean distance from each
centroid to its nearest **other** centroid (KD-tree, :math:`k=2`).
The study-region measure is :math:`A=\sum_{v}n_{v}` (occupied voxel
count, not the centroid bounding box; absolute values are **not**
comparable to PathPrism's 2-D bounding-box form). Density
:math:`\rho=n/A`. Returns :math:`0` if :math:`n<2` or :math:`A\le 0`:

.. math::

   \begin{aligned}
   d_{\mathrm{CSR}} &=
      \begin{cases}
      1\big/(2\sqrt{\rho}) & d=2 \\
      \Gamma(4/3)\big/(4\pi/3)^{1/3}\,\rho^{-1/3}
         = 0.5539602785\,\rho^{-1/3} & d=3
      \end{cases} \\
   \mathrm{nearest\_neighbor\_ratio}
   &= R = d_{\mathrm{NN}}/d_{\mathrm{CSR}}
   \end{aligned}

:math:`R<1` clustering, :math:`R=1` CSR, :math:`R>1` regularity.

Average clustering is the Watts–Strogatz [Watts1998]_ mean of local clustering
coefficients (:math:`0` if :math:`n=0`; a node with
:math:`\deg(v)<2` contributes :math:`0`):

.. math::

   \mathrm{avg\_clustering}
   = \frac{1}{n}\sum_{v\in V_G}
     \frac{2\,t_v}{\deg(v)\,(\deg(v)-1)}

where :math:`t_v` is the number of triangles through :math:`v`.

Path metrics use **hop counts** on :math:`G_{\mathrm{LCC}}` only, and
only when :math:`n_{\mathrm{LCC}}\ge 2` (else :math:`0`):

.. math::

   \begin{aligned}
   \mathrm{avg\_path\_length}
   &= \frac{1}{n_{\mathrm{LCC}}(n_{\mathrm{LCC}}-1)}
      \sum_{s\neq t} d_G(s,t)
      \quad\text{on }G_{\mathrm{LCC}} \\
   \mathrm{diameter}
   &= \max_{s,t} d_G(s,t)
      \quad\text{on }G_{\mathrm{LCC}} \\
   \mathrm{avg\_path\_length\_norm}
   &= \mathrm{avg\_path\_length}/(n_{\mathrm{LCC}}-1) \\
   \mathrm{diameter\_norm}
   &= \mathrm{diameter}/(n_{\mathrm{LCC}}-1)
   \end{aligned}

Centrality on the same LCC (:math:`n_{\mathrm{LCC}}>1`). Betweenness
uses NetworkX ``normalized=True`` (already divided by the undirected
maximum :math:`(n_{\mathrm{LCC}}-1)(n_{\mathrm{LCC}}-2)/2`).
Closeness uses the connected-graph form:

.. math::

   \begin{aligned}
   \mathrm{BC}(v)
   &= \frac{2}{(n_{\mathrm{LCC}}-1)(n_{\mathrm{LCC}}-2)}
      \sum_{s\neq v\neq t}
      \frac{\sigma_{st}(v)}{\sigma_{st}} \\
   \mathrm{CC}(v)
   &= (n_{\mathrm{LCC}}-1)\Big/\sum_{u\neq v} d_G(v,u) \\
   \mathrm{avg\_betweenness}
   &= \overline{\{\mathrm{BC}(v):v\in V(G_{\mathrm{LCC}})\}} \\
   \mathrm{avg\_closeness}
   &= \overline{\{\mathrm{CC}(v):v\in V(G_{\mathrm{LCC}})\}}
   \end{aligned}

Degree assortativity is Newman's degree–degree Pearson correlation
on :math:`G` (unweighted). Non-finite values and exceptions become
:math:`0`:

.. math::

   \mathrm{degree\_assortativity}
   = r_{\deg}(G)

Modularity uses Louvain communities
(``louvain_communities(G, weight='weight', seed=0)``) then
Newman–Girvan :math:`Q`. If :math:`m=0`, :math:`Q=0`. Typical range
roughly :math:`[-0.5,1]`:

.. math::

   \mathrm{modularity}
   = Q
   = \frac{1}{2W}\sum_{i,j}
     \left(A_{ij}-\frac{k_i k_j}{2W}\right)\delta(c_i,c_j)

where :math:`W` is the total edge-weight sum, :math:`k_i` is the
weighted degree, and :math:`c_i` is the Louvain community of node
:math:`i`. With default ``edge_weight='none'``, every :math:`A_{ij}`
is :math:`0` or :math:`1`.

Pairwise metrics
----------------

Prefix ``pair_h{a}_h{b}_`` with :math:`a<b`. Let :math:`V_a,V_b` be the
node sets of the two labels, :math:`n_1=|V_a|`, :math:`n_2=|V_b|`,
:math:`N=n_1+n_2`, and :math:`E_{\mathrm{inter}}` the inter-class
edges only (``edge_type != 'intra'``). Interface metrics below use
:math:`E_{\mathrm{inter}}` only. Whole-graph metrics (components,
modularity, assortativity, betweenness, extended efficiency) use the
**full** pairwise graph (inter + optional intra).

.. math::

   \begin{aligned}
   \mathrm{n\_nodes\_1} &= n_1 \\
   \mathrm{n\_nodes\_2} &= n_2 \\
   \mathrm{n\_edges} &= |E_{\mathrm{inter}}| \\
   \mathrm{edge\_density} &=
      \begin{cases}
      |E_{\mathrm{inter}}|/(n_1 n_2) & n_1 n_2>0 \\
      0 & n_1 n_2=0
      \end{cases}
   \end{aligned}

Distance summaries use the stored ``distance`` :math:`d_e` on
:math:`E_{\mathrm{inter}}` only (closest-voxel :math:`d_{\min}` for
default ``min_distance``; centroid Euclidean for
``centroid_distance`` / ``adjacency``):

.. math::

   \begin{aligned}
   \mathrm{avg\_edge\_distance}
   &= \overline{\{d_e:e\in E_{\mathrm{inter}}\}} \\
   \mathrm{std\_edge\_distance}
   &= \mathrm{sd}(\{d_e:e\in E_{\mathrm{inter}}\})
   \end{aligned}

Contact voxels (adjacency method; :math:`0` unless
``edge_method='adjacency'``). :math:`N_{\mathrm{adj}}(e)` is the
adjacent voxel-**pair** count on edge :math:`e`:

.. math::

   \begin{aligned}
   \mathrm{contact\_voxels\_sum}
   &= \sum_{e\in E_{\mathrm{inter}}} N_{\mathrm{adj}}(e) \\
   \mathrm{contact\_voxels\_mean}
   &= \overline{\{N_{\mathrm{adj}}(e)\}} \\
   \mathrm{contact\_voxels\_max}
   &= \max_e N_{\mathrm{adj}}(e)
      \quad\text{(0 if no inter edges)}
   \end{aligned}

Cross degree of :math:`v\in V_a` counts neighbours in :math:`V_b`
only. Isolated ratio and R21 / R12 (mean other-class neighbours):

.. math::

   \begin{aligned}
   \deg_{\mathrm{cross}}(v)
   &= \#\{u\sim v:\text{label}(u)\neq\text{label}(v)\} \\
   \mathrm{isolated\_ratio\_1}
   &= \#\{v\in V_a:\deg_{\mathrm{cross}}(v)=0\}/n_1 \\
   \mathrm{isolated\_ratio\_2}
   &= \#\{v\in V_b:\deg_{\mathrm{cross}}(v)=0\}/n_2 \\
   \mathrm{avg\_h\{b\}\_per\_h\{a\}}
   &= \overline{\deg_{\mathrm{cross}}}(V_a) \\
   \mathrm{avg\_h\{b\}\_per\_h\{a\}\_norm}
   &= \overline{\deg_{\mathrm{cross}}}(V_a)/n_2 \\
   \mathrm{avg\_h\{a\}\_per\_h\{b\}}
   &= \overline{\deg_{\mathrm{cross}}}(V_b) \\
   \mathrm{avg\_h\{a\}\_per\_h\{b\}\_norm}
   &= \overline{\deg_{\mathrm{cross}}}(V_b)/n_1
   \end{aligned}

AD / CV / EN use **total** degree (intra + inter) so the triple
shares one basis. Hop-normalized AD divides by :math:`N-1`:

.. math::

   \begin{aligned}
   \mathrm{avg\_degree\_1}
   &= \overline{\{\deg(v):v\in V_a\}} \\
   \mathrm{avg\_degree\_1\_norm}
   &= \mathrm{avg\_degree\_1}/(N-1) \\
   \mathrm{avg\_degree\_2}
   &= \overline{\{\deg(v):v\in V_b\}} \\
   \mathrm{avg\_degree\_2\_norm}
   &= \mathrm{avg\_degree\_2}/(N-1) \\
   \mathrm{degree\_cv\_1}
   &= \mathrm{sd}(\deg|_{V_a})/\overline{\deg|_{V_a}} \\
   \mathrm{degree\_cv\_2}
   &= \mathrm{sd}(\deg|_{V_b})/\overline{\deg|_{V_b}} \\
   \mathrm{degree\_entropy\_1}
   &= H(\{\deg(v):v\in V_a\}) \\
   \mathrm{degree\_entropy\_2}
   &= H(\{\deg(v):v\in V_b\})
   \end{aligned}

where :math:`H` is the same Shannon entropy as
``degree_entropy`` above.

Connected components and modularity use the **full** pairwise graph
:math:`G`:

.. math::

   \begin{aligned}
   \mathrm{connected\_components} &= c(G) \\
   \mathrm{connected\_components\_norm} &= c(G)/N \\
   \mathrm{modularity} &= Q(G)
      \quad\text{(same Louvain }Q\text{ as single-habitat)}
   \end{aligned}

Mean betweenness is NetworkX-normalized BC on the **full** graph,
averaged per class (:math:`0` if :math:`N\le 1` or :math:`m_{\mathrm{full}}=0`).
Habitat-label assortativity is Newman's attribute assortativity on
``habitat_label`` (same definedness gate):

.. math::

   \begin{aligned}
   \mathrm{betweenness\_mean\_1}
   &= \overline{\{\mathrm{BC}(v):v\in V_a\}} \\
   \mathrm{betweenness\_mean\_2}
   &= \overline{\{\mathrm{BC}(v):v\in V_b\}} \\
   \mathrm{habitat\_assortativity}
   &= r_{\mathrm{label}}(G)
   \end{aligned}

Extended metrics
----------------

Enabled by ``include_extended_metrics`` (default ``false``; pass
``true`` to opt in — these metrics dominate runtime on large maps). Let
:math:`G^{\star}` be the **analysis subgraph**: the input graph if it
is connected and has edges; otherwise its largest connected component
(empty / edgeless graphs stay as they are). Write
:math:`n^{\star}=|V(G^{\star})|`. Integration metrics below are
:math:`0` when :math:`n^{\star}<2` or :math:`G^{\star}` has no edges.

Global / local efficiency (Latora–Marchiori [Latora2001]_; hop distances, unweighted):

.. math::

   \begin{aligned}
   \mathrm{global\_efficiency}
   &= \frac{1}{n^{\star}(n^{\star}-1)}
      \sum_{i\neq j}\frac{1}{d_{ij}} \\
   \mathrm{local\_efficiency}
   &= \frac{1}{n^{\star}}\sum_{v}
      E_{\mathrm{glob}}\bigl(G^{\star}[N(v)]\bigr)
   \end{aligned}

where :math:`G^{\star}[N(v)]` is the subgraph induced by the
neighbours of :math:`v`, and a node with fewer than two neighbours
(or no edges among them) contributes :math:`0`.

Small-worldness and random-graph nulls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Watts and Strogatz [Watts1998]_ called a network small-world when it
has lattice-like clustering and random-graph path lengths. Humphries
and Gurney [Humphries2006]_ [Humphries2008]_ turned that into one
ratio :math:`\sigma=(C/C_{\mathrm{rand}})/(L/L_{\mathrm{rand}})`,
with :math:`\sigma>1` commonly read as small-world. HABIT uses that
ratio. :math:`C` is **transitivity** (global clustering, not the mean
of local coefficients) and :math:`L` is mean shortest-path length on
:math:`G^{\star}`, matching NetworkX ``sigma``. The exported value is
**0** unless :math:`n^{\star}\ge` ``extended_min_nodes`` (default
``10``), :math:`G^{\star}` is connected, and it has at least one
edge. This is a **descriptor**, not a hypothesis test.

HABIT writes **one** column ``small_world_sigma``. The three
``graph_null_sampler`` values are mutually exclusive **null models**
for :math:`C_{\mathrm{rand}}` and :math:`L_{\mathrm{rand}}`. They
are not three different observed-graph metrics. Do not treat them as
interchangeable, and do not invent a fourth sampler.

**analytic (default) — Erdős–Rényi (ER).**
Humphries' original null is an ER random graph [ErdosRenyi1959]_ with
the same node count :math:`n` and edge count :math:`m`. HABIT uses
the analytic approximations in [Humphries2008]_:
:math:`C_{\mathrm{rand}}\approx\langle k\rangle/n` and
:math:`L_{\mathrm{rand}}\approx\ln n/\ln\langle k\rangle`
(:math:`\langle k\rangle=2m/n`). Humphries drew **1000** ER graphs
only when testing borderline :math:`1\le S\le 3`; the default point
estimate does **not** sample graphs. This null does **not** preserve
the degree sequence.

**config — configuration model.**
Stub-matching random graphs that keep the observed degree sequence
[Newman2001]_ [Milo2004]_. :math:`C_{\mathrm{rand}}` and
:math:`L_{\mathrm{rand}}` are means over ``small_world_nrand``
accepted simple connected graphs (default ``100``). A realization is
rejected if pairing fails or the finished graph is disconnected. If
the fast sampler cannot fill the ensemble, HABIT falls back to
Maslov–Sneppen mixing of the observed graph so ``nrand`` is not
silently shrunk.

**rewire — Maslov–Sneppen.**
Start from a copy of the observed graph and apply double-edge swaps
(about ``small_world_niter`` swaps per edge, default ``100``)
[Maslov2002]_. Every node's degree stays exactly the same; who is
connected is mixed. This is what NetworkX ``smallworld.sigma`` /
``random_reference`` implements (NetworkX default ensemble is only
``nrand=10``; HABIT uses ``100``). Rubinov and Sporns
[Rubinov2010]_ review the same degree-preserving idea for brain
graphs.

Choose the sampler to match the citation, not to "improve" :math:`\sigma`:

* Report Humphries *S* as in the 2008 paper → ``analytic``.
* Match NetworkX ``sigma`` → ``rewire``.
* Textbook configuration-model ensemble → ``config``.

Never write "rewire" in a manuscript if the run used ``analytic`` or
``config``.

Habitat graphs are **spatial**. None of the three nulls constrain
voxel coordinates or contact geometry, so a lattice-like map can
inflate :math:`C` relative to any of them. That is a known limit of
classical small-world tests on imaging graphs
[Rubinov2010]_, not a reason to replace these nulls with an ad-hoc
generator.

Clustering, mean path length, and local/global efficiency use the
same unweighted hop distances as NetworkX. An ensemble, when
requested, is a stacked adjacency batch (Numba on CPU when
installed; otherwise NumPy; one PyTorch CUDA Floyd–Warshall launch
when ``graph_null_device='auto'`` and the work is large enough).

A paper that needs a *p*-value for :math:`\sigma>1` or
:math:`\phi_{\mathrm{norm}}>1` should call
:func:`habit.compare_graph_to_degree_preserving_null` (typically
**100–1000** graphs [VandenHeuvel2011]_).

Rich-club [Colizza2006]_ [McAuley2007]_: HABIT stores the mean of
the **finite** :math:`\phi_{\mathrm{norm}}(k)`. Under the analytic
default, :math:`\phi_{\mathrm{rand}}(k)` comes from **one**
configuration-model graph (NetworkX / Milo point estimate
[Milo2004]_). ``config`` / ``rewire`` reuse the sigma ensemble.

.. math::

   \begin{aligned}
   \mathrm{small\_world\_sigma}
   &= \frac{C/C_{\mathrm{rand}}}{L/L_{\mathrm{rand}}} \\
   \phi(k)
   &= \frac{2 E_{>k}}{n_{>k}(n_{>k}-1)} \\
   \phi_{\mathrm{norm}}(k)
   &= \phi(k)/\phi_{\mathrm{rand}}(k) \\
   \mathrm{rich\_club\_coefficient}
   &= \overline{\{\phi_{\mathrm{norm}}(k):\phi_{\mathrm{norm}}(k)\text{ finite}\}}
   \end{aligned}

Betweenness distribution on :math:`G^{\star}` uses the same
NetworkX-normalized :math:`\mathrm{BC}(v)\in[0,1]` as above. The
``*_norm`` companions **copy** those values when
:math:`n^{\star}\ge 3` (betweenness defined); they are **not**
divided by :math:`(n^{\star}-1)(n^{\star}-2)/2` a second time:

.. math::

   \begin{aligned}
   \mathrm{betweenness\_max}
   &= \max_v \mathrm{BC}(v) \\
   \mathrm{betweenness\_std}
   &= \mathrm{sd}(\{\mathrm{BC}(v)\}) \\
   \mathrm{betweenness\_max\_norm}
   &=
      \begin{cases}
      \mathrm{betweenness\_max} & n^{\star}\ge 3 \\
      0 & n^{\star}<3
      \end{cases} \\
   \mathrm{betweenness\_std\_norm}
   &=
      \begin{cases}
      \mathrm{betweenness\_std} & n^{\star}\ge 3 \\
      0 & n^{\star}<3
      \end{cases}
   \end{aligned}

Degree skewness is the unbiased sample skewness
(``scipy.stats.skew(..., bias=False)``) of degrees on
:math:`G^{\star}` (:math:`0` if :math:`n^{\star}<3`). Pairwise graphs
replace the single value by per-class
``degree_skewness_1`` / ``degree_skewness_2`` on the **full** graph:

.. math::

   \mathrm{degree\_skewness}
   = g_1\bigl(\{\deg(v):v\in V(G^{\star})\}\bigr)

Per-node local efficiency is :math:`E_{\mathrm{glob}}` of the
neighbour-induced subgraph (same :math:`0` rules as above):

.. math::

   \begin{aligned}
   E_{\mathrm{loc}}(v)
   &= E_{\mathrm{glob}}\bigl(G^{\star}[N(v)]\bigr) \\
   \mathrm{node\_local\_efficiency\_min}
   &= \min_v E_{\mathrm{loc}}(v) \\
   \mathrm{node\_local\_efficiency\_std}
   &= \mathrm{sd}(\{E_{\mathrm{loc}}(v)\})
   \end{aligned}

Pairwise extended metrics reuse the whole-graph efficiency /
small-world / rich-club / betweenness-distribution block on
:math:`G^{\star}`, then add per-class maxima of NetworkX-normalized
BC computed on the **full** pairwise graph (normalized by that
graph's :math:`N`). The ``*_norm`` companions copy those values when
:math:`N\ge 3`:

.. math::

   \begin{aligned}
   \mathrm{betweenness\_max\_1}
   &= \max_{v\in V_a}\mathrm{BC}(v) \\
   \mathrm{betweenness\_max\_2}
   &= \max_{v\in V_b}\mathrm{BC}(v) \\
   \mathrm{betweenness\_max\_1\_norm}
   &=
      \begin{cases}
      \mathrm{betweenness\_max\_1} & N\ge 3 \\
      0 & N<3
      \end{cases}
   \end{aligned}

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
   * - ``*_contact_voxels_sum``
     - ``*_norm`` :math:`= x\big/V^{(d-1)/d}` (whole-VOI interface density)
   * - ``*_contact_voxels_mean``, ``*_contact_voxels_max``
     - ``*_norm`` uses each contact edge's local area scale, not global
       :math:`V`. For edge :math:`e=(i,j)`, let
       :math:`s_e=\min(n_i,n_j)^{(d-1)/d}`; the companions are respectively
       :math:`\operatorname{mean}_e(c_e/s_e)` and
       :math:`\max_e(c_e/s_e)`. They can exceed 1 for thin nodes or
       diagonal adjacency.
   * - ``*_avg_node_voxels``, ``*_std_node_voxels``
     - ``*_norm`` :math:`= x/V`
   * - ``*_n_nodes``, ``*_n_nodes_1``, ``*_n_nodes_2``, ``*_n_edges``, ``graph_num_nodes_total``
     - ``*_norm`` :math:`= x/V` (counts as densities)
   * - ``single_h{k}_n_nodes``
     - ``single_h{k}_n_nodes_per_habitat_volume`` :math:`= n/V_k`
   * - ``single_h{k}_n_edges``, ``single_h{k}_connected_components``
     - ``*_per_habitat_volume`` :math:`= x/V_k`
   * - ``single_h{k}_avg_node_voxels``, ``*_std_node_voxels``
     - ``*_fraction`` :math:`= x/V_k`
   * - ``single_h{k}_avg_edge_distance``, ``*_std_edge_distance``,
       ``*_spatial_dispersion``
     - ``*_per_habitat_bbox_diagonal`` :math:`=x/\delta_k`, where
       :math:`\delta_k` is habitat :math:`k`'s bounding-box diagonal
   * - ``pair_h{a}_h{b}_avg_edge_distance``, ``*_std_edge_distance``
     - ``*_per_pair_bbox_diagonal`` :math:`=x/\delta_{ab}`, where
       :math:`\delta_{ab}` is the bounding-box diagonal of
       :math:`L\in\{a,b\}`
   * - ``pair_h{a}_h{b}_contact_voxels_sum``
     - ``*_per_pair_area_scale`` :math:`=
       x/(V_a+V_b)^{(d-1)/d}`; unlike ``*_norm``, labels outside this
       pair do not dilute the denominator
   * - ``pair_*_n_nodes_1`` / ``_n_nodes_2``
     - ``*_per_habitat_volume`` using :math:`V_a` or :math:`V_b`

Hop-normalized path / degree companions (``*_avg_path_length_norm``,
``*_diameter_norm``, ``*_avg_degree_norm``, …) are **already** scaled by
graph size and are **not** divided by :math:`V`.

``*_norm`` does not always mean “a nuisance-free feature”. In particular,
``avg_node_voxels_norm`` and ``*_fraction`` express a node's share of the
whole VOI or habitat; under the default fixed 8-voxel lattice they still
depend on ``block_size`` and ``block_min_coverage``. Similarly,
``*_avg_edge_distance_norm`` is a local edge length relative to the whole
ROI extent: with the default fixed 5-voxel distance threshold it may
decrease as an otherwise similar tumour becomes larger. Report the raw
construction parameters with either companion.

Null-model comparisons for topology
------------------------------------

Some graph quantities are dimensionless but still vary systematically with
node count, edge count, degree sequence, or the statistical behavior of a
maximum/minimum. Dividing them by VOI volume is not a valid correction.
For a hypothesis about organization **beyond the degree sequence**, compare
the observed value against degree-preserving random graphs instead.

This is appropriate for ``avg_clustering``, ``avg_path_length``,
``diameter``, ``global_efficiency``, ``local_efficiency``, ``modularity``,
``degree_assortativity``, raw betweenness maxima/standard deviations,
``node_local_efficiency_min/std``, and pairwise
``habitat_assortativity``. ``degree_entropy`` and degree summaries are
fixed by a degree-preserving null and therefore need sample-size-aware
estimation rather than this null model. Default
``small_world_sigma`` uses an analytic Erdős–Rényi null
[Humphries2008]_. ``rich_club_coefficient`` uses one
degree-preserving graph (or the opt-in ensemble). Report
``graph_null_sampler`` and do not treat these as volume-normalized
quantities.

:func:`habit.compare_graph_to_degree_preserving_null` is an explicit,
opt-in API. It preserves every node's degree, node count, and edge count,
but intentionally does **not** preserve connected components, spatial
coordinates, edge distances, contact area, or habitat labels. It must not
be used to normalize ``contact_voxels_*``, edge-distance, spatial-dispersion,
or node-volume features. Check its ``is_valid`` result: a zero Z score for
an invalid result is a sentinel, not evidence of no difference.

Subject-level extras (always written; independent of
``expected_labels``):

.. math::

   \begin{aligned}
   \mathrm{graph\_num\_habitats}
   &= |\mathcal{H}|
      \quad\text{(labels actually present in this map)} \\
   \mathrm{graph\_num\_nodes\_total}
   &= \sum_{k\in\mathcal{H}} n^{(k)}
   \end{aligned}

where :math:`n^{(k)}` is ``single_h{k}_n_nodes`` for habitats
present in the map.

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
     - ``min_distance`` (default), ``adjacency``, or ``centroid_distance``
   * - ``distance_threshold``
     - Distance threshold in **voxel-index** units (default ``5.0``). Used by ``centroid_distance`` (centroid-to-centroid) and ``min_distance`` (closest-voxel).
   * - ``adjacency_connectivity``
     - For ``adjacency``: default ``corner`` (8-conn in 2-D / 26-conn in 3-D); ``edge`` (8/18); ``face`` (4/6) remains available
   * - ``adjacency_min_voxels``
     - Minimum adjacent voxel-pair count to create an adjacency edge (default ``10``). An edge exists when two regions are adjacent and the contact voxel count is >= 10.
   * - ``edge_weight``
     - ``none`` (default), ``distance``, ``inverse_distance``, or ``contact_voxels``
   * - ``min_region_voxels``
     - Drop connected regions smaller than this (default ``1``)
   * - ``connectivity``
     - Connected-component rule: default ``full`` (8-conn in 2-D / 26-conn in 3-D); ``face`` (4/6) remains available
   * - ``erosion_radius``
     - Binary erosion iterations before labeling (default ``0`` / off; set ``>= 1`` to shrink habitats before edges)
   * - ``node_method``
     - ``uniform_grid`` (default: global VOI lattice; one node per in-cell subregion centroid) or ``component``
   * - ``subdivide_region_voxels``
     - In ``component`` mode, split components larger than this (default ``1000``; ``0`` disables)
   * - ``block_size``
     - Cube edge length in **voxels** (default ``8``, not millimetres). Face-adjacent 8-cubes connect; one empty lattice cell (closest-voxel distance about 8) stays disconnected at ``distance_threshold=5``.
   * - ``block_min_coverage``
     - Minimum **strict** occupied fraction of a cube to keep the cell (default ``0.2``). Applied per cell; tiny in-cell fragments use ``min_region_voxels``
   * - ``pairwise_include_intra_edges``
     - Add same-habitat proximity edges in pairwise graphs (default ``true``); interface metrics still use inter-class edges only
   * - ``include_extended_metrics``
     - Efficiency, one small-world :math:`\sigma`, rich-club, node-distribution summaries (default ``false``; set ``true`` to opt in)
   * - ``extended_min_nodes``
     - Minimum analysis-subgraph node count for either small-world sigma (default ``10``; smaller graphs return ``0``)
   * - ``small_world_nrand``
     - Ensemble size when ``graph_null_sampler`` is ``config`` or ``rewire`` (default ``100``). Ignored by analytic Humphries *S*.
   * - ``small_world_niter``
     - Rewires per edge when ``graph_null_sampler='rewire'`` (default ``100``). Ignored by analytic / ``config``.
   * - ``rich_club_q``
     - Mixing floor for ``graph_null_sampler='rewire'`` (default ``100``), not the number of null graphs.
   * - ``graph_null_sampler``
     - ``analytic`` (default, Humphries ER *S*), ``config``, or ``rewire``. One ``small_world_sigma`` column; the last two replace the analytic value.
   * - ``graph_null_device``
     - Batched C/L device: ``auto`` (default), ``cpu``, ``cuda``, or ``cuda:N``. ``auto`` uses CUDA only when Floyd–Warshall work is large enough.
   * - ``graph_metric_backend``
     - Hop / clustering / Louvain backend: ``networkx`` (default), ``igraph``, or ``auto``. ``igraph`` needs the optional ``[igraph]`` extra (GPL-2.0+; not in ``[all]``). ``auto`` uses igraph when that extra is installed.

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
   * - ``visualization_show_grid``
     - Draw the equal-volume cube lattice on 2D figures (default ``true``)
   * - ``visualization_block_size``
     - Display cube edge in voxels (default ``null`` → extraction ``block_size``, library default 8)
   * - ``visualization_grid_linestyle``
     - Lattice line style (default ``--`` dashed)
   * - ``visualization_save_3d``
     - Also render 3D surface / network views when deps allow (default ``true``; needs ``[view]`` extras)
   * - ``enabled`` / ``n_workers``
     - Legacy v0.1 keys accepted for compatibility; **no effect** (activation is ``feature_types``, figures run serially)

What this family does not claim
-------------------------------

* Distances are **voxel-index units**, not physical millimetres.
  ``avg_edge_distance`` follows the edge method: closest-voxel
  :math:`d_{\min}` for default ``min_distance``, centroid Euclidean
  for ``centroid_distance`` / ``adjacency``. Do not interpret it as a
  millimetre length unless the map is isotropic with 1 mm spacing.
* Small-world :math:`\sigma` is Humphries' ratio on a possibly tiny
  habitat graph. The default ``analytic`` sampler is the closed-form
  ER approximation, not a NetworkX Monte-Carlo draw. ``config`` /
  ``rewire`` are degree-preserving ensembles. None of these is
  Watts–Strogatz inference for a brain connectome, and the value is
  forced to ``0`` below ``extended_min_nodes``.
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
  (``nodes.py``, ``edges.py``, ``proximity.py``, ``metrics.py``,
  ``extended_metrics.py``, ``features.py``, ``traversal.py``).
  ``min_distance`` builds one global edge table. Voxel-neighbour
  sweep and dual-lattice range search run only when
  ``node_method='uniform_grid'`` (``grid_origin`` and
  ``grid_block_size`` are set). The Chebyshev radius is ``0`` when
  ``T < 1``, else ``floor((T-1)/B)+1`` for user ``block_size=B``
  and ``distance_threshold=T``. ``component`` nodes keep the
  all-pairs closest-voxel walk. One all-sources BFS per graph
  yields Brandes betweenness, closeness, mean path length, and
  diameter (NetworkX definitions). Default extract keeps edges as
  integer arrays and runs those hop / clustering / component
  columns on CSR (Compressed Sparse Row) adjacency -- no NetworkX
  object. Brandes sources run in parallel. Louvain modularity uses
  optional igraph when installed, otherwise a CSR Blondel sweep
  (the partition can differ from NetworkX ``seed=0``).
  Node default stays ``uniform_grid``.
* YAML block: ``GraphFeatureBlock`` in ``habit/schemas/workflows/habitat.py``
* Recipe + CSV name: ``habit/recipes/features.py``,
  ``habit/adapters/extract_io.py`` (stem ``habitat_graph_features``)
* Figures: ``habit/viz/habitat_graph.py`` (optional ``[viz]`` / ``[view]``)
* Deprecated shims: ``habit/compat/graph_plugin.py`` (prefer domain / API)

References
----------

Liang J, Jiang X, Reitsam NG, et al. Spatial biomarker discovery via
interpretable semantic learning in histopathology. *Cancer Cell* 2026
(`DOI <https://doi.org/10.1016/j.ccell.2026.05.014>`__).

.. [Watts1998] Watts DJ, Strogatz SH. Collective dynamics of
   'small-world' networks. *Nature* 1998;393:440–442.
   (`DOI <https://doi.org/10.1038/30918>`__)
.. [ErdosRenyi1959] Erdős P, Rényi A. On random graphs I.
   *Publ Math Debrecen* 1959;6:290–297.
.. [Newman2001] Newman MEJ, Strogatz SH, Watts DJ. Random graphs with
   arbitrary degree distributions and their applications.
   *Phys Rev E* 2001;64:026118.
   (`DOI <https://doi.org/10.1103/PhysRevE.64.026118>`__)
.. [Latora2001] Latora V, Marchiori M. Efficient behavior of
   small-world networks. *Phys Rev Lett* 2001;87:198701.
   (`DOI <https://doi.org/10.1103/PhysRevLett.87.198701>`__)
.. [Humphries2006] Humphries MD, Gurney K, Prescott TJ. The brainstem
   reticular formation is a small-world, not scale-free, network.
   *Proc R Soc B* 2006;273:503–511.
   (`DOI <https://doi.org/10.1098/rspb.2005.3354>`__)
.. [Humphries2008] Humphries MD, Gurney K. Network 'small-world-ness':
   a quantitative method for determining canonical network equivalence.
   *PLoS One* 2008;3:e2051.
   (`DOI <https://doi.org/10.1371/journal.pone.0002051>`__)
.. [Maslov2002] Maslov S, Sneppen K. Specificity and stability in
   topology of protein networks. *Science* 2002;296:910–913.
   (`DOI <https://doi.org/10.1126/science.1065103>`__)
.. [Rubinov2010] Rubinov M, Sporns O. Complex network measures of
   brain connectivity: uses and interpretations. *NeuroImage*
   2010;52:1059–1069.
   (`DOI <https://doi.org/10.1016/j.neuroimage.2009.10.003>`__)
.. [Colizza2006] Colizza V, Flammini A, Serrano MA, Vespignani A.
   Detecting rich-club ordering in complex networks. *Nat Phys*
   2006;2:110–115.
   (`DOI <https://doi.org/10.1038/nphys209>`__)
.. [McAuley2007] McAuley JJ, da Fontoura Costa L, Caetano TS. The
   rich-club phenomenon across complex network hierarchies.
   *Appl Phys Lett* 2007;91:084103.
   (`DOI <https://doi.org/10.1063/1.2773951>`__)
.. [Milo2004] Milo R, Kashtan N, Itzkovitz S, Newman MEJ, Alon U.
   Uniform generation of random graphs with arbitrary degree
   sequences. arXiv:cond-mat/0312028, 2004.
   (`arXiv <https://arxiv.org/abs/cond-mat/0312028>`__)
.. [VandenHeuvel2011] van den Heuvel MP, Sporns O. Rich-club
   organization of the human connectome. *J Neurosci*
   2011;31:15775–15786.
   (`DOI <https://doi.org/10.1523/JNEUROSCI.3539-11.2011>`__)

See also
--------

* How-to: :doc:`../../how_to/graph_features`
* Configuration: :doc:`../../configuration/feature_extraction`
* Example: :doc:`../../examples/graph_features`
