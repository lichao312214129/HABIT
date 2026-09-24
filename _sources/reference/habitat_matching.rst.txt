Matching habitat labels across fits and subjects
================================================

Clustering returns integer ids in arbitrary order. Habitat 1 of one fit
need not be habitat 1 of another. This page states when that matters,
which matching method HABIT offers for each situation, how the cohort
method works, what it can and cannot guarantee, and where each algorithm
comes from.

Code: :mod:`habit.precision` (:func:`~habit.precision.align_habitat_map`,
:func:`~habit.precision.align_habitat_maps_to_prototypes`) on top of the
L0 kernel :mod:`habit.kernels.habitat_label_match`. Runnable gallery page:
:doc:`/auto_examples/04_habitat_maps/plot_05_match_labels`. Worked, visual
pages for every case (label switching, overlap cases, the prototype loop
round by round, the four distances, frozen prototypes, effect on cohort
tables): :doc:`/auto_examples/07_habitat_matching/index`. Figures:
:func:`~habit.viz.plot_label_overlap_matrix`,
:func:`~habit.viz.plot_prototype_matching`.

When matching is needed
-----------------------

* **Two-step, direct pooling, apply a saved model.** One cohort model
  defines the habitats; every subject is labelled against the same
  centroids. Ids already mean the same thing everywhere. Do **not**
  rematch.
* **One-step (habitats defined inside each subject).** Every subject has
  its own model and its own id order. Before any cohort table
  ("volume fraction of habitat 2") the ids must be given shared names.
* **Same subject, two fits / two observers / a perturbed retest.** Ids
  must be paired before Dice or stability is computed
  (:func:`~habit.precision.habitat_stability`).

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Situation
     - Call
     - What is compared
   * - **Different subjects (one-step cohort, any habitat counts)**
     - ``align_habitat_maps_to_prototypes(maps, models=...)``
     - Every subject's habitats against shared prototypes (below).
   * - **Same voxels** (retest, perturbation, another preprocessing
       chain, two observers, two seeds)
     - ``align_habitat_map(reference, moving)`` /
       ``habitat_stability(reference, perturbed)``
     - Voxel overlap (Hungarian on the contingency table; Prior et al. 2024)

These are the only two methods, split by what the maps share. When they
label the same voxels, which voxels were grouped together is the most
direct evidence of correspondence, and it needs no feature to be
comparable. When they label different patients no voxel is shared, so
only habitat descriptions can be compared, and the prototype method
compares them without picking a reference subject. Matching two maps of
the same voxels by feature centroids is not offered: the two fits often
live in different feature spaces (for example raw intensities versus
winsorised min–max), where centroid distances have no meaning.

Pairwise feature matching ("which habitat of B is habitat 1 of A") is not
a separate method either: for two subjects it is the prototype method
(see *Two subjects* below). Across a cohort it has a flaw: the answer
depends on who A is.

Why a single reference is not enough
------------------------------------

Three subjects, two habitats each, two clustering features (relative
enhancement and washout, already comparable across subjects):

.. list-table::
   :header-rows: 1

   * - Subject
     - Habitat
     - Enhancement
     - Washout
   * - A
     - 1
     - 1
     - 0
   * - A
     - 2
     - 0
     - 0
   * - B
     - 1
     - 0
     - 1
   * - B
     - 2
     - 0
     - 0
   * - C
     - 1
     - 2
     - 1
   * - C
     - 2
     - 0
     - 2

Pairwise matching onto one reference subject (Hungarian, Euclidean):

* **Reference A.** B→A keeps ids; C→A keeps ids. Groups
  {A1, B1, C1} and {A2, B2, C2}.
* **Reference B.** A→B keeps ids; C→B **swaps** them. Groups
  {A1, B1, C2} and {A2, B2, C1}.

Same data, different reference, and subject C gets the opposite names.
Chaining pairwise matches is also not transitive: A→B and B→C together
imply a C→A pairing that disagrees with matching C to A directly.

A neutral score for any grouping is the within-group sum of squares
(each habitat's squared distance to the mean of its group):

.. list-table::
   :header-rows: 1

   * - Grouping
     - Group means (enhancement, washout)
     - Within-group sum of squares
   * - Reference A
     - (1, 0.667), (0, 0.667)
     - 5.333
   * - Reference B
     - (0.333, 1), (0.667, 0.333)
     - 6.000
   * - Prototype matching
     - (0, 1), (1, 0.333)
     - **4.667**

Prototype matching finds {A2, B1, C2} and {A1, B2, C1}: a tighter
grouping than either reference produces, and the same result whichever
subject starts the search. This example is the regression test
``TestAlignHabitatMapsToPrototypes`` in ``tests/domain/test_precision.py``.

Prototype matching (cohort)
---------------------------

What describes a habitat
^^^^^^^^^^^^^^^^^^^^^^^^

Each habitat is summarised by one feature vector. Pass exactly one
source:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Habitat summary
   * - ``models=`` (default use)
     - The fitted per-subject ``HabitatModel.centroids``: the clustering
       centroids, i.e. the features that defined the habitats. Row
       :math:`i` is habitat ``habitat_ids[i]``. ``feature_names`` must
       agree across subjects.
   * - ``features=``
     - One voxel feature source per map: a
       :class:`~habit.contracts.VoxelFeatureField` (rows placed by
       ``voxel_index``) or a volume / ``ImageVolume`` on the map grid
       with an optional trailing feature axis. Each habitat is summarised
       by the ``reduction`` (``"mean"`` default, or ``"median"``) of its
       voxels. With the clustering field itself this reproduces the
       fitted centroids (demo: largest difference :math:`1.2\times10^{-3}`,
       k-means convergence).
   * - ``centroids=``
     - Your own ``(n_habitats, n_features)`` matrices, rows in
       ``habitat_ids`` order.

Use features that do not feed the downstream model (volume, texture
columns): naming habitats by the quantity you later test would bias that
test.

Algorithm
^^^^^^^^^

Let subject :math:`s` have summaries :math:`x_{s,1},\dots,x_{s,n_s}` and
let :math:`\mu_1,\dots,\mu_K` be shared prototypes, with
:math:`K=\max_s n_s` (the largest habitat count). An assignment
:math:`\pi_s` maps each habitat of :math:`s` to a distinct prototype. The
objective is

.. math::

   J = \sum_{s}\Big[\sum_{i\,\text{matched}} \lVert x_{s,i}-\mu_{\pi_s(i)}\rVert^2
       + \lambda^2\cdot\#\{\text{unmatched habitats of } s\}\Big],

with :math:`\lambda` = ``max_distance``. This is the default metric
(``metric="sqeuclidean"``); other metrics replace the squared distance by
their own cost (next section). By default ``max_distance`` is ``None``:
the penalty term is absent and every habitat is matched. The algorithm
alternates:

1. **Assign.** For each subject, solve a one-to-one assignment of its
   habitats onto prototypes by the Hungarian algorithm (Kuhn 1955;
   Munkres 1957) on the metric cost. Two habitats of one tumour can
   never share a name (a cannot-link constraint; Wagstaff et al. 2001).
2. **Update.** Each prototype moves to the centre that minimises that
   cost: the mean of the summaries assigned to it for squared Euclidean
   (a prototype nobody picked keeps its previous value).

Each step can only lower :math:`J`, so the loop stops after finitely many
rounds. The search is started once from every subject that has exactly
:math:`K` habitats and the start with the lowest :math:`J` is kept, so no
single subject decides the result. Prototypes are then numbered in
lexicographic order of their feature values (first feature most
significant), so ids do not depend on the order of the input list.

This is the relabelling algorithm Stephens (2000) proposed for the
label-switching problem in mixture models (see also Jasra et al. 2005);
the same "cluster each sample, then cluster the clusters into shared
templates" design is standard in cross-sample flow cytometry (FLAME,
Pyne et al. 2009; FlowSOM meta-clustering, Van Gassen et al. 2015).

Two subjects: pairwise matching as a special case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For two subjects :math:`A` and :math:`B`, a matched pair
:math:`(a, b)` has prototype :math:`m = (a+b)/2` and contributes

.. math::

   \lVert a-m\rVert^2 + \lVert b-m\rVert^2 = \tfrac12\lVert a-b\rVert^2

to :math:`J`. Minimising :math:`J` is therefore minimising
:math:`\sum_i \lVert a_i-b_{\pi(i)}\rVert^2`: pairwise Hungarian matching
on squared Euclidean distance. The search also finds the global optimum:
starting from :math:`A`, the first assignment is exactly that optimal
pairing and later rounds cannot raise :math:`J`. With unequal counts
(2 versus 3) the extra habitat takes a prototype of its own, which is
what pairwise matching does with a leftover. Two differences remain:
pairwise matching on plain (not squared) Euclidean distance can pick a
different pairing in rare cases, and the prototype method numbers the
pairs by prototype order instead of copying the reference subject's ids.
Regression test: ``test_two_subjects_equal_pairwise_squared_hungarian``.

Distance metrics
^^^^^^^^^^^^^^^^

``metric=`` changes the cost and, with it, the prototype update, so the
alternating search still never increases its objective:

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - ``metric``
     - Cost between a habitat and a prototype
     - Prototype update
     - Known as
   * - ``"sqeuclidean"`` (default)
     - squared Euclidean distance
     - mean
     - k-means
   * - ``"manhattan"``
     - L1 distance (sum of absolute differences)
     - per-feature median
     - k-medians
   * - ``"cosine"``
     - :math:`1-\cos` of the angle between the vectors
     - mean of unit vectors, renormalised
     - spherical k-means (Dhillon and Modha 2001)
   * - ``"correlation"``
     - :math:`1-r` (Pearson across features)
     - rows centred on their own mean, then as cosine
     - —

``distance`` and ``max_distance`` are in the metric's units: Euclidean
distance, L1 distance, :math:`1-\cos`, or :math:`1-r`.

Which to use. Squared Euclidean and Manhattan compare feature **values**;
Manhattan's median is less pulled by one outlying habitat. Cosine and
correlation discard the level of the feature vector:

* Cosine compares only direction. Habitats with relative enhancement
  (0.5, 0.6) and (2.0, 2.4) point the same way, so cosine calls them the
  same habitat although one enhances four times as much. With a single
  feature of constant sign every habitat has the same direction and all
  costs are equal.
* Correlation compares only the shape of the profile across features.
  With two features every centred profile is one of two directions, so
  it can only tell "first feature higher" from "second feature higher".
  With one feature Pearson :math:`r` is undefined and HABIT raises an
  error; a habitat whose features are all equal raises the same error.

HABIT does not refuse cosine or correlation for two or three features,
but they are meaningful only when many features describe a habitat and
their pattern, not their level, is what defines it. For habitats built
from enhancement or intensity, keep the default. Features on different
scales should be ``standardize="zscore"`` first for every metric.

Naming new subjects with frozen prototypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A validation cohort, or one new patient, must be named with the
prototypes fitted on the training cohort. Refitting on the pooled
cohorts would let the new subjects move the prototypes and change the
definition that the model was trained on. Pass the training result back
in:

.. code-block:: python

   trained = align_habitat_maps_to_prototypes(train_maps, models=train_models)
   named = align_habitat_maps_to_prototypes(new_maps, models=new_models, prototypes=trained)

The new habitats are assigned once to the stored prototypes; nothing is
updated, the stored z-score ``location`` / ``scale`` are reused, and the
aligned maps get the training ``model_id``. The feature names, ``metric``,
``standardize`` and ``reduction`` must equal the stored ones, otherwise
HABIT raises an error. A new subject with more habitats than stored
prototypes keeps the extra habitats unnamed (``prototype_id`` NA, local
id above :math:`K`): the training definition has no habitat for them.
Regression test: ``test_frozen_prototypes_name_new_subjects``.

Different habitat counts
^^^^^^^^^^^^^^^^^^^^^^^^

:math:`K` is always the largest subject habitat count, so every habitat
of every subject has a free prototype. No subject's habitat count ever
changes: habitats are neither merged (many-to-one is excluded by design)
nor dropped. A subject with fewer habitats simply lacks some prototype
ids; those habitats have zero volume in cohort tables.

Three subjects with 2, 3, and 4 habitats (one feature, relative
enhancement; result of the code, regression test
``test_different_habitat_counts_keep_every_habitat``):

.. list-table::
   :header-rows: 1

   * - Prototype
     - Value
     - a (2 habitats)
     - b (3 habitats)
     - c (4 habitats)
   * - 1
     - 0.5
     - 0.5
     - 0.6
     - 0.4
   * - 2
     - 1.0
     - —
     - —
     - 1.0
   * - 3
     - 1.5
     - —
     - 1.4
     - 1.6
   * - 4
     - 2.1
     - 2.0
     - 2.2
     - 2.1

For comparison, pairwise naming onto ``a`` (the smallest subject) leaves
``b``'s 1.4 and ``c``'s 1.0 both as "new id 3" although they were never
compared; that is why pairwise naming is not offered across subjects.

**Optional partial matching** (``max_distance``, off by default). Each
habitat also gets a private "unmatched" option at cost :math:`\lambda^2`.
The Hungarian solve then leaves a habitat unmatched exactly when every
free prototype is farther than :math:`\lambda` (an optimal partial
assignment, the same idea as unbalanced / partial optimal transport,
Chizat et al. 2018, and the unmatched-cluster handling of Azad et al.
2010). An unmatched habitat keeps a subject-local id above :math:`K` and
``prototype_id`` NA; such ids are **not** comparable across subjects, so
keep ``max_distance`` off when the aligned maps feed a cohort feature
table.

Assumptions and limitations
---------------------------

**Comparable features.** Matching compares centroid values directly.
They must mean the same thing in every subject: the same clustering
features, computed so that values are commensurate across patients.
``one_step`` models cluster raw voxel values by default, and raw MRI
signal is not comparable across scanners, protocols, or even patients on
one scanner. Use ratio or normalised maps (for example relative
enhancement via :class:`~habit.voxel_features.ExpressionVoxelFeatures`,
as the gallery page does) or a validated intensity normalisation before
clustering. ``standardize="zscore"`` only rescales each column on the
pooled cohort centroids (unit balance); it does not make incomparable
signal comparable.

**Every habitat is named, even in a shifted tumour.** With prototypes
low 0.8, mid 1.5, high 2.0 and a tumour whose three habitats are 2.6,
3.2, 4.0, one-to-one naming gives low / mid / high, although all three
enhance strongly. The ``distance`` column exposes this (here 1.8, 1.7,
and 2.0 for the three habitats). Report it, check large
distances, and prefer features that are truly comparable across
patients; ``max_distance`` can flag such habitats but then leaves them
without a shared name.

**Local optimum.** Minimising :math:`J` jointly over all subjects is a
multi-dimensional assignment problem, which is NP-hard in general. The
alternating search with multiple starts is a heuristic. Checked against
exhaustive enumeration (all per-subject permutations):

* the five-subject demo cohort with three habitats each (relative
  enhancement, fixed :math:`K=3`): identical optimum, :math:`J = 17.0687`;
* 200 random cohorts (3–5 subjects, :math:`K\in\{2,3\}`, noisy shifted
  prototypes): optimum found in 198; the two misses were 0.2 % and 1.4 %
  above the optimum.

Report ``objective``, ``converged``, and ``seed_subject_id`` with the
results.

**Not evidence that habitats are shared.** A small ``distance`` means the
naming is consistent, not that a habitat is the same biological
phenotype in every patient. For a cohort-level habitat definition prefer
a cohort model (two-step or direct pooling), which avoids post hoc
matching altogether; a hierarchical Dirichlet process (Teh et al. 2006)
is the model-based alternative in which groups share clusters by
construction. Strict cycle-consistent pairwise matching (permutation
synchronisation; Pachauri et al. 2013; Huang and Guibas 2013) is another
route and is not implemented.

Usage
-----

Same calls as :doc:`/auto_examples/04_habitat_maps/plot_05_match_labels`.
Each subject chooses its own habitat count (demo: 3, 2, 2, 2, 2):

.. code-block:: python

   from habit.contracts import Cohort, cohort_from_directory
   from habit.datasets import fetch_demo
   from habit.habitat_model import KMeansHabitatModelFitter
   from habit.pipeline import voxel_units
   from habit.precision import align_habitat_maps_to_prototypes
   from habit.voxel_features import ExpressionVoxelFeatures

   DATA = fetch_demo()  # your preprocessed root
   cohort = cohort_from_directory(DATA, modalities=("pre_contrast", "LAP", "PVP"), roi="LAP")
   extractor = ExpressionVoxelFeatures(
       features={
           "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
           "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
       },
       roi="LAP",
   )
   maps, models, fields = [], [], []
   for subject in cohort:
       field = extractor(subject)
       units = voxel_units(field)
       fitter = KMeansHabitatModelFitter(min_habitats=2, max_habitats=5, validation="silhouette", n_init=3)
       fitter.set_random_state(0)
       model = fitter.fit([units], cohort=Cohort([subject], name=subject.subject_id))
       maps.append(model.assigner()(units))
       models.append(model)
       fields.append(field)

   matched = align_habitat_maps_to_prototypes(maps, models=models)  # clustering centroids
   matched.habitat_maps    # same id = same prototype in every subject
   matched.prototypes      # (K, n_features), row k is habitat k + 1
   matched.assignments     # subject_id, habitat_id, prototype_id, distance

   align_habitat_maps_to_prototypes(maps, features=fields)           # per-habitat voxel means
   align_habitat_maps_to_prototypes(maps, centroids=my_matrices)     # your own summaries
   align_habitat_maps_to_prototypes(maps, models=models, metric="manhattan")  # other metric
   align_habitat_maps_to_prototypes(maps, models=models, max_distance=1.0)  # optional
   align_habitat_maps_to_prototypes(new_maps, models=new_models, prototypes=matched)  # frozen

The aligned maps share one new ``model_id`` (``prototype-…``) derived
from the prototypes and parameters, and their provenance records
``align_habitat_maps_to_prototypes`` with the source, ``metric``,
``reduction``, ``K``, ``max_distance``, and ``standardize`` (plus the
frozen ``model_id`` when prototypes were reused).

References
----------

Assignment

* Kuhn HW. The Hungarian method for the assignment problem. *Naval
  Research Logistics Quarterly* 1955;2:83–97. doi:10.1002/nav.3800020109
* Munkres J. Algorithms for the assignment and transportation problems.
  *J Soc Ind Appl Math* 1957;5:32–38. doi:10.1137/0105003

Label switching and prototype relabelling

* Stephens M. Dealing with label switching in mixture models. *J R Stat
  Soc B* 2000;62:795–809. doi:10.1111/1467-9868.00265
* Jasra A, Holmes CC, Stephens DA. Markov chain Monte Carlo methods and
  the label switching problem in Bayesian mixture modeling. *Stat Sci*
  2005;20. doi:10.1214/088342305000000016
* Wagstaff K, Cardie C, Rogers S, Schrödl S. Constrained k-means
  clustering with background knowledge. *Proc ICML* 2001:577–584
  (conference proceedings, no DOI).
* Dhillon IS, Modha DS. Concept decompositions for large sparse text
  data using clustering. *Mach Learn* 2001;42:143–175.
  doi:10.1023/A:1007612920971

Cross-sample cluster matching

* Pyne S, Hu X, Wang K, et al. Automated high-dimensional flow cytometric
  data analysis. *PNAS* 2009;106:8519–8524. doi:10.1073/pnas.0903028106
* Van Gassen S, Callebaut B, Van Helden MJ, et al. FlowSOM: using
  self-organizing maps for visualization and interpretation of cytometry
  data. *Cytometry A* 2015. doi:10.1002/cyto.a.22625
* Azad A, Langguth J, Fang Y, et al. Identifying rare cell populations in
  comparative flow cytometry. *Lecture Notes in Computer Science (WABI)*
  2010:162–175. doi:10.1007/978-3-642-15294-8_14
* Crow M, Paul A, Ballouz S, Huang ZJ, Gillis J. Characterizing the
  replicability of cell types defined by single cell RNA-sequencing data
  using MetaNeighbor. *Nat Commun* 2018. doi:10.1038/s41467-018-03282-0

Consistency and partial matching

* Pachauri D, Kondor R, Singh V. Solving the multi-way matching problem by
  permutation synchronization. *Advances in Neural Information Processing
  Systems 26* (NeurIPS) 2013 (conference proceedings, no DOI).
* Huang QX, Guibas L. Consistent shape maps via semidefinite programming.
  *Comput Graph Forum* 2013;32:177–186. doi:10.1111/cgf.12184
* Rubner Y, Tomasi C, Guibas LJ. The Earth Mover's Distance as a metric
  for image retrieval. *Int J Comput Vis* 2000;40:99–121.
  doi:10.1023/A:1026543900054
* Chizat L, Peyré G, Schmitzer B, Vialard FX. Scaling algorithms for
  unbalanced optimal transport problems. *Math Comput* 2018;87:2563–2609.
  doi:10.1090/mcom/3303
* Teh YW, Jordan MI, Beal MJ, Blei DM. Hierarchical Dirichlet processes.
  *J Am Stat Assoc* 2006;101:1566–1581. doi:10.1198/016214506000000302

Habitat imaging context

* Gatenby RA, Grove O, Gillies RJ. Quantitative imaging in cancer
  evolution and ecology. *Radiology* 2013. doi:10.1148/radiol.13122697
* Zhou M, Hall L, Goldgof D, et al. Radiologically defined ecological
  dynamics and clinical outcomes in glioblastoma multiforme: preliminary
  results. *Transl Oncol* 2014. doi:10.1593/tlo.13730
* Wu J, Cao G, Sun X, et al. Intratumoral spatial heterogeneity at
  perfusion MR imaging predicts recurrence-free survival in locally
  advanced breast cancer treated with neoadjuvant chemotherapy.
  *Radiology* 2018. doi:10.1148/radiol.2018172462
* Prior O, Macarro C, Navarro V, et al. Identification of precise 3D CT
  radiomics for habitat computation by machine learning in cancer.
  *Radiol Artif Intell* 2024;6(2):e230118. doi:10.1148/ryai.230118
