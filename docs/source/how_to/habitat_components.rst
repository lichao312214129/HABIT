Habitat Spec component catalog
==============================

**Reference** chooser for registered ``Spec`` names. Walk-throughs stay in
the Habitat Guide (:doc:`../examples/index`). This page lists every built-in
name, constructor parameter, and the Python / YAML twin.

Concept and embedding: :doc:`../tutorial/habitat_analysis` ·
:doc:`../examples/habitat_atomic_ops`.

The gallery scripts (one-step, two-step, direct-pooling) show **one**
worked :class:`~habit.spec.HabitatSpec`. This page is the chooser from
that example outward: which registered names exist for each stage, what
each parameter means, and how to write the same choice in Python and YAML.

When a recipe shows ``Spec("raw")`` or ``Spec("kmeans")``, look up that
stage below. Parameter tables are generated at Sphinx build time from each
component constructor. Do not copy them into notebooks — look names and
constructor signatures up at runtime::

   from habit.api.plugins import list_plugins
   from habit.spec import parse_feature_expression
   from habit.voxel_features import RawVoxelFeatures, VoxelFeatureExtractorRegistry

   print([info.name for info in list_plugins("voxel_feature_extractor")])
   print(VoxelFeatureExtractorRegistry.constructor_signature("raw"))
   voxel = VoxelFeatureExtractorRegistry.create("raw", modality="T1")
   voxel = RawVoxelFeatures(modality="T1")

Full live catalog (every domain): :doc:`../api/plugins`.

How a ``HabitatSpec`` is assembled
----------------------------------

A habitat study is an **ordered list of stages**. Each
:class:`~habit.spec.Stage` is a label plus a :class:`~habit.spec.Spec`.
A leaf is one extractor on one series (section 1A). Combining series or
families is a tree from :func:`~habit.spec.parse_feature_expression`
(section 1B)::

   Stage("<label>", Spec("<registered name>", {<params>}))
   Stage("<label>", parse_feature_expression('concat(raw("T1"), voxel_radiomics("T2"))'))

Stage labels (``extract_voxel_features``, ``quantify2``, …) are **not**
role keywords. HABIT infers the scientific role from position + the
component's registry domain. Recommended labels:

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - Recommended label
     - Domain (``list_plugins``)
     - Role
   * - ``extract_voxel_features``
     - ``voxel_feature_extractor``
     - Required first step
   * - ``preprocess1`` / ``preprocess2`` / …
     - ``feature_preprocessing_method``
     - Optional; repeatable; position decides voxel vs post-pool
   * - ``partition``
     - ``supervoxelizer``
     - two-step only
   * - ``extract_supervoxel_features``
     - ``supervoxel_feature_extractor``
     - two-step optional
   * - ``pool``
     - ``pooling``
     - two-step / direct-pooling watershed
   * - ``fit``
     - ``habitat_model_fitter``
     - Required
   * - ``assign``
     - ``habitat_assigner``
     - Required
   * - ``quantify`` / ``quantify2`` / …
     - ``habitat_feature_extractor``
     - Optional; repeatable

Strategy is inferred from the sequence:

* **two_step** — ``partition`` + ``pool``
* **direct_pooling** — ``pool`` only (no partition)
* **one_step** — neither partition nor pool (per-subject habitats)

``kmeans`` and ``gmm`` exist in **two** domains (supervoxelizer and
habitat-model fitter). Place them before ``pool`` to partition, or
immediately before ``assign`` to fit habitats.

Python and YAML are the same document
-------------------------------------

``Spec.to_dict()`` is the YAML component block. A v1 habitat document
stores the same mapping under ``spec:``.

.. code-block:: python

   from habit.spec import HabitatSpec, Spec, Stage

   spec = HabitatSpec(
       name="habitat_one_step",
       stages=(
           Stage("extract_voxel_features", Spec("raw", {"modalities": ["LAP"]})),
           Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow"})),
           Stage("assign", Spec("nearest_centroid")),
           Stage("quantify", Spec("volume")),
       ),
       random_seed=42,
   )

Equivalent v1 YAML (``stages`` form)::

   spec:
     name: habitat_one_step
     random_seed: 42
     stages:
       - name: extract_voxel_features
         component:
           name: raw
           params:
             modalities: [LAP]
       - name: fit
         component:
           name: kmeans
           params:
             min_habitats: 2
             max_habitats: 10
             validation: elbow
       - name: assign
         component:
           name: nearest_centroid
           params: {}
       - name: quantify
         component:
           name: volume
           params: {}

Named-field sugar (``voxel_feature_extractor:``, ``habitat_model_fitter:``,
…) expands to the same stages. Prefer ``stages`` for new Python; both
round-trip. See :doc:`../api/spec` and
:doc:`../configuration/habitat`.

1. Voxel feature extraction
---------------------------

**Required.** Turns each subject's images + ROI into a per-voxel feature
field. First pick **one extractor on one series** (a leaf). Only then
compose several series or families with a combiner. A longer
``modalities`` list is not a substitute for mixing families.

.. _chooser-voxel-leaf:

A. Single-modality voxel extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One method, one series, one column (or one block of columns). No
``concat``. The expression form quotes the modality; the ``Spec`` form
uses ``modality=``::

   from habit.spec import parse_feature_expression

   Stage("extract_voxel_features", Spec("raw", {"modality": "T1"}))
   Stage(
       "extract_voxel_features",
       parse_feature_expression('raw("T1")'),
   )
   Stage(
       "extract_voxel_features",
       parse_feature_expression('local_entropy("T1", kernel_size=3, bins=32)'),
   )
   Stage(
       "extract_voxel_features",
       parse_feature_expression('voxel_radiomics("T2", kernel_radius=3)'),
   )

YAML::

   - name: extract_voxel_features
     component:
       name: raw
       params:
         modality: T1

``voxel_radiomics`` needs the ``pyradiomics`` extra and is much slower
than ``raw`` / ``local_entropy``. Matrix construction can use HABIT's
GPU path (``use_gpu_matrices``) instead of the PyRadiomics C extension;
see :doc:`voxel_texture`.

.. _chooser-voxel-compose:

B. Multi-modality voxel composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A combiner joins **already-extracted** leaves on the same voxels (same
ROI). Two spellings — expression string or nested ``children`` — share
one fingerprint. Modalities in the expression form are **quoted**.

**Same family, several series.** A leaf still accepts a modality list
(``raw("T1", "T2")`` or ``modalities: [T1, T2]``). ``concat`` of two
``raw`` leaves is the same idea written as a tree::

   Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]}))
   Stage(
       "extract_voxel_features",
       parse_feature_expression('concat(raw("T1"), raw("T2"))'),
   )

YAML::

   - name: extract_voxel_features
     component:
       name: raw
       params:
         modalities: [T1, T2]

**Different families.** Texture on T2, intensity on T1 — this needs a
tree, not a longer ``modalities`` list::

   Stage(
       "extract_voxel_features",
       parse_feature_expression(
           'concat(raw("T1"), voxel_radiomics("T2", kernel_radius=3))'
       ),
   )
   Stage(
       "extract_voxel_features",
       parse_feature_expression(
           'concat(local_entropy("T1", kernel_size=3, bins=32), raw("T2"))'
       ),
   )

Structured form of the first tree (same Spec)::

   Stage(
       "extract_voxel_features",
       Spec(
           "concat",
           {
               "children": [
                   {"name": "raw", "params": {"modality": "T1"}},
                   {
                       "name": "voxel_radiomics",
                       "params": {"modality": "T2", "kernel_radius": 3},
                   },
               ],
           },
       ),
   )

YAML (expression — shortest)::

   - name: extract_voxel_features
     component: 'concat(raw("T1"), voxel_radiomics("T2", kernel_radius=3))'

YAML (structured)::

   - name: extract_voxel_features
     component:
       name: concat
       params:
         children:
           - name: raw
             params: {modality: T1}
           - name: voxel_radiomics
             params: {modality: T2, kernel_radius: 3}

**Derived channels — one combiner each.** Do not nest ``ratio``,
``weighted_concat``, and a texture leaf in one expression on this page.
``as_="label"`` renames a **single-column** node so two branches do not
collide::

   Stage(
       "extract_voxel_features",
       parse_feature_expression(
           'ratio(raw("T1"), raw("T2"), as_="t1_over_t2")'
       ),
   )
   Stage(
       "extract_voxel_features",
       parse_feature_expression(
           'weighted_concat(raw("T1", as_="t1w"), raw("T2", as_="t2w"), weights=[2.0, 1.0])'
       ),
   )

Built-ins: ``concat``, ``weighted_concat`` (``weights=[...]``),
``average``, ``ratio``, ``difference``, ``kinetic``, ``expression``.
Grammar (strict; bad input is rejected, not guessed): modalities are
quoted (``raw("T1")``); parameters are ``key=value``; a quoted string
among children is an implicit ``raw``. Bare ``raw(T1)`` is v0.1-only.

Nested trees and column names: :doc:`../examples/feature_composition`.
Arithmetic beyond combiners: :doc:`../examples/custom_voxel_features`
(``expression`` or a custom plugin).

.. include:: _generated_catalog_voxel_feature_extractor.rst

2. Feature preprocessing
------------------------

**Optional, repeatable.** Same method names before and after ``pool``.
Before ``partition`` / ``fit`` they scale the units that clustering sees;
after ``pool`` they are cohort-level and travel with
:class:`~habit.contracts.HabitatModel`.

Typical voxel-level chain: ``winsorize`` then ``minmax``. Do not skip
scaling on two-step / direct-pooling runs — see
:doc:`../examples/habitat_preprocessing`.

Python::

   Stage("preprocess1", Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}))
   Stage("preprocess2", Spec("minmax", {"across_features": False}))

YAML::

   - name: preprocess1
     component:
       name: winsorize
       params:
         winsor_limits: [0.05, 0.05]
         across_features: false
   - name: preprocess2
     component:
       name: minmax
       params:
         across_features: false

.. include:: _generated_catalog_feature_preprocessing_method.rst

3. Supervoxel partition
-----------------------

**two-step only.** All built-in names use ``n_supervoxels`` (not sklearn's
``n_clusters`` / ``n_components``).

Python::

   Stage("partition", Spec("slic", {"n_supervoxels": 50}))
   Stage("partition", Spec("kmeans", {"n_supervoxels": 50, "n_init": 10}))

YAML::

   - name: partition
     component:
       name: kmeans
       params:
         n_supervoxels: 50
         n_init: 10

.. include:: _generated_catalog_supervoxelizer.rst

4. Supervoxel features
----------------------

**two-step optional.** This stage describes **each supervoxel**, after
``partition``. It does **not** replace voxel extraction: mixed T1/T2
science is usually built in section 1, then aggregated here.

The default is to average the voxel field you already built
(``mean_voxel_features``). Omit this stage unless you need a different
description; many two-step recipes rely on the partition's attached
means::

   Stage("extract_supervoxel_features", Spec("mean_voxel_features"))

YAML::

   - name: extract_supervoxel_features
     component:
       name: mean_voxel_features
       params: {}

.. _chooser-supervoxel-leaf:

A. Single-modality supervoxel extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One method, one series. ``mean`` / ``std`` / ``percentile`` aggregate
that series' voxel signal per supervoxel. ``source="working"`` (default)
uses the preprocessed voxel field; ``source="original"`` uses the raw
series the partition saw. ``supervoxel_radiomics`` is **whole-region**
texture per supervoxel label (not a sliding voxel kernel)::

   Stage(
       "extract_supervoxel_features",
       parse_feature_expression('mean("T1")'),
   )
   Stage(
       "extract_supervoxel_features",
       parse_feature_expression('std("T1", as_="t1_spread")'),
   )
   Stage(
       "extract_supervoxel_features",
       parse_feature_expression('percentile("T2", q=90)'),
   )
   Stage(
       "extract_supervoxel_features",
       parse_feature_expression('supervoxel_radiomics("T2")'),
   )

``supervoxel_radiomics`` needs ``pyradiomics``. Values differ from
``voxel_radiomics`` (different spatial support).

.. _chooser-supervoxel-compose:

B. Multi-modality supervoxel composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compose the leaves the same way as voxel combiners. Keep each combiner
to one job — statistics together, or a statistic plus radiomics — and
leave nested trees to :doc:`../examples/feature_composition`::

   Stage(
       "extract_supervoxel_features",
       parse_feature_expression(
           'concat(mean("T1"), std("T1", as_="t1_spread"), percentile("T2", q=90))'
       ),
   )
   Stage(
       "extract_supervoxel_features",
       parse_feature_expression(
           'concat(mean("T1"), supervoxel_radiomics("T2"))'
       ),
   )

YAML::

   - name: extract_supervoxel_features
     component: 'concat(mean("T1"), std("T1", as_="t1_spread"), percentile("T2", q=90))'

.. include:: _generated_catalog_supervoxel_feature_extractor.rst

5. Pool
-------

**two-step and direct-pooling.** Marks the subject → cohort watershed.
There is one built-in name.

Python::

   Stage("pool", Spec("pool"))

YAML::

   - name: pool
     component:
       name: pool
       params: {}

.. include:: _generated_catalog_pooling.rst

6. Fit habitats
---------------

**Required.** Built-in names: ``kmeans`` and ``gmm``. Learns centroids
(cohort-level when ``pool`` is present; per-subject otherwise).

Shared parameters: ``n_habitats``, ``min_habitats``, ``max_habitats``,
``validation``, ``n_init``, ``max_iter``. ``gmm`` also takes
``covariance_type`` (``full`` / ``tied`` / ``diag`` / ``spherical``).

Omit ``n_habitats`` (or pass ``None``) to select K over
``min_habitats..max_habitats`` by ``validation``. There is no string
``"auto"``.

Allowed ``validation`` values:

* **kmeans** — ``elbow`` (default; alias of ``kneedle``), ``kneedle``,
  ``inertia``, ``silhouette``, ``calinski_harabasz``, ``davies_bouldin``,
  ``gap``. A list of these casts one vote each.
* **gmm** — ``bic`` (default), ``aic``, ``davies_bouldin`` (minimise);
  ``silhouette``, ``calinski_harabasz``, ``gap`` (maximise).

Copy-paste a ``fit`` stage::

   Stage(
       "fit",
       Spec(
           "kmeans",
           {
               "min_habitats": 2,
               "max_habitats": 10,
               "validation": "elbow",
               "n_init": 5,
           },
       ),
   )
   Stage("fit", Spec("gmm", {"n_habitats": 4, "covariance_type": "full"}))

YAML::

   - name: fit
     component:
       name: kmeans
       params:
         min_habitats: 2
         max_habitats: 10
         validation: elbow
         n_init: 5

.. include:: _generated_catalog_habitat_model_fitter.rst

7. Assign
---------

**Required.** Maps each unit to the nearest habitat centroid. The built-in
name is ``nearest_centroid``. After ``fit``,
``model.assigner()`` is the same object.

Python::

   Stage("assign", Spec("nearest_centroid"))

YAML::

   - name: assign
     component:
       name: nearest_centroid
       params: {}

.. include:: _generated_catalog_habitat_assigner.rst

8. Quantify
-----------

**Optional, repeatable.** Does not change the habitat map. Light families
(``volume``, ``msi``, ``ith_score``, ``non_radiomics``, ``graph``) need no
extra extra. ``traditional`` / ``whole_habitat`` / ``each_habitat`` need
``pyradiomics``.

Python::

   Stage("quantify", Spec("volume"))
   Stage("quantify2", Spec("msi"))
   Stage("quantify3", Spec("ith_score"))
   Stage("quantify4", Spec("non_radiomics"))
   Stage(
       "quantify5",
       Spec("graph", {"edge_method": "min_distance", "node_method": "uniform_grid"}),
   )

One-step streaming figures (not stages — do not enter the fingerprint).
Pass the same :class:`~habit.kernels.HabitatGraphFeatureOptions` to
``Spec("graph")`` and the graph atoms. Catalog: :doc:`../examples/visualization`.

Python::

   from habit.adapters import DirectoryResultWriter
   from habit.kernels import HabitatGraphFeatureOptions
   from habit.report import (
       ClusterValidation,
       GraphNetwork2D,
       GraphSlice,
       ITH,
       MSI,
       Overlay,
       Report,
       VolumeFractions,
   )

   graph = HabitatGraphFeatureOptions(edge_method="min_distance", block_size=8)
   writer = DirectoryResultWriter("out/study")
   report = Report(
       figures=(
           Overlay(modality="T1"),
           VolumeFractions(),
           MSI(),
           ITH(),
           ClusterValidation(),
           GraphSlice(options=graph),
           GraphNetwork2D(options=graph),
       ),
       figure_layout="by_subject",
       writer=writer,
   )

YAML::

   - name: quantify
     component:
       name: volume
       params: {}
   - name: quantify2
     component:
       name: msi
       params: {}

.. include:: _generated_catalog_habitat_feature_extractor.rst

What to read next
-----------------

* :doc:`../examples/two_step_habitat` — typical paper pipeline
* :doc:`../examples/one_step_habitat` — per-subject habitats
* :doc:`../examples/feature_composition` — worked concat / ratio / ``as_`` trees
* :doc:`../api/spec` — fingerprints, sugar, ``RunPolicy``
* :doc:`../api/plugins` — every domain, including table-ML
* :doc:`segment_habitat` — CLI / YAML bookmark only
