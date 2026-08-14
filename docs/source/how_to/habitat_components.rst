Choosing habitat Spec components
================================

When a recipe shows ``Spec("raw")`` or ``Spec("kmeans")``, this page is the
chooser: **which names exist for each stage, what each parameter means, and
how to write the same choice in Python and YAML**.

Parameter tables below are generated at Sphinx build time from each
component's ``params_model`` (plus the class ``Args:`` text). Do not copy
them into notebooks — look them up::

   from habit import get_param_schema, list_plugins, plugin_catalog

   print([info.name for info in list_plugins("voxel_feature_extractor")])
   print(get_param_schema("raw", "voxel_feature_extractor").model_fields)
   for row in plugin_catalog("habitat_model_fitter"):
       for param in row.params:
           print(row.name, param.name, param.default, param.allowed, param.description)

Full live catalog (every domain): :doc:`../api/plugins`.

How a ``HabitatSpec`` is assembled
----------------------------------

A habitat study is an **ordered list of stages**. Each
:class:`~habit.spec.Stage` is a label plus a :class:`~habit.spec.Spec`::

   Stage("<label>", Spec("<registered name>", {<params>}))

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

   from habit import HabitatSpec, Spec, Stage

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
field. ``raw`` is the usual start (one column per modality). Use
``local_entropy`` / ``voxel_radiomics`` when texture should define
habitats; ``concat`` / ``expression`` / ``kinetic`` to combine leaves.

Python::

   Stage("extract_voxel_features", Spec("raw", {"modalities": ["LAP"]}))
   Stage(
       "extract_voxel_features",
       Spec("local_entropy", {"modalities": ["LAP"], "kernel_size": 3, "bins": 32}),
   )

YAML::

   - name: extract_voxel_features
     component:
       name: raw
       params:
         modalities: [LAP]

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

**two-step optional.** Default science is ``mean_voxel_features`` (mean of
the voxel field inside each supervoxel). Omit the stage unless you need
radiomics or other statistics on the partition.

Python::

   Stage("extract_supervoxel_features", Spec("mean_voxel_features"))

YAML::

   - name: extract_supervoxel_features
     component:
       name: mean_voxel_features
       params: {}

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

**Required.** Learns centroids (cohort-level when ``pool`` is present;
per-subject otherwise). Omit ``n_habitats`` (or pass ``None``) to select
K over ``min_habitats..max_habitats`` by ``validation``. There is no
string ``"auto"``.

Python::

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
       Spec("graph", {"edge_method": "centroid_distance", "distance_threshold": 5.0}),
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

* :doc:`segment_habitat` — CLI / YAML operator path
* :doc:`../examples/one_step_habitat` — the example that prompted this page
* :doc:`../examples/two_step_habitat` — partition + pool
* :doc:`../examples/habitat_analysis_overview` — recipe / atomic / custom
* :doc:`../api/spec` — fingerprints, sugar, ``RunPolicy``
* :doc:`../api/plugins` — every domain, including table-ML
