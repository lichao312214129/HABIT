Quickstart: Python API (5 minutes)
==================================

This is the developer quickstart for the v1.0 **API-first** HABIT: the
Python API is the product, and the CLI/YAML/GUI are thin shells over it.
Prefer clicking through a config file? See the parallel
:doc:`quickstart` (YAML + CLI, no programming).

Prerequisites: :doc:`installation` (``pip install -e .``), Python 3.10.

The two objects you will always meet
------------------------------------

Everything in v1.0 is one of two things:

* a **contract** — plain data (:class:`~habit.contracts.Cohort`,
  :class:`~habit.contracts.HabitatModel`,
  :class:`~habit.contracts.FeatureTable`), and
* a **recipe** — a named study design taking a contract plus a **spec**
  (:class:`~habit.spec.HabitatSpec`, :class:`~habit.spec.MLSpec`) and
  returning a typed result.

Specs are frozen value objects; the same document written as YAML is read by
the CLI. Nothing below touches disk until you say so.

Your first habitat analysis
---------------------------

Fifteen lines, fully deterministic, no files needed — a synthetic cohort
stands in for real images:

.. code-block:: python

   from habit import HabitatSpec, Spec, make_synthetic_cohort
   import habit.recipes as recipes

   cohort = make_synthetic_cohort(n_subjects=6, shape=(24, 24, 24), rng=42)

   spec = HabitatSpec(
       name="habitat_two_step",
       voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
       supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
       habitat_model_fitter=Spec(
           "kmeans",
           {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 5},
       ),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(Spec("volume"),),
       random_seed=42,
   )

   result = recipes.two_step(cohort, spec)
   print(result.habitat_model.summary())

Real output::

   HabitatModel kmeans-1f45d79eaaa3d7b5
     habitats           : 3
     features (2)    : T1, T2
     defining cohort    : n=6, name=synthetic
     modalities         : T1, T2
     cohort digest      : 9e5093ef0a362899...
     produced by        : habitat_model_fitter.kmeans
     habit version      : 1.0.0
     random seed        : 42
     preprocessing state: inertia, selection_report, validation

``result`` holds everything in memory: ``result.habitat_model`` (the fitted
definition), ``result.habitat_maps`` (one label map per subject),
``result.features`` (the habitat feature table), and ``result.manifest``
(provenance, including an auto-generated methods paragraph). Persist with
``result.save("out/study")``.

On real data, only the first line changes::

   from habit import cohort_from_directory
   cohort = cohort_from_directory("processed_images", modalities=["T1", "T2"], roi="T1")

Save the model, apply it later
------------------------------

.. code-block:: python

   from habit import HabitatModel

   result.habitat_model.save("out/habitat_model.habitatmodel")   # self-describing archive
   model = HabitatModel.load("out/habitat_model.habitatmodel")   # later / elsewhere
   prediction = recipes.apply_habitat_model(new_cohort, spec, model)

No refitting happens: the model replays its stored preprocessing state, so
train and predict stay consistent. Full walkthrough:
:doc:`../examples/apply_saved_model`.

Your first tabular model
------------------------

Habitat features (or radiomics, or clinical variables) form a
:class:`~habit.contracts.FeatureTable`; the ML recipes model it directly:

.. code-block:: python

   from habit import MLSpec, Spec, make_synthetic_feature_table
   import habit.recipes as recipes

   table = make_synthetic_feature_table(n_rows=80, n_features=8, rng=42)
   # steps runs in list order. Variance goes first, on the raw table: after
   # z-score every feature variance is ~1, so it would select nothing useful.
   spec = MLSpec(
       name="demo",
       steps=(
           Spec("variance", {"threshold": 0.01}),
           Spec("zscore"),
       ),
       classifier=Spec("LogisticRegression", {"max_iter": 500}),
       metrics=(Spec("accuracy"), Spec("auc")),
   )

   result = recipes.train_model(table, spec, test_size=0.25, seed=42)
   print("Test:", {k: round(v, 3) for k, v in result.test_metrics.items()})
   cv = recipes.cross_validate(table, spec, n_splits=5, seed=42)
   print("CV mean:", {k: round(v, 3) for k, v in cv.mean_metrics.items()})

Real output (the synthetic table has one informative feature, so scores are
near-perfect by construction)::

   Test: {'accuracy': 1.0, 'auc': 1.0}
   CV mean: {'accuracy': 0.988, 'auc': 1.0}

Under a split or a fold the pipeline sees the training rows **only** —
preprocessing and selection can never leak.

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - I want to…
     - Read
   * - Run the same thing from a config file (no code)
     - :doc:`quickstart` (YAML + CLI)
   * - Staged selection (variance before z-score, ANOVA after)
     - :doc:`../examples/ml_advanced`
   * - See full runnable studies with output
     - :doc:`../examples/index`
   * - Load my own images / tables
     - :doc:`../api/data_model` · :doc:`../api/adapters`
   * - Understand specs and migrate v0.1 YAML
     - :doc:`../api/spec`
   * - Look up a class or function
     - :doc:`../api/index` (API Reference)
   * - Tune every YAML field
     - :doc:`../configuration/index`
