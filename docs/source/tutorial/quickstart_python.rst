Quickstart: Python API
========================

Install first (:doc:`installation`). YAML users: :doc:`quickstart`.

1. Habitat analysis (no files)
-------------------------------

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
       habitat_features=(
           Spec("volume"),
           Spec("msi"),
           Spec("ith_score"),
           Spec("non_radiomics"),
           # Spec("traditional"), Spec("whole_habitat"), Spec("each_habitat"),
       ),
       random_seed=42,
   )
   result = recipes.two_step(cohort, spec)
   print(result.habitat_model.summary())
   result.save("out/study")   # optional: write maps + CSV + model

Real data folder (demo pack)::

   from habit import cohort_from_directory
   cohort = cohort_from_directory(
       "demo_data/preprocessed",
       modalities=["pre_contrast", "LAP", "PVP", "delay_3min"],
       roi="LAP",
   )

Or a path-list YAML via :doc:`../how_to/prepare_data` +
``habit.recipes.run_from_yaml(...)``.

2. View
-------

.. code-block:: python

   from habit.viz import view_habitat_napari

   volume = cohort[0].image("T1")
   view_habitat_napari(
       volume.data,
       result.habitat_maps[0].label_array,
       spacing=volume.spacing,
       direction=volume.direction,
   )

Needs napari (:doc:`installation`). Blocks until you close the window.
For fuller 3D review, also open the source volume and ``*_habitats.nrrd``
in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer.

3. Apply a saved model
----------------------

.. code-block:: python

   from habit import HabitatModel

   result.habitat_model.save("out/habitat_model.habitatmodel")
   model = HabitatModel.load("out/habitat_model.habitatmodel")
   prediction = recipes.apply_habitat_model(new_cohort, spec, model)

4. Tabular ML
-------------

.. code-block:: python

   from habit import MLSpec, Spec, make_synthetic_feature_table
   import habit.recipes as recipes

   table = make_synthetic_feature_table(n_rows=80, n_features=8, rng=42)
   spec = MLSpec(
       name="demo",
       steps=(Spec("variance", {"threshold": 0.01}), Spec("zscore")),
       classifier=Spec("LogisticRegression", {"max_iter": 500}),
       metrics=(Spec("accuracy"), Spec("auc")),
   )
   result = recipes.train_model(table, spec, test_size=0.25, seed=42)
   print(result.test_metrics)

Next: :doc:`../examples/index` · :doc:`../api/index` · :doc:`../how_to/prepare_data`
