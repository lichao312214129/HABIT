Quickstart: Python API
========================

Install first (:doc:`installation`). This page is the **same demo** as
:doc:`quickstart`, driven by the **same YAML files** through
:func:`~habit.recipes.run_from_yaml` (the programmatic twin of the CLI).

Work from a directory that already has ``config/`` and ``demo_data/``
(see :doc:`quickstart` steps 1-2)::

   # Windows - Anaconda Prompt
   conda activate habit
   cd D:\my_habit_work          # your work_dir (has config/ + demo_data/)

Paths inside the demo YAML resolve from each YAML file's folder
(``../../demo_data/...``), exactly as for ``habit get-habitat`` /
``habit extract`` / ``habit model``.

1. Habitat analysis (same YAML as CLI)
--------------------------------------

CLI twin::

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

.. code-block:: python

   import habit.recipes as recipes

   result = recipes.run_from_yaml(
       "config/habitat/config_habitat_two_step.yaml",
       workflow="habitat",
       save=True,  # write maps / CSV / model under the YAML out_dir
   )
   print(result.habitat_model.summary())

Outputs land under ``demo_data/results/habitat_two_step/`` (including
``*_habitats.nrrd`` and ``habitat_model.habitatmodel``). With the same
YAML, seed, and data, API and CLI habitat label maps match voxel-wise.

In-memory recipes without YAML (synthetic cohorts, custom ``HabitatSpec``)
live under :doc:`../examples/index` -- useful for notebooks, not for
reproducing this demo's CLI numbers.

2. View
-------

CLI twin::

   habit view demo_data/preprocessed/images/subj001/LAP/...nrrd \
              demo_data/results/habitat_two_step/subj001_habitats.nrrd

.. code-block:: python

   from habit import cohort_from_directory
   from habit.viz import view_habitat_napari

   cohort = cohort_from_directory(
       "demo_data/preprocessed",
       modalities=["pre_contrast", "LAP", "PVP", "delay_3min"],
       roi="LAP",
   )
   # result.habitat_maps are in the same subject order as the YAML cohort
   volume = cohort[0].image("LAP")
   view_habitat_napari(
       volume.data,
       result.habitat_maps[0].label_array,
       spacing=volume.spacing,
       direction=volume.direction,
   )

Needs napari (:doc:`installation`). Blocks until you close the window.
For fuller 3D review, also open the source volume and ``*_habitats.nrrd``
in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer.

3. Apply a saved model (same YAML as CLI predict)
-------------------------------------------------

After step 1, the archive sits at
``demo_data/results/habitat_two_step/habitat_model.habitatmodel``.
CLI twin::

   habit get-habitat --config config/habitat/config_habitat_two_step_predict.yaml -m predict

.. code-block:: python

   import habit.recipes as recipes

   prediction = recipes.run_from_yaml(
       "config/habitat/config_habitat_two_step_predict.yaml",
       workflow="habitat",
       save=True,
   )
   print(len(prediction.habitat_maps), "subjects labelled")

4. Extract habitat features (same YAML as CLI)
----------------------------------------------

CLI twin::

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Needs step 1 outputs (``*_habitats.nrrd`` under the habitats folder named in
that YAML).

.. code-block:: python

   import habit.recipes as recipes

   extract_result = recipes.run_from_yaml(
       "config/feature_extraction/config_extract_features_demo.yaml",
       workflow="extract",
   )
   print(extract_result.output_dir)

5. Tabular ML (same YAML as CLI)
--------------------------------

CLI twin::

   habit model --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml --mode train

Needs the **ML pack** under ``demo_data/ml_data/`` (see :doc:`quickstart`
step 2 / the note before ``habit model``).

.. code-block:: python

   import habit.recipes as recipes

   ml_result = recipes.run_from_yaml(
       "config/machine_learning/config_machine_learning_radiomics_minimal.yaml",
       workflow="model",
       save=True,
   )
   print(ml_result.test_metrics)

Next: :doc:`../examples/index` / :doc:`../api/index` / :doc:`../how_to/prepare_data`
