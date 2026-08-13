CLI / YAML batch workflows on demo_data
=======================================

Verified CLI commands that exercise the shipped ``demo_data/`` tree and
``config/`` YAMLs.

**Prefer the How-to chapter for a self-contained operator path** (terminal,
data layout, ★ fields, success checks):

* :doc:`../how_to/before_you_start` — project root and path rules
* :doc:`../how_to/preprocess` — preprocess / sort-dicom / dicom-info
* :doc:`../how_to/segment_habitat` — ``get-habitat`` + ``habit view``
* :doc:`../how_to/extract_features` / :doc:`../how_to/radiomics`
* :doc:`../how_to/train_model` / :doc:`../how_to/compare_models`
* :doc:`../how_to/auxiliary_tools` — icc / retest / merge-csv / dice

The How-to pages include the same demo commands that were verified against
local ``demo_data/``. This page remains a compact checklist for readers who
already know the layout.

Path semantics (short)
----------------------

* **v0.1 configs** — relative paths resolve against the YAML file's directory
  (shipped demos use ``../../demo_data/...`` from ``config/<domain>/``).
* **v1 configs** — paths as written; run from the repository root or use
  absolute paths.
* Validate first: ``habit check-config -c <yaml>``.

Command checklist
-----------------

::

   habit check-config -c config/habitat/config_habitat_two_step.yaml
   habit preprocess -c config/preprocessing/config_preprocessing_demo.yaml
   habit get-habitat -c config/habitat/config_habitat_two_step.yaml
   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd
   habit extract -c config/feature_extraction/config_extract_features_demo.yaml
   habit radiomics -c config/radiomics/config_traditional_radiomics.yaml
   habit model -c config/machine_learning/config_machine_learning_radiomics.yaml -m train
   habit model -c config/machine_learning/config_machine_learning_clinical.yaml -m train
   habit cv -c config/machine_learning/config_machine_learning_kfold_demo.yaml
   habit compare -c config/model_comparison/config_model_comparison_demo.yaml
   habit icc -c config/auxiliary/config_icc_demo.yaml
   habit dicom-info -i demo_data/dicom -o demo_data/results/htg_dicom_info.csv --one-file-per-folder
   habit merge-csv demo_data/ml_data/breast_cancer_dataset.csv demo_data/ml_data/clinical_feature.csv -o demo_data/results/htg_merged.csv --index-col subject_id
   habit dice --input1 demo_data/preprocessed --input2 demo_data/preprocessed --output demo_data/results/htg_dice_results.csv

Predict / migrate extras::

   habit get-habitat -c config/habitat/config_habitat_two_step_predict.yaml -m predict
   habit migrate-config -c config/habitat/config_habitat_two_step.yaml --dry-run

Figures
-------

``habit get-habitat`` writes the same maps the Python recipes do. After a
two-step run, overlay anatomy + ``*_habitats.nrrd``::

   from habit.viz import plot_habitat_overlay
   import SimpleITK as sitk

   image = sitk.GetArrayFromImage(sitk.ReadImage("demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd"))
   labels = sitk.GetArrayFromImage(sitk.ReadImage("demo_data/results/habitat_two_step/subj001_habitats.nrrd"))
   fig = plot_habitat_overlay(image, labels, axis=0, title="CLI two-step habitats")

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Two-step habitat overlay from get-habitat
   :width: 420

   Overlay from the two-step gallery (same product as ``habit get-habitat``)
   (:func:`~habit.viz.plot_habitat_overlay`).

What to read next
-----------------

* :doc:`../how_to/index` — full operator manual
* :doc:`run_from_yaml` — programmatic twin
* :doc:`../configuration/index` — YAML field reference
