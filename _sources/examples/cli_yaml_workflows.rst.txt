CLI / YAML batch workflows on demo_data
=======================================

Every command on this page was executed verbatim against the repository's
local ``demo_data/`` tree (real DICOM series, preprocessed NIfTI images and
ML feature tables) and the outputs were verified on disk. The shipped demo
configs write to ``demo_data/results/<name>/``; the verification run used
identical copies that only redirected ``out_dir`` to
``demo_data/results/cli_yaml/<name>/`` — nothing else was changed.

The CLI is a thin shell over the Python API: each subcommand loads a YAML
document, translates v0.1 documents through
:class:`~habit.spec.legacy.LegacyConfigAdapter`, and calls the same recipe
:func:`~habit.recipes.run_from_yaml` dispatches to. Anything shown here can
also be driven from Python.

Path semantics
--------------

* **v0.1 configs** — relative paths resolve against the YAML file's own
  directory, not the shell's working directory. The shipped configs use
  ``../../demo_data/...`` because they live in ``config/<domain>/``.
* **v1 configs** (``version: '1.0'``) — paths are used as written (current
  working directory); run them from the repository root or use absolute
  paths.
* **Manifests** — ``data_dir`` may point at another YAML listing
  ``images:``/``masks:`` per subject and modality (see
  ``config/habitat/file_habitat_demo.yaml``,
  ``config/preprocessing/files_preprocessing_dcm2nii_demo.yaml``).

Validate and migrate configs
----------------------------

``check-config`` validates syntax plus the workflow schema without running
anything (``-w`` picks the schema when the path gives no hint); manifests and
PyRadiomics presets take ``--syntax-only``::

   habit check-config -c config/habitat/config_habitat_two_step_cli_demo.yaml -w habitat
   habit check-config -c config/habitat/file_habitat_demo.yaml --syntax-only
   habit check-config -c config/radiomics/parameter.yaml --syntax-only

``migrate-config`` upgrades a v0.1 document to the native v1
``spec/data/policy/output`` layout (``--dry-run`` prints a diff)::

   habit migrate-config -c config/habitat/config_habitat_two_step_cli_demo.yaml \
       -o habitat_two_step_cli.v1.yaml
   habit migrate-config -c config/preprocessing/config_preprocessing_demo.yaml --dry-run

All 64 shipped configs plus 6 manifests and 6 radiomics parameter presets
pass ``check-config`` in the verification run.

DICOM tools
-----------

``dicom-info`` scans a DICOM tree (one file per folder is enough when each
folder holds one series) and exports the tag table::

   habit dicom-info -i demo_data/dicom -o dicom_info.csv --one-file-per-folder

``sort-dicom`` drives dcm2niix to sort/rename a raw DICOM tree
(``config/dicom_sort/config_sort_dicom.yaml`` points at
``demo_data/dicom`` and ``tools/bin/dcm2niix.exe``)::

   habit sort-dicom -c config/dicom_sort/config_sort_dicom.yaml

Image preprocessing
-------------------

``habit preprocess -c <yaml>`` runs the step chain in YAML order. Verified
against ``demo_data/preprocessed/processed_images`` (2 subjects x 3 DCE
phases + masks) and the raw DICOM tree:

* ``config_preprocessing_demo.yaml`` — resample (3 mm) → ANTs rigid
  registration to ``delay2`` → z-score with ``clip_values: [-3, 3]``.
* ``config_preprocessing_minimal.yaml`` / ``config_preprocessing_resample_only.yaml``
  — resample-only chains.
* ``config_preprocessing_demo_elastix.yaml`` — same chain on the elastix
  backend (binaries in ``tools/bin/``).
* ``config_preprocessing_n4_resample_registration.yaml`` — N4 bias
  correction → resample → elastix registration.
* ``config_preprocessing_n4_reg_resample_zscore.yaml`` — N4 → registration
  → resample → z-score.
* ``config_preprocessing_dcm2nii_demo.yaml`` /
  ``config_image_preprocessing_dcm2nii.yaml`` — dcm2nii conversion of the
  demo DICOM series via a manifest, then preprocessing.
* ``config_image_preprocessing.yaml`` — registration-only template
  (adapted: modalities ``delay2/delay3/delay5``, fixed ``delay2``).

::

   habit preprocess -c config/preprocessing/config_preprocessing_demo.yaml

Habitat analysis (train / predict)
----------------------------------

``habit get-habitat -c <yaml> -m train|predict`` runs two-step, one-step and
direct-pooling designs. Verified train runs write ``habitat_model.habitatmodel``,
``habitats.parquet``, per-subject ``*_habitats.nrrd`` and clustering plots under
``demo_data/results/cli_yaml/<run_name>/``:

* ``config_habitat_two_step.yaml`` — raw concat → k-means supervoxels → cohort k-means.
* ``config_habitat_two_step_supervoxel_radiomics_train.yaml`` — supervoxel PyRadiomics
  features (bundled ``params_supervoxel_radiomics.yaml`` preset).
* ``config_habitat_two_step_voxel_radiomics_train.yaml`` — voxel PyRadiomics features.
* ``config_habitat_one_step_*_train.yaml`` — one-step designs (per-subject models;
  no cohort ``.habitatmodel`` artefact yet — see stage-5 note in ``cmd_habitat``).
* ``config_habitat_direct_pooling.yaml`` / ``config_habitat_pooling_voxel_radiomics_train.yaml``.

Predict (requires a prior train artefact)::

   habit get-habitat -c config/habitat/config_habitat_two_step_predict.yaml -m predict
   habit get-habitat -c config/habitat/config_habitat_two_step_voxel_radiomics_predict.yaml -m predict

Native v1 document (``version: '1.0'``) via Python API::

   python -c "from habit.recipes import run_from_yaml; run_from_yaml('config/habitat/config_habitat_two_step_v1.yaml', save=True)"

Traditional radiomics and feature extraction
--------------------------------------------

::

   habit radiomics -c config/radiomics/config_traditional_radiomics.yaml
   habit extract -c config/feature_extraction/config_extract_features_demo.yaml
   habit retest -c config/feature_extraction/config_test_retest.yaml
   habit icc -c config/feature_extraction/config_icc_demo.yaml

Machine learning and model comparison
-------------------------------------

Hold-out train writes ``model.habitpipeline``, ``metrics.json`` and
``all_prediction_results.csv`` (for ``habit compare``)::

   habit model -c config/machine_learning/config_machine_learning_clinical.yaml -m train
   habit model -c config/machine_learning/config_machine_learning_predict.yaml -m predict
   habit cv -c config/machine_learning/config_machine_learning_kfold_demo.yaml
   habit compare -c config/model_comparison/config_model_comparison_demo.yaml

Utilities: dice and merge-csv
-----------------------------

``dice`` compares two batches of masks (here the demo masks against
themselves as a self-consistency smoke — Dice = 1.0)::

   habit dice --input1 demo_data/preprocessed/processed_images \
       --input2 demo_data/preprocessed/processed_images \
       --output dice_results.csv

``merge-csv`` joins feature tables horizontally on a shared ID column::

   habit merge-csv demo_data/ml_data/breast_cancer_dataset.csv \
       demo_data/ml_data/clinical_feature.csv \
       -o merged.csv --index-col subject_id

What to read next
-----------------

* :doc:`run_from_yaml` — the programmatic twin of these CLI runs
* :doc:`../configuration/index` — the YAML field reference
* :doc:`image_preprocessing` — the same preprocessing chain from Python
