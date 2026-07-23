Configuration recipe catalog
============================

Copy a YAML template from the repository ``config/`` tree, edit the ``#%%====``
path blocks, then run the matching ``habit`` command. Relative paths in YAML
resolve from the **config file directory**, not the shell working directory.

.. note::

   The lightweight Windows ZIP ships ``config/`` next to the installers. Source
   checkouts keep the same layout under the repository root.

By workflow
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Workflow
     - Template directory
     - Typical command
   * - Preprocessing
     - ``config/preprocessing/``
     - ``habit preprocess -c …``
   * - DICOM sort
     - ``config/dicom_sort/``
     - ``habit sort-dicom -c …``
   * - Habitat segmentation
     - ``config/habitat/``
     - ``habit get-habitat -c …``
   * - Habitat feature extraction
     - ``config/feature_extraction/``
     - ``habit extract -c …``
   * - Traditional radiomics
     - ``config/radiomics/``
     - ``habit radiomics -c …``
   * - Machine learning
     - ``config/machine_learning/``
     - ``habit model -c …`` / ``habit cv -c …``
   * - Model comparison
     - ``config/model_comparison/``
     - ``habit compare -c …``
   * - ICC / test-retest
     - ``config/auxiliary/``
     - ``habit icc -c …`` / ``habit retest -c …``

Starter demos
-------------

Use these when you want a short path through the demo data:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Goal
     - Suggested YAML
   * - Resample-only preprocess
     - ``config/preprocessing/config_preprocessing_demo.yaml``
   * - Habitat two-step train
     - ``config/habitat/config_habitat_two_step.yaml``
   * - Extract habitat features
     - ``config/feature_extraction/config_extract_features_demo.yaml``
   * - Traditional ROI radiomics
     - ``config/radiomics/config_traditional_radiomics.yaml``
   * - Train a radiomics ML model
     - ``config/machine_learning/config_machine_learning_radiomics.yaml``
   * - K-fold CV demo
     - ``config/machine_learning/config_machine_learning_kfold_demo.yaml``
   * - Compare models
     - ``config/model_comparison/config_model_comparison_demo.yaml``
   * - ICC demo
     - ``config/auxiliary/config_icc_demo.yaml``

Field reference pages
---------------------

- :doc:`preprocessing`
- :doc:`dicom_sort`
- :doc:`habitat`
- :doc:`feature_extraction`
- :doc:`radiomics`
- :doc:`machine_learning`
- :doc:`model_comparison`
- :doc:`auxiliary`
