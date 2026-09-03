Configuration recipe catalog
============================

Copy a YAML template from the repository ``config/`` tree, edit the ``#%%====``
path blocks, then run the matching ``habit`` command. Relative paths in YAML
resolve from the **config file directory**, not the shell working directory.

.. note::

   Source checkouts keep ``config/`` under the repository root (sibling of
   the ``habit/`` package).

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
   * - ICC
     - ``config/auxiliary/``
     - ``habit icc -c …``

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
   * - ICC demo
     - ``config/auxiliary/config_icc_demo.yaml``

Field reference pages
---------------------

- :doc:`preprocessing`
- :doc:`dicom_sort`
- :doc:`habitat`
- :doc:`feature_extraction`
- :doc:`radiomics`
- :doc:`auxiliary`
