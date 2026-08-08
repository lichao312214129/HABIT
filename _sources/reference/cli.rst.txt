Command reference
=================

Every command is a thin shell over the Python API. Operator guides:
:doc:`../how_to/index`.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Command
     - Purpose
     - Guide
   * - ``habit preprocess``
     - Image preprocessing
     - :doc:`../how_to/preprocess`
   * - ``habit sort-dicom``
     - DICOM sort / rename
     - :doc:`../how_to/preprocess`
   * - ``habit get-habitat``
     - Habitat maps
     - :doc:`../how_to/segment_habitat`
   * - ``habit view``
     - Overlay habitat on image (paths, no YAML)
     - :doc:`../how_to/segment_habitat`
   * - ``habit extract``
     - Habitat features
     - :doc:`../how_to/extract_features`
   * - ``habit radiomics``
     - Whole-ROI radiomics
     - :doc:`../how_to/radiomics`
   * - ``habit model`` / ``habit cv``
     - Train / CV / predict
     - :doc:`../how_to/train_model`
   * - ``habit compare``
     - Compare models
     - :doc:`../how_to/compare_models`
   * - ``habit check-config``
     - Validate YAML
     - :doc:`../how_to/before_you_start`
   * - ``habit migrate-config``
     - v0.1 YAML → v1
     - :doc:`../api/spec`
   * - ``habit list-components``
     - List plugins
     - :doc:`../api/plugins`
   * - ``habit icc`` / ``retest`` / ``dicom-info`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`../how_to/auxiliary_tools`

Help: ``habit --help``, ``habit <cmd> --help``. Templates: ``config/``.
