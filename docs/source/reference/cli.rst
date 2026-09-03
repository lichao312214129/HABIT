Command reference
=================

Every command is a thin shell over the Python API. Walk-throughs:
:doc:`../examples/index`. YAML fields: :doc:`../configuration/index`.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Command
     - Purpose
     - Guide
   * - ``habit preprocess``
     - Image preprocessing
     - :doc:`../examples/image_preprocessing`
   * - ``habit sort-dicom``
     - DICOM sort / rename
     - :doc:`../configuration/dicom_sort`
   * - ``habit get-habitat``
     - Habitat maps
     - :doc:`../examples/two_step_habitat`
   * - ``habit view``
     - Overlay habitat on image (paths, no YAML)
     - :doc:`../examples/visualization`
   * - ``habit extract``
     - Habitat features
     - :doc:`../examples/feature_extraction`
   * - ``habit radiomics``
     - Whole-ROI radiomics
     - :doc:`../configuration/radiomics`
   * - ``habit model`` / ``habit cv``
     - Train / CV / predict
     - :doc:`../configuration/index`
   * - ``habit compare``
     - Compare models
     - :doc:`../configuration/index`
   * - ``habit check-config``
     - Validate YAML
     - :doc:`../tutorial/quickstart`
   * - ``habit copy-demo-config``
     - Materialize bundled demo ``config/`` into a work dir
     - :doc:`../tutorial/quickstart`
   * - ``habit fetch-demo``
     - Download the official preprocessed pack once and print its layout
     - :doc:`../examples/data_from_arrays`
   * - ``habit migrate-config``
     - Older YAML → current document
     - :doc:`../api/spec`
   * - ``habit list-components``
     - List plugins
     - :doc:`../api/plugins`
   * - ``habit icc`` / ``dicom-info`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`../how_to/auxiliary_tools`

Help: ``habit --help``, ``habit <cmd> --help``.
Demo templates: ``habit copy-demo-config`` (or ``from habit.api.demo_config import copy_demo_config``).
Imaging pack: ``habit fetch-demo`` (or ``from habit.datasets import fetch_demo``).
