Command reference
=================

Every command is a thin shell over the Python API. YAML fields:
:doc:`../configuration/index`. Habitat maps and extract walk-throughs
are Python-first in the Guide.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Command
     - Purpose
     - Guide
   * - ``habit preprocess``
     - Image preprocessing
     - :doc:`../configuration/preprocessing`
   * - ``habit sort-dicom``
     - DICOM sort / rename
     - :doc:`../configuration/dicom_sort`
   * - ``habit get-habitat``
     - Habitat maps
     - :doc:`../examples/habitat_recipes`
   * - ``habit view``
     - Overlay habitat on image (paths, no YAML)
     - :doc:`../examples/visualization`
   * - ``habit extract``
     - Habitat features
     - :doc:`../examples/feature_extraction`
   * - ``habit radiomics``
     - Whole-ROI radiomics
     - :doc:`../configuration/radiomics`
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
     - :doc:`../configuration/auxiliary`

Help: ``habit --help``, ``habit <cmd> --help``.
Demo templates: ``habit copy-demo-config`` (or ``from habit.api.demo_config import copy_demo_config``).
Imaging pack: ``habit fetch-demo`` (or ``from habit.datasets import fetch_demo``).
