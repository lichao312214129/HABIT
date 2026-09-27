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
     - :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`
   * - ``habit view``
     - Overlay habitat on image (paths, no YAML)
     - :doc:`/auto_examples/04_quantifying_habitats/plot_01_volume_fractions`
   * - ``habit extract``
     - Habitat features
     - :doc:`/auto_examples/04_quantifying_habitats/index`
   * - ``habit radiomics``
     - Whole-ROI radiomics
     - :doc:`../configuration/radiomics`
   * - ``habit check-config``
     - Validate YAML
     - :doc:`/auto_quickstart/plot_quickstart_yaml`
   * - ``habit copy-demo-config``
     - Materialize bundled demo ``config/`` into a work dir
     - :doc:`/auto_quickstart/plot_quickstart_yaml`
   * - ``habit fetch-demo``
     - Download the official preprocessed pack once and print its layout
     - :doc:`/auto_examples/06_manipulating_images/index`
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
Demo templates: ``habit copy-demo-config`` (or ``from habit.utils.demo_config_utils import copy_demo_config``).
Imaging pack: ``habit fetch-demo`` (or ``from habit.datasets import fetch_demo``).
