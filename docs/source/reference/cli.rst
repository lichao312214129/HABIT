Command reference
=================

HABIT v1.0 provides 16 subcommands. Every one is a thin shell over the
Python API (:doc:`../api/index`): a command reads its YAML, builds the same
spec objects, and calls the same recipes.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Command
     - Purpose
     - Guide
   * - ``habit preprocess``
     - Image preprocessing
     - :doc:`../how_to/preprocess`
   * - ``habit sort-dicom``
     - DICOM sort / rename
     - :doc:`../how_to/preprocess` · :doc:`../configuration/dicom_sort`
   * - ``habit get-habitat``
     - Habitat segmentation
     - :doc:`../how_to/segment_habitat`
   * - ``habit extract``
     - Feature extraction
     - :doc:`../how_to/extract_features`
   * - ``habit radiomics``
     - Traditional radiomics
     - :doc:`../how_to/radiomics` · :doc:`../configuration/radiomics`
   * - ``habit model`` / ``habit cv``
     - ML / K-fold
     - :doc:`../how_to/train_model`
   * - ``habit compare``
     - Model comparison
     - :doc:`../how_to/compare_models` · :doc:`../configuration/model_comparison`
   * - ``habit check-config``
     - Validate a YAML without running
     - :doc:`../configuration/index`
   * - ``habit migrate-config``
     - Upgrade a v0.1 YAML to the v1 layout
     - :doc:`../api/spec` · :doc:`../configuration/index`
   * - ``habit icc`` / ``habit retest``
     - Reproducibility analysis
     - :doc:`../how_to/auxiliary_tools` · :doc:`auxiliary`
   * - ``habit dicom-info`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`../how_to/auxiliary_tools` · :doc:`auxiliary`
   * - ``habit gui``
     - Web GUI (under development)
     - :doc:`../gui/index`

Global flags: ``--help`` , ``--version`` . ``--config`` / ``-c`` applies to pipeline commands that take a YAML (not ``dice``, ``dicom-info``, ``merge-csv``, ``gui``). ``--debug`` and ``--resume`` apply to ``habit get-habitat`` only.

Templates: ``config/`` · fields: :doc:`../configuration/index` .
