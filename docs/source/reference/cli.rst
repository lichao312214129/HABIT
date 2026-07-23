Command reference
=================

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
   * - ``habit model`` / ``habit cv``
     - ML / K-fold
     - :doc:`../how_to/train_model`
   * - ``habit compare``
     - Model comparison
     - :doc:`../how_to/compare_models` · :doc:`../configuration/model_comparison`
   * - ``habit gui``
     - Web GUI (under development)
     - :doc:`../gui/index`
   * - ``habit radiomics``
     - Traditional radiomics
     - :doc:`../how_to/radiomics` · :doc:`../configuration/radiomics`
   * - ``habit dicom-info`` / ``icc`` / ``retest`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`../how_to/auxiliary_tools` · :doc:`auxiliary`

Global flags: ``--help`` , ``--version`` . ``--config`` / ``-c`` applies to pipeline commands that take a YAML (not ``dice``, ``dicom-info``, ``merge-csv``, ``gui``). ``--debug`` and ``--resume`` apply to ``habit get-habitat`` only.

Templates: ``config/`` · fields: :doc:`../configuration/index` .
