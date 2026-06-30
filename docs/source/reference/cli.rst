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
     - :doc:`../configuration/preprocessing`
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
     - :doc:`../how_to/compare_models`
   * - ``habit gui``
     - Web GUI
     - :doc:`../gui/index`
   * - ``habit radiomics``
     - Traditional radiomics
     - :doc:`../configuration/auxiliary`
   * - ``habit dicom-info`` / ``icc`` / ``retest`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`auxiliary`

Global flags: ``--config`` / ``-c`` , ``--help`` , ``--version`` , ``--debug`` (mainly ``get-habitat`` ).

Templates: ``config/`` · fields: :doc:`../configuration/index` .
