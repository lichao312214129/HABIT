Traditional radiomics
=====================

Extract whole-ROI PyRadiomics features (no habitat segmentation):

.. code-block:: bash

   habit radiomics --config config/radiomics/config_traditional_radiomics.yaml

**Input**: a folder with ``images/<subject>/<modality>/`` and matching
``masks/<subject>/<modality>/``.

**Output**: feature tables under ``paths.out_dir`` (per-modality and/or combined,
depending on ``export``).

**Configuration**: :doc:`../configuration/radiomics`.

For habitat-aware features use ``habit extract``
(:doc:`extract_features`) instead.
