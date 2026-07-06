Traditional Radiomics
=====================

**Config**: ``feature_types`` includes ``traditional``

**Output**: ``raw_image_radiomics.csv``

Definition
----------

PyRadiomics features extracted from each **preprocessed image modality** within a
tumor ROI. The ROI is a **binary mask** derived from the habitat label map
(voxels with label :math:`\ge 1` → foreground); anatomical ``masks/`` files are
not used. Habitat labels are otherwise ignored—features reflect intratumoral
signal and texture on the raw (multi-delay) images.

Implementation
--------------

- Parameter file: ``params_file_of_non_habitat`` (optional; bundled ``roi`` preset
  → ``habit/resources/radiomics/parameter.yaml`` when omitted)
- Code: ``habit/core/habitat_analysis/habitat_features/builtin_plugins.py``
  (``TraditionalRadiomicsPlugin``) → ``habitat_radiomics.py``

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``{pyradiomics_feature}_of_{modality}``
     - One column per PyRadiomics feature × image modality under the subject folder
   * - (excluded)
     - Columns whose names contain ``diagnostic`` are dropped before export

Feature definitions
-------------------

See `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.
