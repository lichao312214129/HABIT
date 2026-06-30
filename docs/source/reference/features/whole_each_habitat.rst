Whole / Each Habitat Radiomics
==============================

whole_habitat
-------------

Output
~~~~~~

``whole_habitat_radiomics.csv``

Definition
~~~~~~~~~~

PyRadiomics features extracted from the **habitat label map** treated as a single-channel image, masked by the tumor ROI. Parameters: ``parameter_habitat.yaml``.

Feature definitions follow `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

Implementation
~~~~~~~~~~~~~~

``habit/core/habitat_analysis/habitat_features/habitat_radiomics.py``

each_habitat
------------

Output
~~~~~~

``radiomics_of_habitat_{k}.csv`` (one file per habitat label *k*)

Definition
~~~~~~~~~~

For each habitat label, voxels outside that label are masked out on the **original preprocessed image**; PyRadiomics is run on the resulting ROI. Enable via ``feature_types: [each_habitat]``.

Feature definitions follow `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

Implementation
~~~~~~~~~~~~~~

``habit/core/habitat_analysis/habitat_features/habitat_radiomics.py``
