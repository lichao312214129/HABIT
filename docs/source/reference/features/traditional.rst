Traditional Radiomics
=====================

**Config**: ``feature_types`` includes ``traditional``

**Output**: ``raw_image_radiomics.csv``

Definition
----------

PyRadiomics features are extracted from preprocessed images within the tumor ROI (all delays equal). Habitat labels are ignored; features reflect overall intratumoral signal and texture.

Implementation
--------------

- Parameter file: ``params_file_of_non_habitat`` (typically ``parameter.yaml``)
- Code: ``habit/core/habitat_analysis/habitat_features/habitat_radiomics.py``

Feature definitions
-------------------

See `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

References
----------

Wu J et al., *Radiology* 2018 (`PubMed <https://pubmed.ncbi.nlm.nih.gov/29714680/>`__ · `DOI <https://doi.org/10.1148/radiol.2018172462>`__).
