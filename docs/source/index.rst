HABIT: Biomedical Imaging Toolkit
==================================

HABIT (Habitat Analysis: Biomedical Imaging Toolkit) is a comprehensive Python-based tumor "habitat" analysis toolkit designed for medical imaging research.

Core Workflow
-------------

HABIT identifies and characterizes tumor sub-regions with different imaging phenotypes, known as "habitats".

**Image → Voxel Features → Supervoxels (Optional) → Habitats → Habitat Features → Prediction Model (Optional)**

Key Features
------------

* **Image Preprocessing**: DICOM conversion, resampling, registration, normalization
* **Feature Extraction**: Voxel-level, supervoxel-level, habitat-level features
* **Clustering Analysis**: Multiple clustering algorithms (K-Means, GMM, Hierarchical, etc.)
* **Habitat Analysis**: MSI, ITH, radiomics features
* **Machine Learning**: Model training, cross-validation, prediction

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   api/modules
   algorithms
   tutorials
   development
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

GitHub
------

* https://github.com/lichao312214129/HABIT
