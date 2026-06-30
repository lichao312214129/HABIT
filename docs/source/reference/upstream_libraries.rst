Upstream Dependencies and Documentation Links
===============================================

This page summarizes **official third-party library documentation** and **concept references** for HABIT modules. Implementation details, full hyperparameters, and mathematical definitions are authoritative upstream; the HABIT user manual focuses on **config keys, CLI, and data contracts** (see :doc:`../configuration/index`).

Image preprocessing (``habit preprocess``)
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library / tool
     - Documentation
   * - Registration (``backend: ants``)
     - ANTsPy (``ants.registration``)
     - `Registration <https://antspy.readthedocs.io/en/stable/registration.html>`_ · `ants.registration API <https://antspy.readthedocs.io/en/stable/api/ants.registration.html>`_
   * - Registration (``backend: elastix``)
     - Official **elastix** / **transformix** executables (CLI ``-f`` / ``-m`` / ``-out`` / ``-p`` / ``-tp``, etc.; see `elastix command-line reference <https://elastix.dev/doxygen/commandlinearg.html>`_)
     - `Website / download <https://elastix.dev/>`_ · `Model Zoo (LKEB, by data/task) <https://lkeb.ml/modelzoo/>`_ · `Model Zoo (elastix.dev) <https://elastix.dev/modelzoo.html>`_
   * - Registration (``backend: simpleitk``)
     - SimpleITK (``ImageRegistrationMethod``)
     - `ImageRegistrationMethod <https://simpleitk.readthedocs.io/en/master/registrationOverview.html>`_ · `Fundamental Concepts <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`_
   * - ANTs core
     - ANTs
     - `ANTsX/ANTs <https://github.com/ANTsX/ANTs>`_
   * - Resampling / N4 / majority filter
     - SimpleITK
     - `User Guide <https://simpleitk.readthedocs.io/>`_ · `Fundamental Concepts <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`_ · `ResampleImageFilter <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html>`_
   * - DICOM to NIfTI
     - dcm2niix
     - `rordenlab/dcm2niix <https://github.com/rordenlab/dcm2niix>`_

Habitat segmentation (``habit get-habitat``)
--------------------------------------------

Clustering, dimensionality reduction, and related steps rely heavily on **scikit-learn**; superpixel **SLIC** etc. match the implementing class.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library
     - Documentation
   * - K-Means / GMM / pipeline and metrics
     - scikit-learn
     - `Clustering <https://scikit-learn.org/stable/modules/clustering.html>`_ · `Metrics <https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation>`_
   * - Feature scaling, unsupervised preprocessing (sklearn pipeline concepts)
     - scikit-learn
     - `Preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_

Habitat features and traditional radiomics (``habit extract`` / ``habit radiomics``)
-------------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library
     - Documentation
   * - Radiomics features
     - PyRadiomics
     - `PyRadiomics <https://pyradiomics.readthedocs.io/>`_
   * - Image I/O / some geometry
     - SimpleITK
     - same as preprocessing section

Machine learning (``habit model`` / ``habit cv``)
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library
     - Documentation
   * - Preprocessing, classifiers, pipelines, metrics
     - scikit-learn
     - `User Guide <https://scikit-learn.org/stable/user_guide.html>`_ · `API Reference <https://scikit-learn.org/stable/modules/classes.html>`_
   * - Class imbalance resampling (if enabled)
     - imbalanced-learn (optional)
     - `imbalanced-learn <https://imbalanced-learn.org/stable/>`_
   * - Gradient boosting etc. (if configured)
     - XGBoost / LightGBM, etc.
     - `XGBoost Python <https://xgboost.readthedocs.io/>`_ (as needed)
   * - AutoML (if configured)
     - AutoGluon
     - `AutoGluon Tabular <https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html>`_
   * - SHAP plots (if enabled)
     - shap
     - `SHAP <https://shap.readthedocs.io/>`_

Model comparison (``habit compare``)
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library / method
     - Documentation
   * - ROC / PR / calibration plots
     - matplotlib / in-house plotting (see source)
     - Medical papers often cite **TRIPOD** / **CONSORT-AI** reporting guidelines (not library docs)
   * - DeLong test (if enabled)
     - see implementation and literature
     - refer to HABIT output docs and cited statistical literature

ICC / Test–retest / other
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Purpose
     - Library
     - Documentation
   * - Statistics and matrix operations
     - NumPy / SciPy / pandas
     - `NumPy <https://numpy.org/doc/>`_ · `SciPy <https://docs.scipy.org/doc/scipy/>`_ · `pandas <https://pandas.pydata.org/docs/>`_

ROI delineation (external tools)
--------------------------------

HABIT does not include delineation; common tools:

- `ITK-SNAP <http://www.itksnap.org/>`_
- `3D Slicer <https://www.slicer.org/>`_
