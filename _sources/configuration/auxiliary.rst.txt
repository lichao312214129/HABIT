Auxiliary Tools and Data Configuration
======================================

Data configuration parameters
-----------------------------

**Example configuration file:**

.. code-block:: yaml

   # Control whether to auto-read the first file in a directory
   auto_select_first_file: true

   images:
     subject1:
       T1: /path/to/subject1/T1/T1.nii.gz
       T2: /path/to/subject1/T2/T2.nii.gz
     subject2:
       T1: /path/to/subject2/T1/T1.nii.gz
       T2: /path/to/subject2/T2/T2.nii.gz

   masks:
     subject1:
       T1: /path/to/subject1/T1/mask_T1.nii.gz
     subject2:
       T1: /path/to/subject2/T1/mask_T1.nii.gz

**auto_select_first_file**: Whether to auto-read the first file in a directory

- **Type**: boolean
- **Default**: ``true``
- **Description**:

  - ``true``: auto-read the first file in the directory (for converted NIfTI files, etc.).
  - ``false``: keep the directory path unchanged (for tasks like dcm2nii that need the whole folder).

**images**: image data paths

- **Type**: dict
- **Required**: yes
- **Default**: none (required)
- **Description**: nested dict; first level is subject ID, second level is image type (key).

**masks**: mask data paths

- **Type**: dict
- **Required**: no
- **Default**: omit for no mask block
- **Description**: same structure as ``images``. Typically used to specify ROI.

ICC analysis configuration (``habit icc``)
------------------------------------------

Corresponds to ``habit.schemas.workflows.icc.ICCConfig``. Example: ``config/auxiliary/config_icc_demo.yaml``.

**input** (required)

- ``type``: ``files`` or ``directories``
- ``file_groups`` (``type: files``): 2D list; each group is file paths for one ICC replicate set; flat list also accepted (each item treated as a single-file group)
- ``dir_list`` (``type: directories``): directory list; feature files collected from each directory

**output** (required)

- ``path``: result JSON output path

**Optional top-level fields**

- ``metrics``: ICC metric list, e.g. ``icc1``, ``icc2``, ``icc3``, ``icc1k``, ``icc2k``, ``icc3k``, ``multi_icc``, ``cohen_kappa``, ``fleiss_kappa``, ``krippendorff``, etc.; default example is ``[icc3]``
- ``selected_features``: limit feature columns for ICC; ``null`` means all
- ``full_results`` (bool, default ``false``): whether to output full detail
- ``processes`` (int, optional): parallel process count
- ``debug`` (bool, default ``false``)

Test-Retest configuration (``habit retest``)
--------------------------------------------

This section documents **Test-Retest reproducibility** configuration. Example: ``config/auxiliary/config_test_retest.yaml``. Command usage: :doc:`../reference/auxiliary`.

**Required fields**

- ``test_habitat_table``: habitat feature table from test scan (CSV/Excel)
- ``retest_habitat_table``: habitat feature table from retest scan
- ``input_dir``: retest-group NRRD habitat map directory (for mapping/realignment)
- ``out_dir``: analysis output directory

**Optional fields**

- ``features``: feature columns for similarity; ``null`` means all
- ``similarity_method`` (default ``pearson``): ``pearson``, ``spearman``,
  ``kendall`` (alias of spearman), ``euclidean``, ``cosine``,
  ``manhattan``, ``chebyshev``. Assignment is **Hungarian** on
  cohort-z-scored habitat medians (not greedy argmax). See
  :doc:`../examples/habitat_label_match`.
- ``processes`` (default ``4``)
- ``debug`` (default ``false``)

Intermediate NRRD remapping outputs are written under ``out_dir``.

Traditional radiomics CLI configuration (``habit radiomics``)
-------------------------------------------------------------

Moved to :doc:`radiomics`. Example:
``config/radiomics/config_traditional_radiomics.yaml``.

Repository configuration template index
---------------------------------------

Scenario catalog: :doc:`recipe_catalog`. The ``config/`` directory is organized
by function; copy and modify templates directly:

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - Path
     - Purpose
   * - ``config/preprocessing/``
     - Image preprocessing and ``files_preprocessing.yaml`` subject lists
   * - ``config/dicom_sort/``
     - DICOM sort-only (``sort-dicom``)
   * - ``config/habitat/``
     - Habitat train/predict (two_step / one_step / direct_pooling) and ``file_habitat.yaml``
   * - ``config/feature_extraction/``
     - ``habit extract`` habitat feature extraction
   * - ``config/radiomics/``
     - PyRadiomics parameters and ``habit radiomics`` top-level config
   * - ``config/machine_learning/``
     - Standard train/predict, K-fold, clinical/radiomics examples
   * - ``config/model_comparison/``
     - Multi-model ROC/DCA/DeLong comparison
   * - ``config/auxiliary/``
     - ICC, Test-Retest, and other auxiliary analyses

Configuration file validation
-----------------------------

HABIT provides configuration validation to ensure parameter correctness.

**Validation rules:**

1. **Required parameter check**: verify all required parameters are provided
2. **Type check**: verify parameter types are correct
3. **Range check**: verify values are within valid ranges
4. **Dependency check**: verify parameter dependencies are satisfied

**Validation example:**

.. code-block:: python

   from habit.schemas.workflows.habitat import FeatureExtractionConfig

   # Workflow commands validate YAML via Pydantic models, e.g.:
   cfg = FeatureExtractionConfig.model_validate(yaml_dict)

FAQ
---

**Q1: How do I create a configuration file?**

A: You can:

1. Copy an example YAML from ``config/`` (see :doc:`recipe_catalog`) and edit paths
2. Refer to field descriptions on the matching configuration page
3. Create YAML from scratch only if needed (easy to miss required fields)

**Q2: How do I debug a configuration file?**

A: You can:

1. Enable verbose logging with ``debug`` mode
2. Check YAML syntax
3. Add parameters incrementally to locate issues
4. Review error messages
