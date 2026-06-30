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

Corresponds to ``habit.core.machine_learning.feature_selectors.icc.config.ICCConfig``. Example: ``config/auxiliary/config_icc_demo.yaml``.

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
- ``similarity_method`` (default ``pearson``): ``pearson``, ``spearman``, ``kendall``, ``euclidean``, ``cosine``, ``manhattan``, ``chebyshev``
- ``output_dir``: remapped NRRD and other intermediate output directory (used alongside ``out_dir`` in example YAML)
- ``processes`` (default ``4``)
- ``debug`` (default ``false``)

Traditional radiomics CLI configuration (``habit radiomics``)
------------------------------------------------------------

This section documents **traditional radiomics** configuration. Example: ``config/radiomics/config_traditional_radiomics.yaml``.

**paths** (required)

- ``params_file``: PyRadiomics parameter YAML (`PyRadiomics documentation <https://pyradiomics.readthedocs.io/>`_)
- ``images_folder``: root directory containing ``images/`` and ``masks/`` subfolders
- ``out_dir``: feature output directory

**processing**

- ``n_processes`` (default 2)
- ``save_every_n_files`` (default 5): save intermediate results every N files
- ``process_image_types``: list of sequence/type names to process; ``null`` means all
- ``target_labels``: label list extracted from mask (default ``[1]``) for binary foreground

**export**

- ``export_by_image_type``, ``export_combined``, ``export_format`` (``csv`` | ``json`` | ``pickle``), ``add_timestamp``

**logging**

- ``level`` (DEBUG/INFO/…), ``console_output``, ``file_output``

**Backward-compatible top-level fields** (deprecated, equivalent to nested): ``params_file``, ``images_folder``, ``out_dir``, ``n_processes``.

Repository configuration template index
---------------------------------------

Full scenario descriptions and test script paths: **`config/README_CONFIG.md`** (repository root, sibling to the ``habit/`` package).
The ``config/`` directory is organized by function; copy and modify directly:

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

   from habit.core.common.configs.loader import load_config

   # Load and validate configuration
   config = load_config('./config.yaml')

   # Raises an exception if configuration is invalid
   # ValueError: Missing required parameter: data_dir

FAQ
---

**Q1: How do I create a configuration file?**

A: You can:

1. Copy the example YAML for your scenario from the repository root ``config/`` and modify it (scenario list in ``config/README_CONFIG.md``)
2. Refer to field descriptions in each section of this document and add/remove parameters on the copy
3. Create YAML from scratch (not recommended; easy to miss required fields)

**Q2: How do I debug a configuration file?**

A: You can:

1. Enable verbose logging with ``debug`` mode
2. Check YAML syntax
3. Add parameters incrementally to locate issues
4. Review error messages
