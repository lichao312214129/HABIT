Preprocessing Configuration
===========================

Preprocessing configuration parameters
--------------------------------------

This section documents **preprocessing** configuration (``PreprocessingConfig``). Top-level key ``preprocessing`` holds step blocks; keys must match names registered in ``PreprocessorFactory``; YAML order is execution order.

DICOM **sort-only** uses ``habit sort-dicom`` with a separate config; see
:doc:`dicom_sort`. Operations: :doc:`../how_to/preprocess`.

**Example configuration file:**

.. code-block:: yaml

   data_dir: ./config/preprocessing/files_preprocessing.yaml
   out_dir: ./preprocessed
   auto_select_first_file: true

   preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true
       anonymize: false

     n4_correction:
       images: [delay2, delay3, delay5]
       num_fitting_levels: 4

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]
       img_mode: bilinear

     registration:
       images: [delay2, delay3, delay5]
       fixed_image: delay2
       type_of_transform: SyNRA
       metric: MI
       use_mask: false

     histogram_standardization:
       images: [delay2, delay3, delay5]
       target_min: 0.0
       target_max: 100.0

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false
       clip_values: [-3, 3]

     adaptive_histogram_equalization:
       images: [delay2, delay3, delay5]
       alpha: 0.3
       beta: 0.3
       radius: 5

   save_options:
     save_intermediate: true
     intermediate_steps: [dcm2nii, n4_correction, resample]

   processes: 2
   random_state: 42

**Top level (``PreprocessingConfig``)**

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Description
   * - ``data_dir`` / ``out_dir``
     - none (required)
     - Relative paths are resolved against the directory containing this YAML
   * - ``preprocessing``
     - ``{}``
     - Step name → config dict; keys must be registered preprocessor names
   * - ``processes``
     - ``1``
     - Must be ``>= 1``; effective parallelism is ``min(config, CPU cores - 2)``, at least 1
   * - ``random_state``
     - ``42``
     - ``numpy.random.seed`` at ``BatchProcessor.run()`` entry
   * - ``auto_select_first_file``
     - ``true``
     - Whether to auto-select the first file when multiple files exist in a directory
   * - ``preprocessing_input_layout``
     - ``habit_default``
     - Currently only ``habit_default`` directory layout is supported
   * - ``save_options``
     - see table below
     - Intermediate result persistence options

**Preprocessing**: each step shares the common field ``images`` (required, non-empty list).

**dcm2nii**: DICOM conversion

- ``images``: modality key list (**required**).
- ``dcm2niix_path``: executable file or directory; **optional** — if omitted, ``dcm2niix`` is searched on ``PATH``.
- Other common options: ``compress``, ``anonymize``, ``filename_format``, ``adjacent_dicoms``, ``ignore_derived``, ``crop_images``, ``generate_json``, ``verbose``, ``batch_mode``, ``merge_slices``, ``single_file_mode``, etc. (see source code).

**n4_correction**

- ``images`` (required); ``num_fitting_levels`` (default 4); ``num_iterations``; ``convergence_threshold``; ``shrink_factor``; optional ``mask_keys``.

**resample**

- ``images`` (required); ``target_spacing`` [x,y,z] mm; ``img_mode`` (image interpolation, default ``bilinear``); ``padding_mode``; ``align_corners``. Mask resampling uses nearest neighbor.

**registration**

- ``images`` (required, must include ``fixed_image``); ``fixed_image`` (required); floating sequences are all keys in ``images`` except ``fixed_image``. **Do not use** YAML field ``moving_images`` (not read by the implementation and may be passed as an extra keyword to ANTs).
- ``backend`` (optional): ``ants`` (default), ``simpleitk``, ``elastix`` (calls official elastix / transformix executables; optional ``elastix_path`` / ``transformix_path``); see the ``registration`` field notes on this page.
- ``type_of_transform``, ``metric``, ``optimizer``, etc. apply to **ants / simpleitk**; the ``elastix`` backend does not use these keys to drive registration. **All optional values** (ANTS path) are listed in this document; common examples include ``Rigid``, ``Affine``, ``SyN``, ``SyNRA``, etc.
- ``use_mask``; optional ``mask_keys``; ``replace_by_fixed_image_mask``.
- **elastix-specific** (``elastix`` backend):

  - ``elastix_parameter_files``: parameter template ``.txt`` files; choose from `LKEB elastix Model Zoo <https://lkeb.ml/modelzoo/>`_ by data type and registration task.
  - ``elastix_parameter_overrides``: dict overriding parameter values.
  - ``elastix_path``, ``transformix_path``, ``elastix_threads``.
- Additional ANTs-allowed parameters may be passed via extra keys (use with care).

**histogram_standardization**

- ``images`` (required); ``percentiles``; ``target_min`` / ``target_max``; optional ``mask_key`` (for histogram statistics).

**zscore_normalization**

- ``images`` (required); ``only_inmask``; ``mask_key`` (when ``only_inmask`` is true, must exist in ``data``, e.g. shared mask key or ``mask_<modality>``); ``clip_values``.

**adaptive_histogram_equalization**

- ``images`` (required); ``alpha``, ``beta`` ∈ [0,1]; ``radius`` as int or (x,y,z).

**save_options** (``SaveOptionsConfig``)

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Description
   * - ``save_intermediate``
     - ``false``
     - Whether to write intermediate directories
   * - ``intermediate_steps``
     - ``[]``
     - When non-empty, only listed steps write intermediate results; **empty list** with ``save_intermediate: true`` writes every step

DICOM sort configuration (``habit sort-dicom``)
----------------------------------------------

Field reference moved to :doc:`dicom_sort`. Template:
``config/dicom_sort/config_sort_dicom.yaml``.
