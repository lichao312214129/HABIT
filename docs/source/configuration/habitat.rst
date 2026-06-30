Habitat Segmentation Configuration
==================================

Habitat Analysis Configuration Parameters
-----------------------------------------

This section covers **habitat analysis** configuration. CLI: ``habit get-habitat -c <yaml>``. Demo training: ``config/habitat/config_habitat_two_step.yaml``; prediction: ``config/habitat/config_habitat_two_step_predict.yaml``; SLIC supervoxel example: ``config/habitat/config_habitat_two_step_supervoxel_slic.yaml`` (parameter details in **"HabitatSegmentation.supervoxel — SLIC Superpixel Configuration"** below).

**Example configuration file:**

.. code-block:: yaml

   run_mode: train
   pipeline_path: ./results/habitat_pipeline.pkl
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/train

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()

     preprocessing_for_subject_level:
       methods:
         - method: winsorize
           winsor_limits: [0.05, 0.05]
           global_normalize: true
         - method: minmax
           global_normalize: true

     preprocessing_for_group_level:
       methods:
         - method: binning
           n_bins: 10
           bin_strategy: uniform
           global_normalize: false

   HabitatSegmentation:
     clustering_mode: two_step

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10

     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       fixed_n_clusters: null
       random_state: 42
       max_iter: 300
       n_init: 10

   processes: 2
   cap_processes_to_gpu_pool: true
   individual_subject_timeout_sec: 900
   individual_subject_spawn_timeout_sec: 120
   resume: true
   strict_checkpoint_hash: false
   checkpoint_dir: null
   force_rerun_subjects: []
   retry_failed_subjects: false
   individual_subject_auto_retry_rounds: 2
   individual_subject_parallel_mode: persistent
   persistent_worker_max_consecutive_failures: 1
   persistent_worker_recycle_after_tasks: 0
   clear_checkpoint_on_success: false
   plot_curves: true
   save_images: true
   save_results_csv: true
   random_state: 42
   verbose: true
   debug: false

**run_mode**: Run mode

- **Type**: string
- **Required**: no
- **Default**: ``train``
- **Allowed values**: ``train``, ``predict``
- **Description**: ``train`` trains a new model; ``predict`` runs inference with a pretrained pipeline.
- **Example**: ``train``

**pipeline_path**: Pipeline file path

- **Type**: string
- **Required**: no (required in ``predict`` mode)
- **Default**: ``null``
- **Description**: Path to the trained pipeline file.
- **Example**: ``./results/habitat_pipeline.pkl``

**data_dir**/**out_dir** (habitat analysis top level)

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: ``data_dir`` may be a directory or a manifest such as ``file_habitat.yaml``; ``out_dir`` is the default parent directory for results and checkpoints.

**FeatureConstruction**: Feature extraction settings

- **Type**: object
- **Required**: required in ``train``; may be omitted in ``predict``
- **Default**: ``null``
- **Description**: If unset, validation rejects the config or errors according to run mode. Sub-blocks ``voxel_level`` / ``supervoxel_level`` / ``preprocessing_*`` are documented below.

**voxel_level**: Voxel-level feature extraction

- ``method``: Feature extraction method expression

  - **Type**: string
  - **Required**: yes
  - **Default**: none (required)
  - **Description**: Supports functional syntax to combine multiple extractors.
  - **Expression conventions** (shared by ``voxel_level`` / ``supervoxel_level``):

    - When extracting per modality (``raw``, ``voxel_radiomics``, ``supervoxel_radiomics``, etc.), **even for a single modality** wrap with an outer combiner such as ``concat(...)``; inner expressions are per-modality sub-expressions; the outer layer merges results across modalities.
    - Comma-separated tokens inside parentheses are **parameter name placeholders** or **image modality names** (matching ``images/<subject>/<modality>/`` subdirectories), **not** Python keyword arguments; actual paths and numeric values go in a sibling ``params`` dict, or are merged from ``params`` by ``resolve_*_step_params``.
    - Recommended radiomics forms: ``concat(voxel_radiomics(T2, params_file, kernelRadius))``, ``concat(supervoxel_radiomics(T2, params_file))``; ``params_file`` and similar keys may appear only in ``params`` without being listed in the expression string (see ``params`` below).

- ``params``: Voxel-level extractor parameter dictionary

  - **Type**: dict
  - **Required**: no
  - **Default**: ``{}``
  - **Description**: Key-value pairs passed to extractors referenced in ``method`` (e.g. ``params_file`` for ``voxel_radiomics`` may live in ``params`` depending on resolver behavior). Omit the entire ``params`` block when there are no extra parameters (defaults to empty dict).
  - **Available methods and parameters**:

**raw(image_name)**:

      - **Description**: Extract raw image voxel values (most basic feature)
      - **Parameters**: none
      - **Example**: ``raw(delay2)``

**concat(...)**:

      - **Description**: Concatenate multiple feature vectors
      - **Parameters**: accepts multiple feature extraction expressions
      - **Example**: ``concat(raw(delay2), raw(delay3), raw(delay5))``

**kinetic(...)**:

      - **Description**: Extract kinetic features (wash-in/wash-out slopes, etc.)
      - **Parameters**:

        - ``timestamps`` (str, required): path to timestamps file
        - accepts multiple ``raw(image_name)`` expressions

      - **Example**: ``kinetic(raw(LAP), raw(PVP), raw(delay_3min), timestamps=...)``
      - **Extracted features**:

        - ``wash_in_slope``: wash-in slope
        - ``wash_out_slope_lap_pvp``: wash-out slope from LAP to PVP
        - ``wash_out_slope_pvp_dp``: wash-out slope from PVP to delay phase

**local_entropy(...)**:

      - **Description**: Compute local entropy (local texture complexity)
      - **Parameters**:

        - ``kernel_size`` (int, default: ``3``): local neighborhood size
        - ``bins`` (int, default: ``32``): histogram bin count

      - **Example**: ``local_entropy(raw(delay2), kernel_size=5, bins=32)``

**voxel_radiomics(...)**:

      - **Description**: Extract voxel-level radiomics features
      - **Parameters**:

        - ``params_file`` (str, required): PyRadiomics parameter file path
        - ``kernelRadius`` (int, CT habitat recommended: ``3``; code default ``1`` if unset): local neighborhood radius (1=3×3×3, 3=7×7×7)
        - ``voxelBatch`` (int, default: ``1000``): voxel batch size; ``-1`` processes all ROI voxels at once (native PyRadiomics, no batching). Positive values limit memory (GPU or large ROI: ``512``–``1000`` recommended)
        - ``useTorchRadiomics`` (str, default: ``auto``): ``auto`` uses TorchRadiomics when torch is installed and CUDA is available, otherwise CPU PyRadiomics; ``true`` forces torch; ``false`` always CPU
        - ``torchDevice`` (str, default: ``auto``): single GPU device when ``torchGpus`` is unset
        - ``torchGpus`` (list/int/str): allowed GPU indices, e.g. ``[0, 1, 2]`` or ``"0,1,2"``; overrides ``torchDevice`` when set
        - ``torchGpuCount`` (int, optional): use first N GPUs from ``torchGpus``
        - ``torchDtype`` (str, default: ``float32``): Torch compute dtype (``float32`` or ``float64``; ``float64`` closer to CPU PyRadiomics)

      - **Voxel GLCM note**: Use ``config/radiomics/params_voxel_radiomics.yaml`` (explicit list of 21 stable GLCM features).
        If ``params_file`` lists only ``glcm:`` without feature names, PyRadiomics computes all 24 GLCM features; in
        small neighborhoods with many uniform voxels, GLCM degenerates to 1×1 matrices and **MCC / Imc1 / Imc2**
        feature values or mutual information can crash or produce NaN on CUDA/MKL. HABIT auto-replaces with the
        21 stable features and logs a warning when GLCM is unrestricted; if features are explicitly listed in
        ``params_file``, user configuration is respected.

      - **CT voxel texture (R3B12)**: For CT habitat voxel radiomics, literature recommends ``kernelRadius: 3`` and ``binWidth: 12`` HU
        (R3B12 configuration; better repeatability and robustness to kernel/binning than R1B25). Parameter file:
        ``params_voxel_radiomics.yaml``; neighborhood radius is set in habitat config ``voxel_level.params``.
        Reference: Petersen A, et al. Identification of Precise 3D CT Radiomics for Habitat Computation
        by Machine Learning in Cancer. *Radiol Artif Intell*. 2024;6(2):e230118.
        https://doi.org/10.1148/ryai.230118

      - **Example**: ``concat(voxel_radiomics(T2, params_file, kernelRadius))`` with ``params_file``, ``kernelRadius``, etc. in ``params``

  - **Full examples**:

    .. code-block:: yaml

       # Simple concatenation of raw images
       voxel_level:
         method: concat(raw(delay2), raw(delay3), raw(delay5))

       # Kinetic features
       voxel_level:
         method: kinetic(raw(LAP), raw(PVP), raw(delay_3min))
         params:
           timestamps: ./timestamps.txt
       
       # Combine local entropy and raw values
       voxel_level:
         method: concat(raw(delay2), local_entropy(raw(delay2)))
         params:
           kernel_size: 5
           bins: 32

       # Voxel-level radiomics (texture features, slower; single modality still needs concat)
       voxel_level:
         method: concat(voxel_radiomics(T2, params_file, kernelRadius))
         params:
           params_file: ./config/radiomics/params_voxel_radiomics.yaml
           kernelRadius: 3
           voxelBatch: 1000
           useTorchRadiomics: auto
           # torchGpus: [0, 1]
           # torchGpuCount: 2

- ``params``: Global parameters

  - **Type**: dict
  - **Required**: no
  - **Default**: ``{}``
  - **Description**: Shared parameters for all extractors. ``voxel_radiomics``-specific keys (``voxelBatch``, ``useTorchRadiomics``, etc.) belong in ``params`` and **need not** appear in the ``method`` expression string; keys not listed in the expression are auto-merged and forwarded.
  - **Common parameters**:

    - ``timestamps`` (str): timestamps file path (for kinetic)
    - ``kernel_size`` (int): local neighborhood size (for local_entropy)
    - ``bins`` (int): histogram bin count (for local_entropy)
    - ``params_file`` (str): PyRadiomics parameter file (for voxel_radiomics)
    - ``kernelRadius`` (int): voxel radiomics neighborhood radius (for voxel_radiomics)
    - ``voxelBatch`` (int): voxel radiomics batch size (for voxel_radiomics; default ``1000``; ``-1`` = no batching)
    - ``useTorchRadiomics`` (str): TorchRadiomics acceleration (``auto`` / ``true`` / ``false``)
    - ``torchDevice`` (str): single GPU device (when ``torchGpus`` unset)
    - ``torchGpus`` (list/int/str): allowed GPU list
    - ``torchGpuCount`` (int): cap on GPUs actually used
    - ``torchDtype`` (str): Torch dtype (voxel_radiomics torch backend)

**supervoxel_level**: Superpixel-level feature extraction (optional)

- **Block default**: ``null`` (omit = no supervoxel block; ``two_step`` training usually requires it)

- ``supervoxel_file_keyword``: Superpixel file glob pattern

  - **Type**: string
  - **Required**: no (when ``supervoxel_level`` is configured)
  - **Default**: ``*_supervoxel.nrrd``
  - **Description**: Matches existing supervoxel segmentation files (from two_step mode).
  - **Example**: ``"*_supervoxel.nrrd"``

- ``method``: Feature aggregation/extraction method

  - **Type**: string
  - **Required**: no (recommended when ``supervoxel_level`` is configured)
  - **Default**: ``mean_voxel_features()``
  - **Description**: Defines how voxel features aggregate to supervoxels, or direct supervoxel extraction. For multi-modality extractors such as ``supervoxel_radiomics``, expression conventions match ``voxel_level`` (single modality still needs outer ``concat(...)`` etc.; see "Expression conventions" above).
  - **Available methods and parameters**:

**mean_voxel_features()**:

      - **Description**: Mean of voxel features within each supervoxel (most common)
      - **Parameters**: none
      - **Use case**: Aggregate voxel-level features (from ``voxel_level``) to supervoxel level
      - **Example**: ``mean_voxel_features()``

**concat(supervoxel_radiomics(<modality>, params_file), ...)**:

      - **Description**: Per supervoxel label, extract **whole-ROI** radiomics texture (not voxel kernel neighborhoods). Must sit inside ``concat(...)`` (or another outer combiner); single-modality example: ``concat(supervoxel_radiomics(T2, params_file))``.
      - **Discretization**: One PyRadiomics ``_applyBinning`` on the union mask of all supervoxels (``sv_map > 0``), then per-label ``cMatrices``
      - **Matrix backend**: ``useSupervoxelCext`` default ``auto``: use C extension batch matrix build when ``supervoxel_cext`` is compiled (``pip install -e .``); otherwise fallback Torch/PyRadiomics stacked matrix path. ``false`` **forces** Torch/PyRadiomics stacked matrices (``matrix_backend=torch_cmatrices``) even if C extension exists
      - **Feature backend**: When ``useTorchRadiomics`` resolves to torch, TorchRadiomics (GPU/CPU torch); else CPU PyRadiomics (same semantics)
      - **Parameters** (in ``FeatureConstruction.supervoxel_level.params``; may inherit torch keys from ``voxel_level.params``):

        - ``params_file`` (str, required): PyRadiomics parameter YAML (featureClass / setting only); recommend ``params_file`` placeholder in ``method`` sub-expression, actual path in ``supervoxel_level.params`` (or only in ``params`` merged by resolver)
        - ``supervoxelBatch`` (int): batch group size, default ``64`` (not kernel radius)
        - ``supervoxelUnionBboxCrop`` (bool): crop to union bbox, default ``true``
        - ``useSupervoxelCext`` (str | bool): ``auto`` / ``true`` / ``false``, default ``auto``; must be in ``supervoxel_level.params`` (not in ``params_file``)
        - ``useTorchRadiomics`` (str): ``auto`` / ``true`` / ``false``
        - ``torchGpus`` / ``torchGpuCount`` / ``torchDevice`` / ``torchDtype``: same as voxel level

      - **Note**: ``kernelRadius`` is for ``voxel_radiomics`` only; ``supervoxel_radiomics`` does not use it
      - **Use case**: Texture radiomics directly from supervoxel regions without ``voxel_level`` features
      - **Example**: ``concat(supervoxel_radiomics(T2, params_file))`` with ``params.params_file: ./config/radiomics/params_supervoxel_radiomics.yaml``

  - **Method comparison**:

    - ``mean_voxel_features()``: depends on ``voxel_level`` features; fast; suitable for most cases
    - ``concat(supervoxel_radiomics(...), ...)``: standalone ROI radiomics; union-mask bin once + per-label extract; feature values **differ** from legacy per-label ``execute`` (per-label bin)

  - **Full examples**:

    .. code-block:: yaml

       # Scenario 1: aggregate voxel features (recommended)
       supervoxel_level:
         supervoxel_file_keyword: '*_supervoxel.nrrd'
         method: mean_voxel_features()

       # Scenario 2: direct radiomics (single modality still needs concat; replace T2 with modality under data_dir)
       supervoxel_level:
         supervoxel_file_keyword: '*_supervoxel.nrrd'
         method: concat(supervoxel_radiomics(T2, params_file))
         params:
           params_file: ./config/radiomics/params_supervoxel_radiomics.yaml
           supervoxelBatch: 64
           useSupervoxelCext: auto
           useTorchRadiomics: auto
           # torchGpus: [0, 1]

       # Scenario 2b: multi-modality supervoxel radiomics
       supervoxel_level:
         supervoxel_file_keyword: '*_supervoxel.nrrd'
         method: concat(
           supervoxel_radiomics(T1, params_file),
           supervoxel_radiomics(T2, params_file))
         params:
           params_file: ./config/radiomics/params_supervoxel_radiomics.yaml
           useSupervoxelCext: auto
           useTorchRadiomics: auto

- ``params``: Parameters

  - **Type**: dict
  - **Required**: no
  - **Default**: ``{}``
  - **Description**: Parameters for extractors. Omit ``params`` for parameterless methods like ``mean_voxel_features()``. Common ``supervoxel_radiomics`` keys:
    ``params_file``, ``supervoxelBatch``, ``supervoxelUnionBboxCrop``, ``useSupervoxelCext``,
    ``useTorchRadiomics``, ``torchGpus``, ``torchGpuCount``, ``torchDtype`` (torch keys may inherit from
    ``voxel_level.params``).

**preprocessing_for_subject_level**: Subject-level preprocessing (optional)

- ``methods``: Preprocessing method list

  - **Type**: list
  - **Required**: no
  - **Default**: ``[]``
  - **Description**: Preprocess features per subject to reduce within-subject outliers and scale differences. Dispatched by
    ``PreprocessingMethodFactory`` with DataFrame in/out (see "Habitat feature preprocessing implementation and extension" below).
  - **Note**: In ``two_step`` and ``direct_pooling``, subject level must not use column-dropping methods (``variance_filter``, ``correlation_filter``), or concatenated subjects will have inconsistent columns; ``two_step`` rejects such configs at validation. ``one_step`` may use column-dropping methods at subject level (per-subject clustering).
  - **Supported methods and parameters**:

**winsorize (winsorization)**:

      - ``winsor_limits`` (list, default: ``[0.05, 0.05]``): lower and upper tail truncation fractions
      - ``global_normalize`` (bool, default: ``false``): global normalization across all features

**minmax (min-max normalization)**:

      - ``global_normalize`` (bool, default: ``false``): global normalization

**zscore (Z-score standardization)**:

      - ``global_normalize`` (bool, default: ``false``): global standardization

**robust (robust standardization)**:

      - ``global_normalize`` (bool, default: ``false``): global normalization
      - Scales using interquartile range (IQR); robust to outliers

**log (log transform)**:

      - ``global_normalize`` (bool, default: ``false``): global transform
      - Handles negative values (shift then log)

**variance_filter (low-variance filter)**:

      - ``variance_threshold`` (float, default: ``0.0``): keep features with variance above threshold
      - Note: drops feature columns

**correlation_filter (high-correlation filter)**:

      - ``corr_threshold`` (float, default: ``0.95``): drop redundant features when |correlation| exceeds threshold
      - ``corr_method`` (str, default: ``spearman``): ``pearson`` / ``spearman`` / ``kendall``
      - Note: drops feature columns

  - **Examples**:

    .. code-block:: yaml

       # Winsorize then normalize
       - method: winsorize
         winsor_limits: [0.05, 0.05]
         global_normalize: true
       - method: minmax
         global_normalize: true
       
       # Z-score standardization
       - method: zscore
         global_normalize: false

**preprocessing_for_group_level**: Group-level preprocessing (optional)

- ``methods``: Preprocessing method list

  - **Type**: list
  - **Required**: no
  - **Default**: ``[]``
  - **Description**: Preprocess features at cohort level; often used for discretization to stabilize clustering.
  - **Applicable modes**: ``two_step`` and ``direct_pooling`` only; ``one_step`` pipeline has no group step—configured group preprocessing has no effect.
  - **Supported methods and parameters**:

**binning (feature discretization / binning)**:

      - ``n_bins`` (int, default: ``10``): number of bins
      - ``bin_strategy`` (str, default: ``uniform``): binning strategy:

        - ``uniform``: equal-width bins
        - ``quantile``: quantile (equal-frequency) bins
        - ``kmeans``: K-means cluster binning

      - ``global_normalize`` (bool, default: ``false``): global binning across features

**winsorize (winsorization)**:

      - ``winsor_limits`` (list, default: ``[0.05, 0.05]``): lower and upper tail truncation fractions
      - ``global_normalize`` (bool, default: ``false``): global normalization

**minmax / zscore / robust / log**:

      - Same as ``preprocessing_for_subject_level``, applied after group aggregation

**variance_filter / correlation_filter (recommended at group level)**:

      - Column dropping for unsupervised feature selection; reduces noise and redundancy
      - ``variance_filter`` parameter: ``variance_threshold``
      - ``correlation_filter`` parameters: ``corr_threshold``, ``corr_method``
      - Recommendation: fit column set during training; reuse same columns at prediction

  - **Examples**:

    .. code-block:: yaml

       # Uniform binning (recommended for habitat analysis)
       - method: binning
         n_bins: 10
         bin_strategy: uniform
         global_normalize: false
       
       # Quantile binning (equal frequency)
       - method: binning
         n_bins: 20
         bin_strategy: quantile
         global_normalize: false

**YAML structure essentials**

``preprocessing_for_*_level`` must contain a **single** ``methods:`` key whose value is a **list**; do not duplicate ``methods:`` or place list items outside the ``methods:`` block:

.. code-block:: yaml

   # Correct
   preprocessing_for_group_level:
     methods:
       - method: winsorize
         winsor_limits: [0.05, 0.05]
       - method: variance_filter
         variance_threshold: 0.0

   # Wrong: duplicate methods key; variance_filter not in list
   preprocessing_for_group_level:
     methods:
       - method: winsorize
     methods:
       - method: variance_filter

**Preprocessing activation matrix by clustering_mode**

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18

   * - Preprocessing block
     - one_step
     - two_step
     - direct_pooling
   * - ``preprocessing_for_subject_level``
     - active
     - active (``variance_filter`` / ``correlation_filter`` forbidden)
     - active
   * - ``preprocessing_for_group_level``
     - **inactive**
     - active (Stage 2 group level)
     - active (after pooling, group level)

**Recommended pipeline (train → predict reuse)**

- **two_step / direct_pooling (train)**: subject-level ``winsorize`` + ``minmax`` (or ``zscore``) → group-level ``binning`` (discretization aids clustering) → optional ``variance_filter`` / ``correlation_filter`` (column-dropping; ``fit`` caches column set during training).
- **two_step / direct_pooling (predict)**: same ``methods`` order as training; ``PreprocessingState`` loaded from ``habitat_pipeline.pkl``—**do not change thresholds** on column-dropping methods to avoid column mismatch.
- **one_step**: configure ``preprocessing_for_subject_level`` only; group block is ignored.

**Habitat feature preprocessing implementation and extension**

- **Unified interface**: Built-in and custom methods implement ``BaseFeaturePreprocessing``,
  registered via ``@register_preprocessing`` on ``PreprocessingMethodFactory``.
- **Execution path**: ``preprocessing_for_subject_level`` → stateless subject-level
  ``apply_stateless_preprocessing``; ``preprocessing_for_group_level`` →
  ``PreprocessingState.fit/transform`` (training caches ``baseline`` and per-step
  ``step_states``; prediction reuses).
- **Column-dropping methods**: ``variance_filter``, ``correlation_filter`` set
  ``changes_columns=True``; ``two_step`` forbids them at subject level (see note above).
- **Adding a method**:

  1. See ``habit/core/habitat_analysis/feature_preprocessing/custom_preprocessing_template.py``
  2. Append method name to ``config_schemas.PreprocessingMethod.method`` Literal
  3. If column-dropping, update ``DROPPING_PREPROCESSING_METHODS``
  4. Ensure module import so registration decorator runs

- **Compatibility**: YAML format unchanged; legacy ``habitat_pipeline.pkl`` with pre-refactor
  ``PreprocessingState`` requires re-train.

**HabitatSegmentation**: Habitat segmentation settings

- **Type**: object
- **Required**: required in ``train``; recommended in ``predict`` (at least ``clustering_mode``)
- **Default**: ``null`` (if fully omitted, Pydantic uses ``HabitatSegmentationConfig`` default sub-blocks; see table below)

- ``clustering_mode``: Clustering strategy

  - **Type**: string
  - **Required**: no
  - **Default**: ``two_step``
  - **Allowed values**:

    - ``one_step``: cluster voxels directly.
    - ``two_step``: build supervoxels first, then cluster supervoxels into habitats.
    - ``direct_pooling``: pool voxels across all subjects then cluster (compute-heavy).

  - **Example**: ``two_step``

**supervoxel**: Supervoxel clustering settings

- **Applicable modes**:

  - ``two_step``: this block configures **per-subject** voxel → supervoxel clustering (``algorithm``, ``n_clusters``, etc.).
  - ``one_step``: ``algorithm`` and ``one_step_settings`` for **per-subject** voxel → habitat; ``n_clusters`` overridden when auto-selecting k.
  - ``direct_pooling``: supervoxel block not used for clustering (schema defaults may remain).

- ``algorithm``: Clustering or segmentation algorithm

  - **Type**: string
  - **Default**: ``kmeans``
  - **Allowed values** (aligned with ``SupervoxelClusteringConfig``):

    - ``kmeans``: K-means; clusters by feature vectors only—supervoxels may be spatially scattered.
    - ``gmm``: Gaussian mixture; underlying default ``covariance_type='full'``; ``SupervoxelClusteringConfig`` **does not** expose this field—YAML may ignore it.
    - ``slic``: SLIC superpixels; joint multi-channel features and 3D coordinates for spatially compact supervoxels. **Dedicated parameters** (``compactness`` / ``sigma`` / ``enforce_connectivity`` and tuning) see :ref:`habitat_slic_config` below.

- ``n_clusters``: Supervoxel count (or SLIC ``n_segments``)

  - **Type**: integer
  - **Default**: ``50``
  - **Description**:

    - ``two_step``: fixed supervoxel count per subject; must be less than ROI voxel count. Typical **30–100**; small ROI **20–30**.
    - ``one_step`` with empty ``fixed_n_clusters``: upper bound for auto k search (see ``one_step_settings.max_clusters``), not the final fixed value.

- ``random_state``: Random seed

  - **Type**: integer or ``null``
  - **Default**: ``null`` (inherits ``HabitatAnalysisConfig.random_state``)

- ``max_iter``: Maximum iterations

  - **Type**: integer
  - **Default**: ``300``
  - **Description**: ``kmeans`` / ``gmm``: sklearn max iterations; ``slic`` differs—see :ref:`habitat_slic_config`.

- ``n_init``: Number of initializations

  - **Type**: integer
  - **Default**: ``10``
  - **Description**: ``kmeans`` / ``gmm``: sklearn ``n_init``; ``slic``: only in ``one_step`` auto k—see :ref:`habitat_slic_config`.

- ``compactness`` / ``sigma`` / ``enforce_connectivity``: active only when ``algorithm: slic``—see :ref:`habitat_slic_config`.

- ``one_step_settings``: Nested one-step auto cluster count (see **one_step_settings** below). **Not used for per-subject supervoxel clustering in ``two_step``** (two-step always uses fixed ``n_clusters``).

- **Full example** (K-Means supervoxels):

  .. code-block:: yaml

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10

.. _habitat_slic_config:

HabitatSegmentation.supervoxel — SLIC Superpixel Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``HabitatSegmentation.supervoxel.algorithm: slic``, habit calls
``skimage.segmentation.slic`` inside the ROI, jointly segmenting **multi-channel voxel features**
(last dimension = feature channel) with **3D spatial coordinates** into ``n_clusters`` supervoxels. Anisotropic voxel spacing ``spacing``
is **read automatically** from the mask NIfTI header—**no** YAML entry required.

**Use cases**

- Recommended for ``two_step`` stage 1: supervoxels are more spatially compact than pure K-Means supervoxels.
- ``one_step`` may also use ``slic`` with ``one_step_settings`` for per-subject auto k.
- High-dimensional ``voxel_radiomics`` features incur much higher compute and memory than ``raw`` / ``mean_voxel_features`` (builds ``[Z, Y, X, n_features]`` feature volume in ROI bbox).

**YAML path**: keys under ``HabitatSegmentation.supervoxel``; repo template
``config/habitat/config_habitat_two_step_supervoxel_slic.yaml``.

**Parameter reference**

- ``n_clusters``: Target supervoxel count (``n_segments``)

  - **Default**: ``50`` (shared with ``supervoxel`` block)
  - **Tuning**: small ROI **20–30**; finer texture **60–100**; must be **less than** total ROI voxels.

- ``compactness``: Balance between feature similarity and spatial compactness

  - **Type**: float
  - **Default**: ``0.1``
  - **Description**: Passed to ``skimage.segmentation.slic``. Features are usually winsorized / minmax-normalized before SLIC, so **0.1** is a reasonable start (different scale from large compactness values common on RGB images).
  - **Tuning**:

    - **Increase** (``0.15``–``0.3``): more spatial grouping, smoother boundaries, fewer fragments.
    - **Decrease** (``0.05``): follows feature differences more closely; boundaries track intensity/texture change but may fragment.

- ``sigma``: Gaussian smoothing width before SLIC

  - **Type**: float
  - **Default**: ``0.0`` (no smoothing)
  - **Description**: Gaussian smooth on **multi-channel feature volume** inside ROI before segmentation; units are **voxels** (with auto ``spacing`` for anisotropy). Suppresses high-frequency noise in voxel/texture features; **does not** replace winsorize etc. in ``preprocessing_for_subject_level``.
  - **Tuning**:

    - ``0.0``: sharpest boundaries; use when preprocessing is already strong.
    - ``0.3``–``0.5``: mild denoising; try first for **voxel radiomics** (e.g. ``0.5``).
    - ``> 1.0``: may blur real small structures—generally not recommended.

- ``enforce_connectivity``: Enforce spatial connectivity

  - **Type**: bool
  - **Default**: ``true``
  - **Description**: When ``true``, merges isolated small regions to avoid "floating" supervoxels; **keep true** for medical ROIs.

- ``max_iter``: SLIC iteration count

  - **Type**: integer
  - **Schema default**: ``300`` (shared field name with kmeans; **do not use 300 for SLIC**)
  - **Recommended**: ``10`` (maps to ``max_num_iter``); ``15``–``20`` possible with diminishing returns.

- ``n_init``: Number of initializations

  - **Type**: integer
  - **Default**: ``10``
  - **Description**: **Only** for embedded KMeans in ``find_optimal_clusters`` under ``one_step``; **no effect** on SLIC ``fit`` in ``two_step``.

- ``random_state``: Random seed

  - **Default**: ``null`` (inherits top-level ``random_state``); explicit ``42`` recommended for reproducibility.

**Comparison with K-Means supervoxels**

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Algorithm
     - Advantages
     - Caveats
   * - ``kmeans``
     - Fast; simple
     - No spatial adjacency constraint; supervoxels may scatter
   * - ``slic``
     - Spatially compact supervoxels; good two-step local units
     - Slower, more memory with high-dim radiomics; needs mask / coordinates

**Recommended example** (``two_step`` supervoxel stage):

.. code-block:: yaml

   HabitatSegmentation:
     clustering_mode: two_step
     supervoxel:
       algorithm: slic
       n_clusters: 50
       compactness: 0.1
       sigma: 0.5
       enforce_connectivity: true
       max_iter: 10
       random_state: 42

**one_step_settings note**: Nested block is ``one_step`` only; under ``two_step``, SLIC supervoxel count is fixed by ``n_clusters``—**do not** rely on ``one_step_settings`` for k selection.

**one_step_settings**: One-step mode settings (``one_step`` only)

- ``min_clusters``: Minimum cluster count

  - **Type**: integer
  - **Default**: ``2``
  - **Description**: Lower bound for automatic selection

- ``max_clusters``: Maximum cluster count

  - **Type**: integer
  - **Default**: ``10``
  - **Description**: Upper bound for automatic selection

- ``fixed_n_clusters``: Fixed cluster count

  - **Type**: integer or null
  - **Default**: ``null``
  - **Description**: If set, skips auto selection and uses this value.

- ``selection_method``: Auto-selection metric

  - **Type**: string
  - **Default**: ``silhouette``
  - **Allowed values and meaning**:

    - ``silhouette``: silhouette coefficient (-1 to 1; closer to 1 = tighter clusters)
    - ``calinski_harabasz``: Calinski-Harabasz index (higher = better)
    - ``davies_bouldin``: Davies-Bouldin index (lower = better)
    - ``inertia``: within-cluster sum of squares (lower = tighter; Kneedle elbow internally)
    - ``kneedle``: Kneedle on normalized inertia curve (max deviation point)

  - **Recommendation**: ``silhouette`` (strong overall performance)

- ``plot_validation_curves``: Plot validation curves

  - **Type**: bool
  - **Default**: ``true``
  - **Description**: Plots metrics vs cluster count to interpret auto selection

**habitat**: Habitat clustering settings

- ``algorithm``: Clustering algorithm

  - **Type**: string
  - **Default**: ``kmeans``
  - **Allowed values**:

    - ``kmeans``: K-means clustering
    - ``gmm``: Gaussian mixture model

- ``max_clusters``: Maximum habitat count

  - **Type**: integer
  - **Required**: no
  - **Default**: ``10``
  - **Description**: Upper bound when auto-selecting habitat count. Recommended: 5–10.
  - **Example**: ``10``

- ``min_clusters``: Minimum habitat count

  - **Type**: integer
  - **Default**: ``2``
  - **Description**: Lower bound when auto-selecting habitat count.

- ``habitat_cluster_selection_method``: Auto-selection metrics

  - **Type**: list or string
  - **Default**: ``inertia`` (YAML may use string or single-element list)
  - **Allowed values and meaning**:

    - ``inertia``: within-cluster SS (lower better for kmeans; Kneedle internally)
    - ``kneedle``: Kneedle on normalized inertia curve
    - ``silhouette``: silhouette (-1 to 1; closer to 1 better)
    - ``calinski_harabasz``: Calinski-Harabasz (higher better)
    - ``davies_bouldin``: Davies-Bouldin (lower better)
    - ``aic``: Akaike information criterion (lower better; gmm only)
    - ``bic``: Bayesian information criterion (lower better; gmm only)

  - **Description**: Multiple metrics may be specified; system combines them to pick best habitat count.
  - **Example**: ``[inertia, silhouette]``

- ``fixed_n_clusters``: Fixed habitat count

  - **Type**: integer or null
  - **Default**: ``null``
  - **Description**: If set to a number, skips auto selection and uses that habitat count.

- ``random_state``: Random seed

  - **Type**: integer or ``null``
  - **Default**: ``null`` (inherits ``HabitatAnalysisConfig.random_state``)
  - **Description**: **direct_pooling / two_step** group habitat clustering; **one_step** per-subject voxel→habitat (overrides ``supervoxel.random_state`` when set explicitly).

- ``max_iter``: Maximum iterations

  - **Type**: integer
  - **Default**: ``300`` (kmeans) or ``100`` (gmm)

- ``n_init``: Number of initializations

  - **Type**: integer
  - **Default**: ``10`` (kmeans) or ``1`` (gmm)

- **Full examples**:

  .. code-block:: yaml

     # Auto-select habitat count (recommended)
     habitat:
       algorithm: kmeans
       max_clusters: 10
       min_clusters: 2
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       fixed_n_clusters: null
       random_state: 42
     
     # Fixed habitat count
     habitat:
       algorithm: kmeans
       fixed_n_clusters: 5
       random_state: 42

**postprocess_supervoxel / postprocess_habitat**: Connected-component postprocessing

- **Type**: dict
- **Required**: no
- **Default**: ``enabled: false``
- **Description**:

  - ``postprocess_supervoxel`` applies to supervoxel label maps (mainly two_step stage).
  - ``postprocess_habitat`` applies to final habitat label maps (one_step/two_step/direct_pooling).
  - Current implementation uses SimpleITK fast path: remove small components per label, then reassign by nearest seed label.
  - Reduces fragmentation while preserving ROI voxel coverage.

- **Sub-parameters** (aligned with ``ConnectedComponentPostprocessConfig``):

  - ``enabled`` (bool, default: ``false``)
  - ``min_component_size`` (int, default: ``30``, ≥1)
  - ``connectivity`` (1 / 2 / 3): 6/18/26-neighbor
  - ``reassign_method``: currently only ``neighbor_vote`` (placeholder)
  - ``max_iterations`` (int, default ``3``, ≥1): cleanup iteration cap

- **Example**:

  .. code-block:: yaml

     HabitatSegmentation:
       postprocess_supervoxel:
         enabled: false
         min_component_size: 30
         connectivity: 1
         reassign_method: neighbor_vote
         max_iterations: 3

       postprocess_habitat:
         enabled: true
         min_component_size: 30
         connectivity: 1
         reassign_method: neighbor_vote
         max_iterations: 3

**plot_curves**: Generate and save plots

- **Type**: bool
- **Default**: ``true``
- **Description**: During group clustering auto k search, if a logger is passed, logs ``Trying N cluster(s) [i/total]`` and ``Cluster search finished: selected k=...`` for progress (forced off in ``predict`` mode).

Habitat Stage-1 Parallelism and Checkpoint Resume (Top-Level Field Reference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These fields sit at the **top level** of the habitat YAML (same level as ``data_dir``). ``predict`` mode ignores parallelism and checkpoint fields. Resume fields are documented below (``resume``, ``strict_checkpoint_hash``, ``.habitat_checkpoint``, etc.); CLI: ``habit get-habitat --resume``.

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 14 36

   * - Field
     - Type
     - Default
     - config_hash
     - Summary
   * - ``processes``
     - int
     - ``2``
     - no
     - Stage 1 max parallel workers; peak memory ≈ ``processes × per-subject memory``
   * - ``cap_processes_to_gpu_pool``
     - bool
     - ``true``
     - no
     - Torch CUDA radiomics: ``true`` caps workers to ``len(torchGpus)``; ``false`` keeps ``processes`` and shares GPUs across workers
   * - ``individual_subject_timeout_sec``
     - float / ``null``
     - ``900``
     - no
     - Per-subject wall-clock cap (seconds); ``null`` disables
   * - ``individual_subject_graceful_shutdown_sec``
     - float
     - ``15``
     - no
     - Seconds to wait after ``terminate()`` before ``kill()`` on timeout
   * - ``individual_subject_spawn_timeout_sec``
     - float / ``null``
     - ``120``
     - no
     - Spawn-phase cap; ``null`` disables (avoids parent stuck on import)
   * - ``on_subject_failure``
     - str
     - ``continue``
     - no
     - ``continue`` log failure and proceed; ``fail_fast`` abort on first failure
   * - ``oom_backoff``
     - bool
     - ``true``
     - no
     - After ``MemoryError``, reduce workers by ``oom_reduce_workers_by`` (not native crashes)
   * - ``oom_reduce_workers_by``
     - int
     - ``1``
     - no
     - Workers removed per OOM event
   * - ``resume``
     - bool
     - ``true``
     - —
     - Skip completed subjects from checkpoint; ``train`` only
   * - ``checkpoint_dir``
     - str / ``null``
     - ``null``
     - no
     - Default ``<out_dir>/.habitat_checkpoint``
   * - ``force_rerun_subjects``
     - list[str]
     - ``[]``
     - no
     - Subject IDs to force rerun on resume
   * - ``retry_failed_subjects``
     - bool
     - ``false``
     - no
     - On **next** ``resume``, rerun all ``failed_subjects`` in manifest
   * - ``individual_subject_auto_retry_rounds``
     - int
     - ``2``
     - no
     - In-run Stage 1 auto-retry rounds; ``0`` disables
   * - ``individual_subject_parallel_mode``
     - str
     - ``persistent``
     - no
     - ``persistent`` long-lived workers; ``isolated`` spawn per subject
   * - ``persistent_worker_max_consecutive_failures``
     - int
     - ``1``
     - no
     - Under ``persistent``, restart slot after N consecutive failures
   * - ``persistent_worker_recycle_after_tasks``
     - int
     - ``0``
     - no
     - Under ``persistent``, recycle worker after N successes; ``0`` off
   * - ``clear_checkpoint_on_success``
     - bool
     - ``false``
     - no
     - Delete checkpoint dir after full train success

**processes** (habitat analysis top level): Parallel workers for per-subject steps

- **Type**: integer
- **Default**: ``2`` (must be ``> 0``)
- **Description**: See table above; interacts with ``cap_processes_to_gpu_pool`` and ``torchGpus`` in ``FeatureConstruction.*.params``.

**cap_processes_to_gpu_pool** (habitat analysis top level): Cap Stage 1 workers to GPU pool size

- **Type**: bool
- **Default**: ``true``
- **Description**: When ``useTorchRadiomics`` uses CUDA (``true`` or ``auto`` with CUDA detected):

  - ``true`` (default): effective workers ``min(processes, len(torchGpus))``, one GPU per slot (``gpuSlotIndex``), less VRAM contention;
  - ``false``: full ``processes``; workers share GPUs via ``gpuSlotIndex % len(torchGpus)``—good for "single GPU, many CPU" parallel non-GPU steps, but GPU radiomics may OOM on same card.

- **Not in config_hash**; may change on resume.
- **No effect** on CPU-only (``useTorchRadiomics: false`` or no CUDA).

**individual_subject_timeout_sec** (habitat analysis top level): Per-subject wall-clock cap in parallel Stage 1

- **Type**: float / int (seconds) or ``null``
- **Default**: ``900`` (15 minutes); omit in YAML to use default.
- **Description**: On timeout, skip subject (record failure) and continue; ``null`` disables per-subject timeout. Child processes may still run in background until they exit.

**individual_subject_graceful_shutdown_sec** (habitat analysis top level): Grace period after timeout before kill

- **Type**: float (seconds)
- **Default**: ``15``
- **Description**: After ``individual_subject_timeout_sec``, parent calls ``terminate()``, waits this many seconds, then ``kill()`` on isolated child.

**individual_subject_spawn_timeout_sec** (habitat analysis top level): Spawn-phase wall-clock cap

- **Type**: float / int (seconds) or ``null``
- **Default**: ``120``
- **Description**: Cap from dispatch to subject processing start; on timeout, mark subject failed and continue—avoids parent blocked forever on spawn/import. ``null`` = no spawn time limit.

**on_subject_failure** (habitat analysis top level): Failure policy for parallel per-subject Stage 1

- **Type**: string
- **Default**: ``continue``
- **Allowed values**:

  - ``continue``: record failed subjects; continue to Stage 2 if any succeeded
  - ``fail_fast``: abort entire run on first failure or timeout

**oom_backoff** (habitat analysis top level): Reduce parallelism after memory errors

- **Type**: bool
- **Default**: ``true`` (built-in default); repo ``config/habitat/*.yaml`` examples often use ``false``—tune for your RAM
- **Description**: When ``true``, isolated child ``MemoryError`` reduces worker count for pending subjects by ``oom_reduce_workers_by`` (minimum 1). **Does not handle** native crashes (e.g. Windows exit ``3221225477`` / ``0xC0000005``).

**oom_reduce_workers_by** (habitat analysis top level): Workers removed per OOM

- **Type**: integer
- **Default**: ``1``
- **Description**: Only when ``oom_backoff: true``.

**resume** (habitat analysis top level): Stage 1 checkpoint resume

- **Type**: bool
- **Default**: ``true``
- **Description**: When ``true``, reads ``manifest.json`` from ``checkpoint_dir`` (default ``<out_dir>/.habitat_checkpoint``), skips ``completed_subjects``, loads ``subjects/{id}.pkl``; subjects in ``failed_subjects`` are **not auto-retried on next** ``resume`` unless ``retry_failed_subjects: true`` or listed in ``force_rerun_subjects``. **Within the same** ``train`` run, ``individual_subject_auto_retry_rounds`` retries Stage 1 failures by default. ``run_mode: train`` only.
- **CLI**: ``habit get-habitat --resume`` equivalent to ``resume: true``.
- **See also**: checkpoint / ``resume`` fields on this page.
- **Parallel reliability plan**: ``docs/HABITAT_PARALLEL_RELIABILITY_PLAN.md`` at repo root (GPU worker slots, processes cap, Phase 2/3 roadmap).

**strict_checkpoint_hash** (habitat analysis top level): Error on incompatible checkpoint hash

- **Type**: bool
- **Default**: ``false``
- **Description**: With ``resume: true``. When ``true``, ``manifest.json`` ``config_hash`` or ``run_mode`` mismatch raises ``CheckpointConfigHashError`` and preserves checkpoint; when ``false`` (default), logs warning and deletes checkpoint for fresh run. Legacy hash migration for Stage-2-only changes still allows resume.
- **Not in config_hash**; may change on resume.

**checkpoint_dir** (habitat analysis top level): Checkpoint root directory

- **Type**: string or ``null``
- **Default**: ``null`` (``train`` → ``<out_dir>/.habitat_checkpoint``; ``predict`` → ``<out_dir>/.habitat_predict_checkpoint``)
- **Description**: Must match previous run for resume; may differ from ``out_dir`` when set explicitly.

**force_rerun_subjects** (habitat analysis top level): Subject IDs to force rerun

- **Type**: list of strings
- **Default**: ``[]``
- **Description**: With ``resume: true``, still reprocess listed subjects (removed from completed/failed and rerun).

**retry_failed_subjects** (habitat analysis top level): Rerun all failed subjects from checkpoint

- **Type**: bool
- **Default**: ``false``
- **Description**: With ``resume: true``, enqueue all ``failed_subjects`` from ``manifest.json`` for Stage 1 rerun. Successful subjects still skipped unless also in ``force_rerun_subjects``.

**individual_subject_auto_retry_rounds** (habitat analysis top level): In-run auto-retry for failed subjects

- **Type**: integer
- **Default**: ``2``
- **Description**: After first Stage 1 parallel pass, if ``failed_subjects`` remain, rerun up to this many rounds in the same process (failed only). ``0`` disables (legacy behavior). Unlike ``retry_failed_subjects`` (next **resume** only), this applies in the **current** ``get-habitat`` / ``fit()`` call. With ``on_subject_failure: fail_fast``, error only after all retry rounds exhausted.

**individual_subject_parallel_mode** (habitat analysis top level): Stage 1 parallel execution strategy

- **Type**: string
- **Default**: ``persistent``
- **Allowed values**: ``isolated``, ``persistent``
- **Description**: ``persistent`` (default): one long-lived child per worker slot, reused within the same ``train`` run (including auto-retry rounds), amortizing import/spawn. ``isolated``: spawn per subject (stronger isolation; unpickleable pipeline or spawn debugging). Single GPU persistent is still serial—main benefit is startup cost. When ``processes=1`` and ``individual_subject_timeout_sec: null``, both modes run sequentially in main process without spawn. Ignored in ``predict``.

**persistent_worker_max_consecutive_failures** (habitat analysis top level): Persistent worker restart threshold

- **Type**: integer
- **Default**: ``1``
- **Description**: Only when ``individual_subject_parallel_mode: persistent``. After this many consecutive failures on a slot, parent terminates and restarts that worker.

**persistent_worker_recycle_after_tasks** (habitat analysis top level): Periodic persistent worker recycle

- **Type**: integer
- **Default**: ``0``
- **Description**: ``persistent`` only. Worker exits after this many successful tasks and parent respawns—mitigates slow GPU memory leaks. ``0`` disables.

**clear_checkpoint_on_success** (habitat analysis top level): Delete checkpoint after successful train

- **Type**: bool
- **Default**: ``false``
- **Description**: When ``true``, deletes entire checkpoint directory after Stage 1 + Stage 2 complete successfully.

**config_hash and resume compatibility**

- **In hash** (Stage 1 per-subject; change clears checkpoint): ``data_dir``, ``FeatureConstruction.voxel_level`` / ``preprocessing_for_subject_level`` / ``supervoxel_level``, ``HabitatSegmentation.clustering_mode``, per-subject clustering block (``two_step`` → ``supervoxel``; ``one_step`` → ``supervoxel`` + ``habitat``).
- **Not in hash** (may ``resume: true``): ``preprocessing_for_group_level``, group ``habitat.*`` for ``two_step``/``direct_pooling``, ``processes``, ``cap_processes_to_gpu_pool``, ``strict_checkpoint_hash``, ``individual_subject_timeout_sec``, ``individual_subject_graceful_shutdown_sec``, ``individual_subject_spawn_timeout_sec``, ``plot_curves``, ``save_results_csv``, ``habitats_results_format``, ``save_images``, ``verbose``, ``debug``, ``on_subject_failure``, ``oom_backoff``, ``oom_reduce_workers_by``, ``retry_failed_subjects``, ``individual_subject_auto_retry_rounds``, ``individual_subject_parallel_mode``, ``persistent_worker_max_consecutive_failures``, ``persistent_worker_recycle_after_tasks``, ``force_rerun_subjects``, ``out_dir``, etc.
- ``manifest.json`` also stores ``individual_config_hash`` (same as ``config_hash``); legacy full-hash-only manifests migrate hash on Stage 2-only changes and keep pkls.
- On ``resume: true`` startup, program compares hash. Individual hash mismatch without Stage 2 drift: default (``strict_checkpoint_hash: false``) warns and deletes checkpoint; ``true`` raises ``CheckpointConfigHashError``.

**Checkpoint directory layout**

.. code-block:: text

   <checkpoint_dir>/
   ├── manifest.json      # completed_subjects, failed_subjects, config_hash, individual_config_hash, stage
   └── subjects/
       └── {subject_id}.pkl

**Checkpoint boundaries by clustering_mode**

- ``two_step`` / ``one_step``: saved after ``merge_supervoxel_features`` (``supervoxel_df``)
- ``direct_pooling``: saved after ``individual_preprocessing`` (voxel-level ``features``; larger pkls)
- Stage 2 (combine / concat / group clustering) has **no** checkpoint

**save_results_csv**: Save habitat results table

- **Type**: bool
- **Default**: ``true``
- **Description**: When ``true``, writes ``habitats.parquet`` (default) or ``habitats.csv`` per ``habitats_results_format``.

**habitats_results_format**: Habitat results table file format

- **Type**: string
- **Default**: ``parquet``
- **Allowed values**: ``parquet``, ``csv``
- **Description**: ``parquet`` smaller and faster—good for large ``direct_pooling`` voxel tables; ``csv`` opens in Excel. Output filename is ``habitats.parquet`` or ``habitats.csv``.
- **Example**:

  .. code-block:: yaml

     save_results_csv: true
     habitats_results_format: parquet

**random_state** (habitat analysis top level)

- **Type**: integer
- **Default**: ``42``

**debug** (habitat analysis top level)

- **Type**: bool
- **Default**: ``false``

**habitat_pipeline.pkl** (training artifact at ``<out_dir>/habitat_pipeline.pkl``)

- **Contents**: joblib-serialized fitted ``HabitatPipeline`` (group clustering model, ``PreprocessingState``, config, etc.).
- **Auto slimming on save** (``HabitatPipeline.save()`` calls ``prepare_pipeline_for_save``):

  - Removes training ``labels_`` (prediction needs centers/model params only)
  - **Does not write** ``mask_info_cache`` (NRRD write loads mask from ``data_dir`` on demand via ``FeatureService.load_mask_info``)
  - Does not write ``_train_checkpoint`` (checkpoints remain in ``checkpoint_dir``)

- **Size**:

  - No longer inflated by ``save_images: true``: mask volume decoupled from pkl; ``direct_pooling`` large queues typically **tens to hundreds of MB** (mainly feature dimension and model params vs subject count)
  - Legacy pkls embedding ``mask_array`` need **re-train and save** to shrink

**save_images**: Save image outputs from the run (``*_habitats.nrrd``, etc.)

- **Type**: bool
- **Default**: ``true``
- **Description**: Maps to ``HabitatAnalysisConfig.save_images``. When ``true``, train/predict write habitat label maps; masks load from ``config.data_dir`` at write time—**not** stored in ``habitat_pipeline.pkl``. When ``false``, downstream analysis via ``habitats.parquet`` / ``habitats.csv`` without NRRD output.

**verbose**: Verbose run logging

- **Type**: bool
- **Default**: ``true``

