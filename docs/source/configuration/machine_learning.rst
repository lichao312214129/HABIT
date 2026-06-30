Machine Learning Configuration
==============================

Machine learning configuration parameters
---------------------------------------

This section documents **machine learning** configuration. CLI: ``habit model -c <yaml>`` (K-fold: ``habit cv``). Demo: ``config/machine_learning/config_machine_learning_radiomics.yaml``; prediction: ``config/machine_learning/config_machine_learning_predict.yaml``.

**Example configuration file:**

.. code-block:: yaml

   run_mode: train
   input:
     - path: ./results/features/combined_features.csv
       name: training_data
       subject_id_col: Subject
       label_col: label
   output: ./demo_data/results/ml/train
   random_state: 42

   split_method: stratified
   test_size: 0.3

   resampling:
     enabled: false
     method: random_over
     position: before_model
     ratio: 1.0

   normalization:
     method: z_score

   feature_selection_methods:
     - method: variance
       params:
         threshold: 0.0
     - method: correlation
       params:
         threshold: 0.9
   
   models:
     RandomForest:
       params:
         n_estimators: 100
         random_state: 42
     LogisticRegression:
       params:
         max_iter: 1000
   
   is_visualize: true
   is_save_model: true
   
   visualization:
     enabled: true
     plot_types: [roc, dca, calibration, pr, confusion, shap]
     dpi: 600
     format: pdf

**Prediction mode YAML example** (``run_mode: predict``):

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./demo_data/results/ml/clinical/models/LogisticRegression_final_pipeline.pkl

   input:
     - path: ./demo_data/ml_data/clinical_feature.csv
       subject_id_col: subjID
       label_col: label   # required when evaluate: true

   output: ./demo_data/results/ml/clinical/predictions

   evaluate: true
   output_label_col: predicted_label
   output_prob_col: predicted_probability

**run_mode** (YAML)

- **Type**: string
- **Default**: ``train``
- **Allowed values**: ``train``, ``predict``
- **Description**: train and predict share the same ML config structure. ``predict`` requires ``pipeline_path``; ``models`` is ignored in predict mode.

**mode / run_mode (CLI)**

- **Command**: ``habit model --mode <train|predict>``
- **Description**: ``--mode`` **overrides** YAML ``run_mode`` (see ``cmd_ml.run_ml``). CLI takes precedence.

**pipeline_path**

- **Type**: string
- **Required**: **required** when ``run_mode`` is ``predict``
- **Default**: ``null``
- **Description**: path to saved ``*_final_pipeline.pkl``.

**random_state** (``MLConfig`` top level)

- **Type**: integer
- **Required**: no
- **Default**: ``42``
- **Description**: split, K-fold, resampling fallback, models without seed in ``models.*.params``, etc.

**input**: input data configuration

- **Type**: list
- **Required**: yes
- **Description**: each element is ``InputFileConfig``. Predict mode uses only ``input[0].path`` as the data table.
- **Sub-parameters**:

  - ``path``: feature CSV/Excel path (**required**, no default).
  - ``name``: dataset name; **default** ``""``.
  - ``subject_id_col``: subject ID column (**required**, no default).
  - ``label_col``: label column (**required**, no default).
  - ``features``: use only these columns as features; **default** ``null`` (auto-infer numeric feature columns).
  - ``features_from_log``: parse feature column names from log/auxiliary file; **default** ``null``.
  - ``split_col``: custom split grouping column; **default** ``null``.
  - ``pred_col``: existing prediction column name; **default** ``null``.

**output**: output directory

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: results, models, and plots directory. CLI writes logs here: ``processing.log`` for training, ``prediction.log`` for prediction (see ``habit.cli_commands.commands.cmd_ml``).

**split_method**: data split method

- **Type**: string
- **Default**: ``stratified``
- **Allowed values**: ``random``, ``stratified``, ``custom``

**test_size**: test set proportion

- **Type**: float
- **Default**: ``0.3``
- **Range**: (0, 1)

**train_ids_file** / **test_ids_file**

- **Type**: string path (optional)
- **Default**: ``null``
- **Description**: used when ``split_method: custom``; text files with one subject ID per line for fixed train/test split.

**n_splits** / **stratified**

- **Type**: integer / boolean
- **Default**: ``n_splits=5``, ``stratified=true``
- **Description**: for ``habit cv`` (K-fold); field definitions in ``MLConfig``.

**normalization**: feature normalization settings

- ``method``: normalization method

  - **Type**: string
  - **Default**: ``z_score``
  - **Allowed values**:

    - ``z_score``: Z-Score standardization (StandardScaler)
    - ``min_max``: min-max scaling (MinMaxScaler)
    - ``robust``: robust scaling (RobustScaler)
    - ``max_abs``: max absolute value scaling (MaxAbsScaler)
    - ``normalizer``: L1/L2 normalization (Normalizer)
    - ``quantile``: quantile transform (QuantileTransformer)
    - ``power``: power transform (PowerTransformer)

- ``params``: method-specific parameters

  - **Type**: dict
  - **Default**: ``{}``
  - **Description**: pass different parameters per method. Omit ``params`` for default behavior (e.g. ``z_score``, ``max_abs``).
  - **Parameters supported per method**:

**z_score (StandardScaler)**:

      - ``with_mean`` (bool, default: ``true``): center data before scaling
      - ``with_std`` (bool, default: ``true``): scale to unit variance

**min_max (MinMaxScaler)**:

      - ``feature_range`` (list, default: ``[0, 1]``): target range, e.g. ``[0, 1]`` or ``[-1, 1]``

**robust (RobustScaler)**:

      - ``with_centering`` (bool, default: ``true``): center data before scaling
      - ``with_scaling`` (bool, default: ``true``): scale by IQR
      - ``quantile_range`` (list, default: ``[25.0, 75.0]``): quantile range for scaling (IQR)

**max_abs (MaxAbsScaler)**:

      - no special parameters (defaults suffice)

**quantile (QuantileTransformer)**:

      - ``n_quantiles`` (int, default: ``1000``): number of quantiles
      - ``output_distribution`` (str, default: ``uniform``): ``uniform`` or ``normal``
      - ``subsample`` (int, default: ``10000``): max samples for quantile estimation

**power (PowerTransformer)**:

      - ``method`` (str, default: ``yeo-johnson``): ``yeo-johnson`` or ``box-cox``

  - **Examples**:

    .. code-block:: yaml

       # Z-Score (no extra params needed)
       normalization:
         method: z_score

       # Min-max to [-1, 1]
       normalization:
         method: min_max
         params:
           feature_range: [-1, 1]
       
       # Robust scaling (outlier-resistant)
       normalization:
         method: robust
         params:
           quantile_range: [25.0, 75.0]

**resampling**: training set resampling (``ResamplingConfig``)

- **YAML key**: ``resampling`` (recommended). Legacy key ``sampling`` is auto-mapped to ``resampling`` on load (see ``MLConfig._migrate_legacy_sampling_key``).
- **Required**: no
- **Default**: ``enabled: false`` (no class resampling on training set)
- **Description**: resampling applies to **training** data only; validation/test sets are not resampled.

- ``enabled``: whether enabled

  - **Type**: boolean
  - **Default**: ``false``

- ``method``: algorithm

  - **Type**: string
  - **Default**: ``random_over``
  - **Allowed values**: ``random_over`` | ``random_under`` | ``smote`` (SMOTE requires ``imbalanced-learn``)

- ``position``: position in the pipeline

  - **Type**: string
  - **Default**: ``before_model``
  - **Allowed values**: ``before_feature_selection``, ``before_normalization``, ``after_normalization``, ``before_model``
  - **Description**: order relative to feature selection / normalization / modeling (see ``habit.core.machine_learning`` workflow).

- ``ratio``: resampling ratio; must be **> 0**

  - **Default**: ``1.0``

- ``random_state``: random seed (default ``null``, inherits ``MLConfig.random_state``; explicit value overrides top level)

- **When it runs**: training calls internal ``_resample_training_data`` before model fit; holdout and K-fold share this logic.

- **Log keywords** (confirm execution): ``Sampling enabled``, ``Sampling completed``, etc.

- **Example**:

  .. code-block:: yaml

     resampling:
       enabled: true
       method: random_over
       position: before_model
       ratio: 1.0
       random_state: 42

**feature_selection_methods**: feature selection method list

- **Type**: list
- **Default**: ``[]`` (empty list = no feature selection step)
- **Description**: sequential feature selection steps; each method has specific parameters.
- **Methods and parameters**:

**variance (variance threshold)**:

    - ``threshold`` (float, default: ``0.0``): remove features with variance below this
    - ``top_k`` (int, optional): keep top k highest-variance features (overrides threshold if set)
    - ``top_percent`` (float, optional): keep top x% highest-variance features (0–100)
    - ``plot_variances`` (bool, default: ``true``): plot variance distribution

**correlation (correlation filter)**:

    - ``threshold`` (float, default: ``0.8``): remove one of pairs with correlation above this
    - ``method`` (str, default: ``spearman``): ``pearson``, ``spearman``, ``kendall``
    - ``visualize`` (bool, default: ``false``): generate correlation heatmap

**anova (ANOVA)**:

    - ``p_threshold`` (float, default: ``0.05``): p-value threshold
    - ``n_features_to_select`` (int, optional): select top n features (overrides p_threshold if set)
    - ``plot_importance`` (bool, default: ``true``): plot feature importance

**chi2 (chi-square test)**:

    - ``p_threshold`` (float, default: ``0.05``): p-value threshold
    - ``n_features_to_select`` (int, optional): select top n features
    - ``plot_importance`` (bool, default: ``true``): plot feature importance
    - **Note**: non-negative features only

**lasso (Lasso regularization)**:

    - ``cv`` (int, default: ``10``): cross-validation folds
    - ``n_alphas`` (int, default: ``100``): number of alpha values
    - ``alphas`` (list, optional): custom alpha list
    - ``random_state`` (int, default: ``42``): random seed
    - ``visualize`` (bool, default: ``false``): coefficient path plot

**rfecv (recursive feature elimination + CV)**:

    - ``estimator`` (str, default: ``RandomForestClassifier``): estimator, options:

      - classifiers: ``LogisticRegression``, ``RandomForestClassifier``, ``SVC``, ``GradientBoostingClassifier``, ``XGBClassifier``
      - regressors: ``LinearRegression``, ``RandomForestRegressor``, ``SVR``, ``GradientBoostingRegressor``, ``XGBRegressor``

    - ``step`` (int, default: ``1``): features removed per iteration
    - ``cv`` (int, default: ``5``): CV folds
    - ``scoring`` (str, default: ``roc_auc``): scoring metric
    - ``min_features_to_select`` (int, default: ``1``): minimum features to keep
    - ``n_jobs`` (int, default: ``-1``): parallel jobs (``-1`` = all CPUs)
    - ``random_state`` (int, optional): random seed

**statistical_test (t-test / Mann-Whitney U, automatic or forced)**:

    - ``p_threshold`` (float, default: ``0.05``)
    - ``n_features_to_select`` (int, optional): overrides p threshold if set
    - ``normality_test_threshold`` (float, default: ``0.05``): Shapiro-Wilk normality threshold
    - ``force_test`` (str, optional): ``ttest`` or ``mannwhitney``; auto-select by normality if unset
    - ``plot_importance`` (bool, default: ``true``)

**icc (stability filter from ICC JSON)**:

    - ``icc_results`` / ``icc_results_path`` (str): JSON path from ``habit icc``
    - ``keys`` / ``groups`` (list): group names to check in ICC results
    - ``threshold`` (float, default: ``0.75``)
    - ``metric`` (str, optional): e.g. ``ICC3``, ``ICC2``

**mrmr (minimum redundancy maximum relevance)**:

    - ``n_features`` (int, default: ``10``)
    - ``task_type`` (str, default: ``classification``): ``classification`` or ``regression``

**vif (variance inflation factor, collinearity removal)**:

    - ``max_vif`` (float, default: ``10.0``)
    - ``visualize`` (bool, default: ``false``)

**stepwise (Python stepwise logistic regression)**:

    - ``direction`` (str, default: ``backward``): ``forward``, ``backward``, ``both``
    - ``threshold_in`` / ``threshold_out`` (float, default: ``0.05``): when ``criterion='pvalue'``
    - ``criterion`` (str, default: ``aic``): ``aic``, ``bic``, or ``pvalue``
    - ``verbose`` (bool, default: ``false``)

**stepwise_r (R stepwise regression; requires R)**:

    - same parameters as ``stepwise``; method key name is ``stepwise_r``

**univariate_logistic (univariate logistic regression)**:

    - ``alpha`` (float, default: ``0.05``): significance level

- **Examples**:

  .. code-block:: yaml

     # Variance threshold
     feature_selection_methods:
       - method: variance
         params:
           threshold: 0.0
           plot_variances: true
     
     # Correlation filter + ANOVA
     feature_selection_methods:
       - method: correlation
         params:
           threshold: 0.9
           method: spearman
       - method: anova
         params:
           p_threshold: 0.05

**models**: model training settings

- **Type**: dict (model name → ``ModelConfig``)
- **Default**: ``null`` (required and non-empty when ``run_mode: train``; ignored when ``predict``)
- **Description**: one or more models to train.

- **Supported model types and common parameters**:

**LogisticRegression**:

    - ``max_iter`` (int, default: ``100``): max iterations
    - ``C`` (float, default: ``1.0``): inverse regularization strength
    - ``penalty`` (str, default: ``l2``): ``l1``, ``l2``, ``elasticnet``
    - ``solver`` (str, default: ``lbfgs``): optimizer
    - ``random_state`` (int): random seed

**RandomForest**:

    - ``n_estimators`` (int, default: ``100``): number of trees
    - ``max_depth`` (int, optional): max tree depth
    - ``min_samples_split`` (int, default: ``2``): min samples to split
    - ``min_samples_leaf`` (int, default: ``1``): min samples per leaf
    - ``max_features`` (str/int, default: ``sqrt``): max features per split
    - ``random_state`` (int): random seed

**XGBoost**:

    - ``n_estimators`` (int, default: ``100``): boosting rounds
    - ``max_depth`` (int, default: ``3``): max tree depth
    - ``learning_rate`` (float, default: ``0.1``): learning rate
    - ``subsample`` (float, default: ``1.0``): row subsample ratio
    - ``colsample_bytree`` (float, default: ``1.0``): column subsample ratio
    - ``random_state`` (int): random seed

**SVM**:

    - ``C`` (float, default: ``1.0``): regularization
    - ``kernel`` (str, default: ``rbf``): ``linear``, ``poly``, ``rbf``, ``sigmoid``
    - ``gamma`` (str/float, default: ``scale``): kernel coefficient
    - ``probability`` (bool, default: ``false``): enable probability estimates
    - ``random_state`` (int): random seed

**KNN**:

    - ``n_neighbors`` (int, default: ``5``): number of neighbors
    - ``weights`` (str, default: ``uniform``): ``uniform``, ``distance``
    - ``metric`` (str, default: ``minkowski``): distance metric

**DecisionTree**:

    - ``max_depth`` (int, optional)
    - ``min_samples_split`` (int, default: ``2``)
    - ``min_samples_leaf`` (int, default: ``1``)
    - ``random_state`` (int)

**MLP**:

    - ``hidden_layer_sizes`` (tuple/list, default: ``(100,)``)
    - ``activation`` (str, default: ``relu``)
    - ``max_iter`` (int, default: ``200``)
    - ``random_state`` (int)

**AdaBoost / GradientBoosting**:

    - ``n_estimators`` (int, default: ``100``)
    - ``learning_rate`` (float, default: ``1.0`` for AdaBoost, ``0.1`` for GradientBoosting)
    - ``random_state`` (int)

**GaussianNB / MultinomialNB / BernoulliNB**:

    - mostly sklearn defaults; ``MultinomialNB`` requires non-negative features

**AutoGluon**:

    - ``time_limit`` (int): training time limit (seconds)
    - ``presets`` (str, default: ``medium_quality``): ``best_quality``, ``high_quality``, ``medium_quality``

- **Example**:

  .. code-block:: yaml

     # Train multiple models
     models:
       LogisticRegression:
         params:
           max_iter: 1000
           C: 1.0
           random_state: 42
       
       RandomForest:
         params:
           n_estimators: 200
           max_depth: 10
           random_state: 42
       
       XGBoost:
         params:
           n_estimators: 100
           max_depth: 5
           learning_rate: 0.1
           random_state: 42

**is_visualize**: enable visualization

- **Type**: boolean
- **Default**: ``true``

**visualization**: visualization settings (``VisualizationConfig``)

- ``enabled``: **default** ``true``
- ``plot_types``: **default** ``[roc, dca, calibration, pr, confusion, shap]``; values match listed type names
- ``dpi``: **default** ``600``
- ``format``: **default** ``pdf``

**is_save_model**: save trained pipeline to ``output`` (default ``true``).

**Predict-mode-only fields** (when ``run_mode: predict``)

- ``evaluate`` (bool, default ``false``): compute metrics after prediction if labels exist.
- ``output_label_col`` (default ``predicted_label``): predicted class column in output table.
- ``output_prob_col`` (default ``predicted_probability``): probability column in output table.
- ``probability_class_index``: class index for probability column in multiclass (``None`` = all or per implementation).
- ``binary_positive_class_index`` (default ``1``): positive class index in probability vector for binary classification.

**Model comparison configuration (``habit compare``, ``ModelComparisonConfig``)**

- ``output_dir`` (**required**): summary output directory.
- ``files_config`` (**list**): one entry per model.

  - ``path`` (**required**): prediction CSV/Excel.
  - ``subject_id_col`` / ``label_col`` / ``prob_col`` (**required**).
  - ``pred_col`` / ``split_col`` (optional).
  - ``model_name`` or ``name`` or filename stem inferred from ``path``.

- ``merged_data``: ``enabled``, ``save_name`` (default ``combined_predictions.csv``).
- ``split`` (internal ``SplitConfig``): split after merge (default ``enabled: false``).
- ``visualization``: ``roc`` / ``dca`` / ``calibration`` / ``pr_curve`` sub-blocks with ``enabled``, ``save_name``, ``title``, ``n_bins`` (calibration), etc.
- ``delong_test``: ``enabled``, ``save_name`` (default ``delong_results.json``).
- ``metrics``: ``basic_metrics``, ``youden_metrics``, ``target_metrics`` (includes ``targets`` dict with (0,1) threshold targets).
