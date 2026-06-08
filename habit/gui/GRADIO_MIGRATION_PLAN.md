# HABIT GUI Gradio 迁移与参数补全详细方案（修订版）

> 本版已对照 `habit/core/*/config_schemas.py`、各 Tab 现有 Streamlit 实现、以及 `config/` 下官方 YAML 模板逐项核对。

## 1. 迁移目标
1. **彻底移除 Streamlit**：解决状态管理混乱、页面频繁刷新、长耗时任务导致 UI 卡死的问题。
2. **参数对齐真实 Schema**：以 Pydantic 模型 + 官方 config 模板为准，补全旧 GUI 硬编码/遗漏项。
3. **提升用户体验**：
   - 使用 Gradio `yield` 实现后台任务日志实时输出。
   - 保留 `tkinter` 原生路径选择器。
   - 错误提示中文化（`translate_pydantic_error`）。

## 2. 当前进度

| 模块 | 状态 | 备注 |
|------|------|------|
| `app.py` | ✅ 完成 | 6 Tab 全部接入 Gradio |
| `utils.py` + `pipeline_runner.py` | ✅ 完成 | 共享日志/路径/YAML 工具 |
| `tab_preprocess.py` | ✅ 完成 | 含步骤顺序、elastix、SimpleITK、extra YAML |
| `tab_habitat.py` | ✅ 完成 | 含个体/群体预处理 YAML、checkpoint 高级参数 |
| `tab_dicom_sort.py` | ✅ 完成 | 含 f/dcm2niix_path/extra_args/output_dir |
| `tab_extract.py` | ✅ 完成 | feature_types 全量 |
| `tab_ml.py` | ✅ 完成 | MLConfig、custom split、全模型/selector |
| `tab_compare.py` | ✅ 完成 | split_col、metrics、6 模型槽位 |
| `requirements.txt` | ✅ 完成 | streamlit → gradio |

---

## 3. 各 Tab 参数核对（对照源码）

### Tab 1: DICOM 整理 (`tab_dicom_sort.py`)

**Schema**: `habit.core.dicom_sort.config_schema.DicomSortConfig`

| 字段 | 旧 Streamlit | 当前方案 | 说明 |
|------|-------------|---------|------|
| `data_dir` | ✅ | ✅ | 必填 |
| `out_dir` | ✅ | ✅ | 必填 |
| `f` | ✅（模板选择） | ⚠️ 写成 `filename_format` | Schema 主字段是 **`f`**；`filename_format` 仅为 deprecated 别名 |
| `dcm2niix_path` | ✅（高级折叠） | ❌ 遗漏 | 可选，指定 dcm2niix 可执行文件 |
| `extra_args` | ✅（高级折叠） | ❌ 遗漏 | 逗号分隔的 dcm2niix 附加参数 |
| `output_dir` | ❌ | ❌ 遗漏 | 可选，覆盖 dcm2niix `-o` 目标（不同于 `out_dir`） |
| `processes` | ❌ | ❌ **方案写错** | **DicomSortConfig 中不存在此字段** |

**Gradio 改造要点**：
- 保留命名模板选择 + 自定义 `f` 输入。
- 高级区：`dcm2niix_path`、`extra_args`、`output_dir`。
- 删除方案中错误的 `processes` 描述。

---

### Tab 2: 图像预处理 (`tab_preprocess.py`)

**Schema**: `habit.core.preprocessing.config_schemas.PreprocessingConfig`

#### 3.2.1 顶层字段

| 字段 | 旧 Streamlit | 当前 Gradio | 说明 |
|------|-------------|------------|------|
| `data_dir` / `out_dir` | ✅ | ✅ | — |
| `processes` | ✅ | ✅ | — |
| `auto_select_first_file` | ✅ | ✅ | — |
| `random_state` | ✅（从 loaded） | ✅ | — |
| `preprocessing_input_layout` | ✅（硬编码 habit_default） | ❌ 遗漏 | 目前仅支持 `habit_default` |
| `save_options.save_intermediate` | ✅ | ✅ | — |
| `save_options.intermediate_steps` | ✅（空列表） | ❌ 遗漏 | 可指定只保存某些步骤 |

#### 3.2.2 预处理步骤（`Preprocessing` 字典）

**旧 Streamlit 支持且可调整顺序的步骤**（`DEFAULT_STEP_ORDER`）：
- `n4_correction`, `resample`, `zscore_normalization`, `registration`

**当前 Gradio 问题**：
- ❌ **丢失步骤顺序调整**（▲/▼ 上移下移）；YAML 中 key 插入顺序即执行顺序。
- ❌ **未覆盖** `config_image_preprocessing.yaml` 中记录的其它步骤：
  - `histogram_standardization`
  - `adaptive_histogram_equalization`
  - `reorientation`
  - `dcm2nii`（预处理流水线内转换，区别于 Tab1 独立 sort-dicom）
  - `load_image`

**各步骤参数缺口**：

| 步骤 | 旧 Streamlit 已有 | 当前 Gradio 缺口 |
|------|------------------|-----------------|
| `n4_correction` | images, num_fitting_levels | — |
| `resample` | images, target_spacing | — |
| `zscore_normalization` | images, only_inmask | `mask_key` |
| `registration` | images, fixed_image, type_of_transform, metric, use_mask, replace_by_fixed_image_mask(硬编码) | `replace_by_fixed_image_mask`(用户不可配)；**elastix 全套**：`elastix_parameter_files`, `elastix_path`, `transformix_path`, `elastix_parameter_overrides`, `elastix_threads`；**simpleitk 调参**：`number_of_histogram_bins`, `metric_sampling_percentage`, `shrink_factors_per_level`, `smoothing_sigmas_per_level`, `learning_rate`, `number_of_iterations`, `bspline_mesh_size`, `bspline_order`；`mask_key` |

**Gradio 改造要点**：
- 恢复步骤启用 + **顺序可调** UI。
- `backend` 切换时动态显示：`ants/simpleitk` 配准参数 vs `elastix` 参数文件区。
- `intermediate_steps` 多选（当 `save_intermediate=true`）。

---

### Tab 3: 影像生境聚类分割 (`tab_habitat.py`)

**Schema**: `habit.core.habitat_analysis.config_schemas.HabitatAnalysisConfig`

#### 3.3.1 顶层运行控制（旧 Streamlit 大量遗漏）

| 字段 | 旧 Streamlit | 当前方案 | 优先级 |
|------|-------------|---------|--------|
| `run_mode` / `data_dir` / `out_dir` / `pipeline_path` | ✅ | ✅ | 高 |
| `processes` | ✅ | ✅ | 高 |
| `resume` / `plot_curves` / `random_state` | ✅ | ✅ | 高 |
| `cap_processes_to_gpu_pool` | ❌ | ❌ | 中（GPU radiomics 场景） |
| `individual_subject_timeout_sec` | ❌ | ❌ | 中 |
| `individual_subject_graceful_shutdown_sec` | ❌ | ❌ | 低 |
| `individual_subject_spawn_timeout_sec` | ❌ | ❌ | 低 |
| `on_subject_failure` (continue/fail_fast) | ❌ | ❌ | 中 |
| `oom_backoff` / `oom_reduce_workers_by` | ❌ | ❌ | 中 |
| `strict_checkpoint_hash` | ❌ | ❌ | 中 |
| `checkpoint_dir` | ❌ | ❌ | 中 |
| `force_rerun_subjects` | ❌ | ❌ | 中 |
| `retry_failed_subjects` | ❌ | ❌ | 中 |
| `individual_subject_auto_retry_rounds` | ❌ | ❌ | 中 |
| `individual_subject_parallel_mode` (isolated/persistent) | ❌ | ❌ | 中 |
| `persistent_worker_max_consecutive_failures` | ❌ | ❌ | 低 |
| `persistent_worker_recycle_after_tasks` | ❌ | ❌ | 低 |
| `clear_checkpoint_on_success` | ❌ | ❌ | 低 |
| `save_images` | ❌ | ❌ | 中 |
| `save_results_csv` / `habitats_results_format` (parquet/csv) | 部分硬编码 | ❌ | 中 |
| `verbose` / `debug` | debug 从 loaded 继承 | ❌ | 低 |

#### 3.3.2 `HabitatSegmentation`

| 区块 | 旧 Streamlit | 当前方案缺口 |
|------|-------------|-------------|
| `clustering_mode` | ✅ | — |
| `supervoxel` (two_step) | algorithm, n_clusters | `max_iter`, `n_init`, `random_state`, `compactness`, `sigma`, `enforce_connectivity`（SLIC）；`one_step_settings`（one_step 模式用） |
| `habitat` | algorithm, min/max_clusters, selection_method, fixed_n_clusters | `habitat_cluster_selection_method` 应为 **列表**；`max_iter`, `n_init`, `random_state`, `parallel_cluster_search`, `cluster_search_workers` |
| `postprocess_supervoxel` | ❌ 完全遗漏 | `enabled`, `min_component_size`, `connectivity`, `reassign_method`, `max_iterations` |
| `postprocess_habitat` | ❌ 完全遗漏 | 同上 |

#### 3.3.3 `FeatureConstruction`（旧 Streamlit 最严重遗漏）

| 区块 | 旧 Streamlit | 必须补全 |
|------|-------------|---------|
| `voxel_level.method` | 模板/自定义 | ✅ 保留 |
| `voxel_level.params` | kernelRadius, params_file | 还需：`voxelBatch`, `useTorchRadiomics`, `torchGpus` 等 |
| `supervoxel_level` (two_step) | 硬编码默认值 | 需暴露：`supervoxel_file_keyword`, `method`, `params_file`, `supervoxelUnionBboxCrop`, `useSupervoxelCext`, `supervoxelBatch` 等 |
| **`preprocessing_for_subject_level`** | **硬编码 winsorize+minmax，UI 不可配** | **必须做成可编辑方法列表** |
| **`preprocessing_for_group_level`** | **硬编码 winsorize+minmax，UI 不可配** | **必须做成可编辑方法列表**；官方模板还含 `variance_filter`, `correlation_filter` |

**`PreprocessingMethod` 支持的方法**（`config_schemas.py`）：
`winsorize`, `minmax`, `zscore`, `robust`, `log`, `binning`, `variance_filter`, `correlation_filter`

各方法可选参数：`global_normalize`, `winsor_limits`, `n_bins`, `bin_strategy`, `variance_threshold`, `corr_threshold`, `corr_method`

**重要校验规则**（Schema `model_validator`）：
- `two_step` 模式下，`preprocessing_for_subject_level` **禁止** `variance_filter` / `correlation_filter`（须放到 group_level）。

**Gradio UI 建议**：
- 个体/群体预处理各用「方法多选 + 按方法展开参数」或「固定最多 N 步流水线」。
- `clustering_mode` 切换时联动显示/隐藏 `supervoxel_level`、`supervoxel` 块、`one_step_settings`。
- 高级运行控制折叠到「断点续跑 / 并行 / 超时 / OOM」分组。

---

### Tab 4: 生境特征提取 (`tab_extract.py`)

**Schema**: `habit.core.habitat_analysis.config_schemas.FeatureExtractionConfig`

| 字段 | 旧 Streamlit | 当前方案 | 说明 |
|------|-------------|---------|------|
| `raw_img_folder` | ✅ | ✅ | — |
| `habitats_map_folder` | ✅ | ✅ | — |
| `out_dir` | ✅ | ✅ | — |
| `params_file_of_non_habitat` | ✅ | ✅ | — |
| `params_file_of_habitat` | ✅ | ✅ | — |
| `habitat_pattern` | ✅ | ✅ | — |
| `n_processes` | ✅ | ✅ | — |
| `feature_types` | ✅ multiselect | ✅ | `traditional`, `non_radiomics`, `whole_habitat`, `each_habitat`, `msi`, `ith_score` |
| `n_habitats` | ✅ | ✅ | 0/null = 自动检测 |
| `debug` | 从 loaded 继承 | ⚠️ 需暴露 | 可选 checkbox |

**纠正**：Tab 4 **没有**「患者级 / 超体素级 / 体素级」提取开关；层级语义由 `feature_types` 表达。

---

### Tab 5: 机器学习建模 (`tab_ml.py`)

**Schema**: `habit.core.machine_learning.config_schemas.MLConfig`（**不是** `MachineLearningConfig`）

#### 3.5.1 方案中的错误描述（须删除/更正）

| 原方案写法 | 实际情况 |
|-----------|---------|
| Schema 名 `MachineLearningConfig` | 应为 **`MLConfig`** |
| Scaler: StandardScaler 等 sklearn 类名 | 应为 **`normalization.method`**：`z_score`, `min_max`, `robust`, `max_abs`, `normalizer`, `quantile`, `power` + `params` |
| 特征选择 PCA / RFE / GridSearch | **MLConfig 无 GridSearch/RandomSearch**；特征选择是 **`feature_selection_methods` 列表**，方法名来自 selector registry |
| 仅列 SVM/RF/LR/XGBoost | 后端还支持 **KNN, DecisionTree, AdaBoost, MLP, GradientBoosting, GaussianNB, MultinomialNB, BernoulliNB, AutoGluonTabular** 等 |

#### 3.5.2 完整字段清单

**数据输入 `input[]`**（旧 Streamlit 仅单文件）：
- `path`, `name`, `subject_id_col`, `label_col`
- 遗漏：`features`（显式特征列）, `features_from_log`, `split_col`, `pred_col`

**拆分**：
- `split_method`: random / stratified / **custom**
- `test_size`, `random_state`, `stratified`
- custom 模式：`train_ids_file`, `test_ids_file`（旧 GUI **完全未暴露**）

**K-Fold**（`habit cv`）：
- `n_splits`（旧 GUI 通过工作流类型切换）

**`normalization`**：method + params（各 method 有不同 params，见 `config_machine_learning.yaml`）

**`resampling`**：
- `enabled`, `method`, `ratio`, `random_state`
- 遗漏：`position`（before_feature_selection / before_normalization / after_normalization / before_model）

**`feature_selection_methods`**（旧 GUI 仅 4 种）：

| 旧 GUI 已有 | 代码库还支持但未暴露 |
|------------|-------------------|
| variance | lasso, rfecv, mrmr, stepwise, stepwise_r, anova, chi2, univariate_logistic, icc |

每种 selector 有独立 `params`（如 correlation 的 `method`, `visualize`, `before_z_score` 等）。

**`models`**：字典，key = ModelFactory 注册名，value = `params`。

**标志位**：`is_visualize`, `is_save_model`

**`visualization`**：`enabled`, `plot_types`（roc, dca, calibration, pr, confusion, shap）, `dpi`, `format`

**predict 模式专有**（旧 GUI 基本未暴露）：
- `evaluate`, `output_label_col`, `output_prob_col`
- `probability_class_index`, `binary_positive_class_index`

---

### Tab 6: 模型多指标比对 (`tab_compare.py`)

**Schema**: `habit.core.machine_learning.config_schemas.ModelComparisonConfig`

#### 3.6.1 `files_config[]` 每行字段

| 字段 | 旧 Streamlit | 当前方案 |
|------|-------------|---------|
| `path` | ✅ | ✅ |
| `model_name` / `name` | ✅ | ✅ |
| `subject_id_col` | ✅ | ✅ |
| `label_col` | ✅ | ✅ |
| `prob_col` | ✅ | ✅ |
| `pred_col` | ✅（选填） | ⚠️ 方案初版遗漏 |
| `split_col` | ❌ 未暴露 | ❌ 遗漏（官方模板常用） |

#### 3.6.2 其它配置块（旧 Streamlit 部分硬编码）

| 区块 | 旧 Streamlit | 缺口 |
|------|-------------|------|
| `merged_data` | 硬编码 enabled=true | 应暴露 `enabled`, `save_name` |
| `split` | 硬编码 enabled=false | 应暴露 `enabled`（按 train/test 分组分析） |
| `visualization.roc/dca/calibration` | 3 个 checkbox | 还应暴露 `pr_curve`；各子项 `save_name`, `title`, `n_bins` |
| `delong_test` | enabled checkbox | `save_name` |
| `metrics.basic_metrics` | 硬编码 enabled=true | 应暴露 |
| `metrics.youden_metrics` | 硬编码 enabled=true | 应暴露 |
| `metrics.target_metrics` | 硬编码 enabled=false | 应暴露 `enabled` + `targets` 字典 |

---

## 4. Gradio UI 分层策略（避免表单爆炸）

并非所有 Schema 字段都需平铺在主界面。建议三层：

1. **常用区（默认展开）**：路径、模式、核心算法参数、一键运行。
2. **进阶区（折叠 Panel）**：并行/超时/断点/checkpoint、后处理、可视化格式等。
3. **专家区（YAML 直编或「加载已有配置」）**：
   - 复杂嵌套结构（如 `preprocessing_for_group_level` 多步流水线、`elastix_parameter_overrides`）优先支持 **加载 YAML → 表单回填 → 保存 YAML**，避免在 Gradio 里复刻全部嵌套编辑器。
   - 对极长 tail 参数（如 `persistent_worker_recycle_after_tasks`）可仅在「高级」区提供，或保留 YAML 编辑入口。

---

## 5. 实施顺序（修订）

| 阶段 | 内容 | 验收标准 |
|------|------|---------|
| P0 | 修订 `tab_preprocess.py` 缺口 | 步骤顺序、elastix、intermediate_steps、replace_by_fixed_image_mask |
| P1 | `tab_habitat.py` | FeatureConstruction 个体/群体预处理可配；HabitatSegmentation 补全；checkpoint 相关 |
| P2 | `tab_dicom_sort.py` + `tab_extract.py` | 对照 DicomSortConfig / FeatureExtractionConfig 100% 常用字段 |
| P3 | `tab_ml.py` | MLConfig 完整；custom split；predict 参数；全部 selector/model |
| P4 | `tab_compare.py` | ModelComparisonConfig 完整；split_col；metrics 全套 |
| P5 | 移除 Streamlit 残留；`requirements.txt` 换 gradio | `habit gui` 端到端跑通 demo |

---

## 6. 代码规范（不变）
1. 每个 Tab 独立 `run_pipeline` + `yield` 日志。
2. 统一 `select_local_path`。
3. `ValidationError` → `translate_pydantic_error`。
4. 配置写出前必须过 Pydantic 校验。
5. 图表文字英文（habit 包内绘图规则）。

---

*请确认修订版方案后，按 P0 → P1 顺序执行；优先补全 Tab 2 与 Tab 3 的参数缺口。*
