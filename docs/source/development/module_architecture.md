# 模块代码架构说明

本文档面向 HABIT 开发者，用于快速理解各主要模块的代码组织、调用链、数据契约和扩展点。

整体依赖规则与包级拓扑见同目录 Sphinx 源码 [architecture.rst](architecture.rst)；本文更关注「读代码时从哪里开始、改功能时改哪里」。若需网页版渲染，请以 `architecture.rst` 与 `module_architecture.rst` 为准构建文档。

---

## 总览

HABIT 的主干能力可以按下面的数据链理解：

```
CLI / Python API
  -> Domain Configurator
  -> Domain Workflow / Pipeline
  -> CSV / NRRD / PKL artifacts
  -> downstream workflow
```

三个业务域之间**不直接** `import` 彼此的内部实现：

- `habit.core.preprocessing`：图像预处理，输出标准化影像。
- `habit.core.habitat_analysis`：habitat 分割、habitat map 与特征表。
- `habit.core.machine_learning`：读取 CSV 特征并训练、预测、比较模型。

它们通过**文件产物**衔接，而不是通过跨域对象引用衔接。跨域共享能力放在 `habit.core.common` 和 `habit.utils`。

---

## 入口与装配层

### 关键文件

| 文件 | 职责 |
|------|------|
| `habit/cli.py` | Click 根入口，集中注册所有 `habit <subcommand>` 命令。 |
| `habit/cli_commands/commands/cmd_*.py` | 业务命令的薄封装：加载配置、创建 configurator、调用业务对象。 |
| `habit/core/common/configurators/base.py` | configurator 公共基类，处理 logger、输出目录和轻量服务缓存。 |
| `habit/core/common/configurators/habitat.py` | 装配 habitat 分析、habitat 特征抽取、传统 radiomics、test-retest。 |
| `habit/core/common/configurators/ml.py` | 装配 ML workflow、KFold、模型比较、评估、报告、绘图组件。 |
| `habit/core/common/configurators/preprocessing.py` | 装配图像预处理 `BatchProcessor`。 |

### 标准调用链

```
habit <command> -c config.yaml
  -> habit/cli.py
  -> cmd_<domain>.run_*
  -> ConfigClass.from_file(...)
  -> DomainConfigurator(...)
  -> create_<service>()
  -> service.run / fit / predict / process_batch
```

### 命令组织

有两类：

- **业务流程命令**：`preprocess`、`get-habitat`、`extract`、`model`、`cv`、`compare`、`icc`、`radiomics`、`retest`。这些命令通常走 YAML 配置和 domain configurator。
- **辅助工具命令**：`dicom-info`、`merge-csv`、`dice`。主要由命令行参数驱动；`dicom-info` 和 `merge-csv` 有对应 `cmd_*.py`，`dice` 直接从 `habit/cli.py` 调用 `habit.utils.dice_calculator`。

### 维护注意事项

- 新增业务命令：优先新增 `cmd_<name>.py`，再在 `habit/cli.py` 注册。
- 新增需要 YAML 的业务服务：先定义 `BaseConfig` 子类，再在对应 domain configurator 添加 `create_*`。
- configurator 内部的业务 `import` 应放在 factory 方法内部，避免 `habit.core.common` **顶层** `import` 业务子包。

---

## `habit.core.preprocessing`

### 模块定位

`preprocessing` 子包负责把原始 DICOM/NIfTI 数据变成后续 habitat 或 radiomics 可直接读取的标准化影像。它是**图像层面**的批处理 pipeline。

不要与 `habit.core.habitat_analysis` 管线内的 subject-level / group-level **特征**预处理混淆。

### 关键文件

| 文件 | 职责 |
|------|------|
| `config_schemas.py` | `PreprocessingConfig` 和每个步骤配置的 Pydantic schema。 |
| `image_processor_pipeline.py` | `BatchProcessor`，按 subject 并行调度整条预处理链。 |
| `base_preprocessor.py` | `BasePreprocessor`，所有具体预处理器的统一接口。 |
| `preprocessor_factory.py` | `PreprocessorFactory`，通过注册名创建具体 preprocessor。 |
| `load_image.py` | `LoadImagePreprocessor`，把路径读取成 `SimpleITK.Image`。 |
| `dcm2niix_converter.py` | `Dcm2niixConverter`，调用外部 dcm2niix。 |
| `resample.py` / `registration.py` / `n4_correction.py` 等 | 具体预处理步骤，均通过 factory 注册。 |

### 数据流

```
PreprocessingConfig
  -> PreprocessingConfigurator.create_batch_processor()
  -> BatchProcessor.process_batch()
  -> LoadImagePreprocessor
  -> PreprocessorFactory.create(step_name)
  -> BasePreprocessor.__call__(subject_data)
  -> processed images on disk
```

`BatchProcessor` 的核心循环是按 subject 处理。每个 subject 的数据放在 `subject_data` 字典中，常见键包括：

- `subj`：subject ID。
- 影像 modality 键，例如 `delay2`、`t1`、`flair`。
- mask 键，例如 `mask_delay2` 或配置中指定的 mask 名。
- `output_dirs`：当前 subject 的输出目录集合。

配置中的 `Preprocessing` 字段是有序的「步骤名 → 参数」映射。步骤名必须等于 `PreprocessorFactory` 中注册的名字；每个步骤通常通过 `images` 指定要处理的 modality。

### 扩展点：新增预处理步骤

1. 新建继承 `BasePreprocessor` 的类。
2. 实现 `__call__(self, data: Dict[str, Any]) -> Dict[str, Any]`。
3. 用 `PreprocessorFactory.register("step_name")` 注册。
4. 确保模块会被 `habit.core.preprocessing` 导入，否则注册装饰器不会执行。
5. 在配置模板或用户文档中补充该 `step_name` 的参数说明。

### 维护注意事项

- `LoadImagePreprocessor` 常作为 `BatchProcessor` 的**隐式**前置步骤，用户 YAML 通常不需要写 `load_image`。
- 预处理器应尽量只读写自己声明处理的键。
- 进度条统一使用 `habit.utils.progress_utils.CustomTqdm` 或经 `parallel_utils` 间接使用。

---

## `habit.core.habitat_analysis`

### 模块定位

`habit_analysis` 子包负责从多模态影像和 ROI mask 中生成 habitat map、habitat 标签表和可复用的训练 pipeline；也包含 habitat map 后续特征抽取和传统 radiomics 抽取的实现。

### 关键文件

| 文件 | 职责 |
|------|------|
| `habitat_analysis.py` | `HabitatAnalysis` deep module：`build`、`fit`、`predict`、持久化、结果后处理。 |
| `config_schemas.py` | `HabitatAnalysisConfig`、`FeatureExtractionConfig`、`RadiomicsConfig` 等。 |
| `pipelines/base_pipeline.py` | `HabitatPipeline`、`BasePipelineStep`、`IndividualLevelStep`、`GroupLevelStep`。 |
| `pipelines/steps/*.py` | 体素特征、subject 预处理、个体聚类、supervoxel 聚合、群体聚类等步骤。 |
| `managers/feature_manager.py` | 特征抽取、特征预处理、supervoxel 级特征计算。 |
| `managers/clustering_manager.py` | 聚类算法、最佳聚类数选择、训练状态。 |
| `managers/result_manager.py` | habitat / supervoxel 图像和 CSV 落盘。 |
| `algorithms/` | KMeans、GMM、DBSCAN、SLIC 等聚类策略实现。 |
| `extractors/` | voxel / supervoxel 级特征抽取器与表达式解析。 |
| `analyzers/` | 已生成 habitat map 上的特征提取、传统 radiomics 等。 |

### Pipeline 结构

`HabitatAnalysis` 通过 `_PIPELINE_RECIPES` 按 `HabitatsSegmention.clustering_mode` 选择 recipe：

- `two_step`：voxel → supervoxel → population habitat。
- `one_step`：每个 subject 内 voxel → habitat。
- `direct_pooling`：跨 subject pooling 后统一聚类。

recipe 返回 `(name, step)` 列表，交给 `HabitatPipeline`。pipeline 自动把 step 分成两类：

- `IndividualLevelStep`：逐 subject 处理，可并行。
- `GroupLevelStep`：所有 subject 汇总后处理。

### 训练数据流

```
images + masks
  -> VoxelFeatureExtractor
  -> SubjectPreprocessingStep
  -> IndividualClusteringStep
  -> CalculateMeanVoxelFeaturesStep / SupervoxelFeatureExtractionStep
  -> MergeSupervoxelFeaturesStep
  -> CombineSupervoxelsStep / ConcatenateVoxelsStep
  -> GroupPreprocessingStep
  -> PopulationClusteringStep
  -> habitats.csv + habitat maps + habitat_pipeline.pkl
```

### 预测数据流

预测不会重新 `fit` pipeline：

1. `HabitatPipeline.load(pipeline_path)` 读取训练产物。
2. `HabitatAnalysis._inject_managers_into_pipeline` 用 `_PIPELINE_MANAGER_ATTRS` 白名单注入当前运行时 manager。
3. `FeatureManager` 更新当前数据路径与日志目标，保留已训练状态。
4. 调用 `pipeline.transform(...)`。

### 扩展点

**新增 clustering mode**

- 新增 recipe 函数，返回明确 step 列表。
- 把模式名加入 `_PIPELINE_RECIPES`。
- 新的输出后处理尽量集中在 `HabitatAnalysis` 或 `ResultManager`，不要在 CLI 层分叉。

**新增 pipeline step**

- 继承 `IndividualLevelStep` 或 `GroupLevelStep`。
- 明确 `transform` 的输入/输出结构。
- 需要共享服务时通过构造函数接收 manager，不要在 step 内重新创建 manager。

### 维护注意事项

- 旧 `strategies/` 与 `pipeline_builder.py` 已删除，不要沿用旧三层结构。
- manager 注入使用显式白名单 `_PIPELINE_MANAGER_ATTRS`；新增 manager 必须同步修改白名单与注入逻辑。
- 更细说明：`habit/core/habitat_analysis/ARCHITECTURE.md`、`habit/core/habitat_analysis/PIPELINE_DESIGN.md`。

---

## `habit.core.machine_learning`

### 模块定位

`machine_learning` 子包读取 CSV 特征表，完成训练、预测、K 折验证、模型比较、统计检验、绘图与报告。不关心 CSV 来自 habitat、传统 radiomics 还是临床表。

### 关键文件

| 文件 | 职责 |
|------|------|
| `config_schemas.py` | `MLConfig`、模型配置、特征选择配置、比较配置等。 |
| `base_workflow.py` | `BaseWorkflow`：配置、数据管理、pipeline builder、callbacks 骨架。 |
| `workflows/holdout_workflow.py` | `MachineLearningWorkflow`：训练与预测主 workflow。 |
| `workflows/kfold_workflow.py` | `MachineLearningKFoldWorkflow`。 |
| `workflows/comparison_workflow.py` | `ModelComparison`。 |
| `data_manager.py` | 读取、合并、切分 CSV。 |
| `pipeline_utils.py` | `PipelineBuilder`、`FeatureSelectTransformer`。 |
| `models/` | `ModelFactory` 与具体模型。 |
| `feature_selectors/` | selector 注册表、`run_selector`、ICC 等。 |
| `evaluation/`、`visualization/`、`reporting/`、`callbacks/` | 评估、绘图、报告、回调。 |

### 训练调用链

```
MLConfig
  -> MLConfigurator.create_ml_workflow()
  -> MachineLearningWorkflow.run_pipeline()
  -> DataManager
  -> PipelineBuilder
  -> FeatureSelectTransformer
  -> scaler / imputer
  -> ModelFactory.create_model()
  -> callbacks
  -> model PKL + reports + figures
```

`PipelineBuilder` 把 YAML 配置转为 sklearn `Pipeline`。常见顺序：

```
imputer
  -> feature selection before scaling
  -> scaler
  -> feature selection after scaling
  -> model
```

`FeatureSelectTransformer` 在 sklearn pipeline 内调用 `run_selector`；selector 注册表按函数签名注入 `X`、`y`、`selected_features`、`outdir` 等。

### 预测和比较

- 预测模式仍使用 `MachineLearningWorkflow`，由 `MLConfig.run_mode` 分发；加载 `*_final_pipeline.pkl` 输出预测（及可选评估）。
- `ModelComparison` 不训练模型；组合 `MultifileEvaluator`、`Plotter`、`ThresholdManager`、`ReportExporter`。

### 扩展点

**新增模型**

1. 继承项目模型基类或兼容 sklearn estimator。
2. `ModelFactory.register("model_name")` 注册。
3. 在配置 schema 或模板中补充参数。
4. 可选依赖保持 import 边界清晰。

**新增特征选择**

1. 在 `feature_selectors` 中实现 selector。
2. 加入 `selector_registry`。
3. 明确属于 scaling 前还是 scaling 后。
4. 输出可被 `FeatureSelectTransformer` 消费。

### 维护注意事项

- 训练与预测共用 `MLConfig` 与 `MachineLearningWorkflow`，不要恢复独立 `PredictionWorkflow`。
- **绘图文字必须用英文**（项目约定）。
- 评估/绘图/报告尽量放在对应子包或 callback，workflow 只做编排。

---

## `habit.core.common`

### 模块定位

`common` 是 core 内共享基础设施：配置、服务装配、少量横切工具。可被业务域使用，但**不应**在模块顶层 `import` 任一业务域的重型实现。

### 关键文件

| 文件 | 职责 |
|------|------|
| `config_base.py` | `BaseConfig`、`from_file` / `from_dict` / `to_dict`。 |
| `config_loader.py` | YAML/JSON 读写与路径解析。 |
| `config_validator.py` | 历史兼容校验入口。 |
| `config_accessor.py` | dict 与 Pydantic 访问辅助。 |
| `configurators/` | 域专用服务装配入口。 |
| `dependency_injection.py` | 通用 DI 容器（当前非主路径）。 |
| `dataframe_utils.py` | 表格清洗辅助。 |

### 边界

- `config_loader`：配置文件如何读、路径如何解析。
- `BaseConfig` 与各 domain schema：配置结构是否合法。
- `configurators`：已验证配置如何组装成可运行服务。

不要把带领域语义的运行逻辑塞进 `common`。

---

## `habit.utils`

### 模块定位

`utils` 是全包横切工具层，**不依赖** `habit.core`，可被 preprocessing、habitat、ML、CLI 辅助工具共同使用。

### 常见工具

| 模块 | 职责 |
|------|------|
| `progress_utils.py` | 统一进度条 `CustomTqdm`。 |
| `parallel_utils.py` | 并行 map，整合进度条与错误收集。 |
| `log_utils.py` | 日志初始化与子进程日志恢复。 |
| `io_utils.py` | 通用 I/O、image/mask 路径扫描。 |
| `dicom_utils.py` | DICOM 元信息扫描与提取。 |
| `dice_calculator.py` | Dice 批量计算。 |
| `visualization_utils.py` / `visualization.py` | 绘图风格与可视化辅助。 |
| `habitat_postprocess_utils.py` | habitat map 连通域等后处理。 |
| `file_system_utils.py` / `import_utils.py` / `image_converter.py` | 文件、动态 import、格式转换。 |

### 维护注意事项

- 开发 `habit` 包时，通用工具统一放在 `habit/utils`。
- 进度条用 `CustomTqdm` 或 `parallel_utils`，不要直接引入 `tqdm`。
- 绘图与图中标签用英文。
- `utils` 不要 `import` 业务 domain；若工具出现领域耦合，迁回对应子包。

---

## 辅助工具命令

| 命令 | 实现位置 | 说明 |
|------|----------|------|
| `habit dicom-info` | `cmd_dicom_info.py` + `habit/utils/dicom_utils.py` | 扫描 DICOM、列出或导出指定 tag。 |
| `habit merge-csv` | `cmd_merge_csv.py` | 按索引列横向合并 CSV/Excel。 |
| `habit dice` | `habit/cli.py` + `habit/utils/dice_calculator.py` | 批量 Dice。 |

以上命令不走 domain configurator；若后续需要复杂装配与 YAML，可迁至 `cmd_*.py` + schema + configurator 的标准模式。

---

## 开发者阅读顺序

1. 读 [architecture.rst](architecture.rst)，掌握全局依赖方向。
2. 按要改的功能选 domain：
   - 图像预处理：`BatchProcessor`、`PreprocessorFactory`。
   - habitat 分割：`HabitatAnalysis`、`_PIPELINE_RECIPES`。
   - ML 建模：`MachineLearningWorkflow`、`BaseWorkflow`、`PipelineBuilder`。
3. 再深入具体 step、model、selector、extractor。
4. 最后看配置 schema 与配置模板，确认用户可配置面。

维护时优先保持「入口薄、domain 内聚、domain 间通过文件契约解耦」。
