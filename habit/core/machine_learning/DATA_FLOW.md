# machine_learning 数据流总览（全子模块版）

本文档按“真实执行顺序”解释 `habit/core/machine_learning` 下**所有主要子模块**如何参与数据流。

目标是回答：

1. 数据从哪来，到哪去。
2. 每个子模块在什么时候介入。
3. 每个子模块的输入和输出是什么。

---

## 1. 全链路总图

```text
YAML/Dict
  -> config_schemas (校验并转配置对象)
  -> workflows (编排)
  -> data_manager (读表/合并/切分)
  -> pipeline_utils (组装 selector->scaler->model pipeline)
      -> feature_selectors (筛特征)
      -> models (创建模型)
  -> resampling (可选重采样)
  -> runners (训练/验证/预测主循环)
      -> evaluation (指标/阈值)
      -> statistics (比较流程统计检验)
  -> core (结构化结果合同，仅 train/kfold/predict)
  -> reporting (写模型/CSV/JSON)
  -> visualization (绘图)
```

---

## 2. 全子模块介入时机总表

| 子模块 | 什么时候介入 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `config_schemas.py` | 启动最早 | 原始配置 | `MLConfig` / `ModelComparisonConfig` |
| `__init__.py` | import 时 | 子模块公共对象 | 对外统一 API 导出 |
| `workflows/` | 配置校验后 | 配置对象 | 执行顺序与组件编排 |
| `data_manager.py` | 训练/预测前 | 输入文件配置 | `X/y`、split 数据、inference DataFrame |
| `pipeline_utils.py` | 模型训练前 | 特征名、模型名、模型参数 | sklearn Pipeline |
| `feature_selectors/` | pipeline 训练阶段 | `X/y`、方法名、参数 | 特征子集 |
| `models/` | pipeline 构建阶段 | 模型名、参数 | estimator |
| `resampling.py` | 每次 `fit` 前（可选） | `X_train/y_train` | 重采样后的 `X/y` |
| `runners/` | 执行核心循环时 | context + plan (+ X/y) | 运行结果对象 |
| `core/` | runner 结束时（train/kfold/predict） | 计算结果 | `RunResult/KFoldRunResult/InferenceResult` |
| `evaluation/` | 训练评估与比较评估阶段 | `y_true/y_prob/y_pred` | 指标与阈值结果 |
| `statistics/` | compare 的统计检验阶段 | 标签与概率数组 | DeLong/校准检验统计量 |
| `reporting/` | 执行末尾落盘 | core 结果对象或比较指标对象 | `*.pkl/*.csv/*.json` |
| `visualization/` | 报告阶段 | 预测 payload + 配置 | `*.pdf` 图 |

---

## 3. 四条主流程（全模块视角）

## 3.1 Holdout Train

```text
MLConfigurator.create_ml_workflow()
  -> HoldoutWorkflow.__init__()
      -> WorkflowPlan(core) + RunnerContext
  -> HoldoutWorkflow.fit()
      -> DataManager.load_data() / split_data()
      -> HoldoutRunner.run()
          -> PipelineBuilder.build()
              -> FeatureSelectTransformer (feature_selectors)
              -> ModelFactory.create_model (models)
          -> Resampler.fit_with_resampling() (enabled 时)
          -> calculate_metrics() (evaluation)
          -> DatasetSnapshot + RunResult (core)
      -> ModelStore.save() (reporting)
      -> ReportWriter.write() (reporting)
      -> PlotComposer.render() -> PlotManager (visualization)
```

产物：

- `models/<model_name>_final_pipeline.pkl`
- `ml_standard_summary.csv`
- `ml_standard_results.json`
- `all_prediction_results.csv`
- `standard_train_*.pdf` / `standard_test_*.pdf`

## 3.2 KFold CV

```text
MLConfigurator.create_kfold_workflow()
  -> KFoldWorkflow.run()
      -> _load_and_prepare_data() (data_manager)
      -> KFoldRunner.run(X, y)
          -> KFold/StratifiedKFold split
          -> 每折训练: pipeline_utils + feature_selectors + models + resampling
          -> 每折评估: evaluation
          -> 聚合: AggregatedModelResult (core)
          -> 返回 KFoldRunResult (core)
      -> ModelStore.save_kfold_ensembles() (reporting)
      -> ReportWriter.write() (reporting)
      -> PlotComposer.render() (visualization)
```

产物：

- `models/<model_name>_ensemble_final.pkl`
- `ml_kfold_summary.csv`
- `ml_kfold_results.json`
- `kfold_*.pdf`

## 3.3 Predict Inference

```text
HoldoutWorkflow.run(run_mode=predict)
  -> InferenceRunner.run()
      -> 校验 pipeline_path / input path
      -> DataManager.load_inference_data()
      -> joblib.load(pipeline)
      -> predict / predict_proba
      -> optional calculate_metrics (evaluation)
      -> InferenceResult (core)
  -> ReportWriter.write(InferenceResult) (reporting)
```

产物：

- `prediction_results.csv`
- `evaluation_metrics.csv`（仅 evaluate 且有标签时）

## 3.4 Compare（模型比较）

> 注意：比较流是独立通道，**不走 core 的 RunResult 合同**。

```text
MLConfigurator.create_model_comparison()
  -> ModelComparison.setup()
      -> MultifileEvaluator.read_prediction_files() (evaluation)
      -> 合并预测表与 split 分组
  -> ModelComparison.run_evaluation()
      -> basic/youden/target 指标 (evaluation.metrics)
      -> threshold transfer (evaluation.threshold_manager)
      -> DeLong 等统计检验 (statistics)
      -> 绘图 (visualization)
      -> MetricsStore 聚合 (reporting.report_exporter)
  -> ReportExporter.merge_and_save_metrics()
```

产物：

- `metrics/metrics.json`
- `delong_results.json`
- `roc_curves.pdf`、`decision_curves.pdf`、`calibration_curves.pdf`、`precision_recall_curves.pdf`

---

## 4. 重点解释：`core` 只是全链路中的一层

你前面提到“不只是 core”，这里明确定位：

- `core` 的职责是“结构化交接”，不是训练/评估/落盘本身。
- train/kfold/predict 三条流中，`core` 都在 runner 结束后接管结果。
- compare 流不依赖 `core` 结果对象，而是直接走 `evaluation + statistics + report_exporter`。

也就是说：

- 没有 `core`，train/kfold/predict 的跨层接口会变乱。
- 但没有 `data_manager/pipeline_utils/models/evaluation/reporting`，流程同样跑不起来。

---

## 5. 子模块更细粒度说明（目录级）

## 5.1 `feature_selectors/` 子目录

- `selector_registry.py`：方法注册、查询、执行中心。
- `transformer.py`：和 sklearn pipeline 的桥接器。
- `icc/`：测试-重测 ICC 分析工具链（含标签映射与 ICC 计算）。

## 5.2 `runners/` 子目录

- `context.py`：RunnerContext 依赖注入容器。
- `base.py`：runner 公共能力。
- `holdout.py` / `kfold.py` / `inference.py`：三条执行主循环。

## 5.3 `reporting/` 子目录

- `model_store.py`：模型文件持久化。
- `report_writer.py`：train/kfold/predict 报告落盘。
- `plot_composer.py`：图生成调度。
- `report_exporter.py`：compare 流指标结果导出。

## 5.4 `core/` 子目录

- `plan.py`：`WorkflowPlan`（运行快照）。
- `dataset.py`：`DatasetSnapshot`（数据快照）。
- `results.py`：各种结果 dataclass。
- `protocols.py`：`WorkflowResult` 协议。

---

## 6. 改动定位速查（按问题类型）

- 配置字段/校验规则：`config_schemas.py`
- 数据读表/合并/切分：`data_manager.py`
- pipeline 结构：`pipeline_utils.py`
- 特征筛选算法：`feature_selectors/`
- 模型接入与注册：`models/`
- 类别不平衡处理：`resampling.py`
- 训练/验证主循环：`runners/`
- 结果结构定义：`core/`
- 指标和阈值逻辑：`evaluation/`
- 统计显著性检验：`statistics/`
- 报告和模型落盘：`reporting/`
- 图生成逻辑：`visualization/`
# machine_learning 数据流总览（细化版）

本文档专门回答三个问题：

1. 数据从哪里来，经过哪些模块，最终变成什么文件。
2. 每个子模块在流程中的具体介入时机。
3. `core` 契约层到底在什么时候被创建、什么时候被消费。

---

## 1. 一张图看全链路

```text
配置(MLConfig / ModelComparisonConfig)
  -> workflows (编排)
  -> runners (训练/验证/预测执行)
  -> core (结构化结果契约)
  -> reporting (模型/报告/图落盘)
  -> 输出文件(*.pkl, *.csv, *.json, *.pdf)
```

更细一点：

```text
input tables
  -> data_manager (读取/合并/切分)
  -> pipeline_utils (构建 sklearn pipeline: selector -> scaler -> model)
  -> models + feature_selectors + evaluation
  -> core.results (RunResult/KFoldRunResult/InferenceResult)
  -> reporting + visualization
```

---

## 2. 模块职责与介入时机（按执行顺序）

### 2.1 `workflows`（先介入）

- 入口层编排：决定跑 `train`、`predict`、`kfold` 还是 `compare`。
- 负责创建 `WorkflowPlan`、runner、reporting 组件。
- 不做模型公式计算，不写指标公式，只负责“接线与顺序”。

典型触发：

- `HoldoutWorkflow.run()`：根据 `run_mode` 分发到 `fit()` 或 `predict()`。
- `KFoldWorkflow.run()`：固定执行 K 折流程。

### 2.2 `data_manager`（训练/验证前置）

- 读取 `config.input` 的表格文件（csv/tsv/xlsx）。
- 多表合并、重复列重命名、标签列回挂、train/test 切分。
- 推理模式下用 `load_inference_data()` 读单文件。

关键输出：

- holdout：`X_train/X_test/y_train/y_test`
- kfold：完整 `X/y`
- predict：原始推理 DataFrame

### 2.3 `pipeline_utils` + `feature_selectors` + `models`（训练核心）

- `PipelineBuilder` 构建统一 sklearn Pipeline：
  - `selector_before`（标准化前特征筛选）
  - `scaler`
  - `selector_after`（标准化后特征筛选）
  - `model`（`ModelFactory` 创建）
- `feature_selectors` 通过注册机制执行具体筛选算法。
- `models` 通过工厂统一创建各模型实例。

### 2.4 `runners` + `evaluation`（执行与评估）

- `HoldoutRunner`：训练 + train/test 预测 + 指标计算。
- `KFoldRunner`：逐折训练评估 + 跨折聚合。
- `InferenceRunner`：加载已保存 pipeline，生成预测，可选评估。
- `evaluation` 提供 `PredictionContainer`、指标计算、阈值策略。

### 2.5 `core`（在 runner 结束时介入，作为中间总线）

`core` 不是计算模块，它在“runner 输出结果”这一刻介入：

- `WorkflowPlan`：在 workflow 初始化时创建（运行快照）。
- `DatasetSnapshot`：在 holdout runner 结束时创建（训练/测试数据快照）。
- `RunResult` / `KFoldRunResult` / `InferenceResult`：runner 最终返回值。
- `to_legacy_results()`：给旧逻辑提供历史字典视图。

可以理解为：`core` 把“运行产物”标准化，然后交给 reporting。

### 2.6 `reporting` + `visualization`（最后介入，负责落盘）

- `ModelStore`：保存模型 (`*.pkl`)
- `ReportWriter`：保存 summary/results/prediction csv/json
- `PlotComposer`：调 `PlotManager` 生成图（pdf）
- `visualization`：具体绘图实现

---

## 3. Holdout（train）详细数据流

## 3.1 时间线

```text
MLConfigurator.create_ml_workflow()
  -> HoldoutWorkflow.__init__()
      -> 创建 WorkflowPlan (core.plan)
      -> 创建 HoldoutRunner / InferenceRunner
  -> HoldoutWorkflow.fit()
      -> DataManager.load_data()
      -> HoldoutRunner.run()
          -> split_data()
          -> PipelineBuilder.build()
          -> 模型训练 + 预测 + calculate_metrics()
          -> 组装 DatasetSnapshot + RunResult (core.results)
      -> ModelStore.save(RunResult)
      -> ReportWriter.write(RunResult)
      -> PlotComposer.render(RunResult)
```

## 3.2 每个子模块在此流程中的作用

- `workflows`：决定顺序（先算，后存，后画图）。
- `data_manager`：产出 train/test split。
- `pipeline_utils`：构建每个模型 pipeline。
- `feature_selectors`：在 pipeline 中筛选特征。
- `models`：提供最终 estimator。
- `evaluation`：计算 train/test 指标。
- `core`：将计算结果封装成 `RunResult`。
- `reporting`：把 `RunResult` 写成文件。
- `visualization`：绘制标准 train/test 曲线图。

## 3.3 典型输出文件

- `models/<model_name>_final_pipeline.pkl`
- `ml_standard_summary.csv`
- `ml_standard_results.json`
- `all_prediction_results.csv`
- `standard_train_*.pdf` / `standard_test_*.pdf`

---

## 4. KFold（cv）详细数据流

## 4.1 时间线

```text
MLConfigurator.create_kfold_workflow()
  -> KFoldWorkflow.__init__()
      -> 创建 WorkflowPlan (core.plan)
      -> 创建 KFoldRunner
  -> KFoldWorkflow.run()
      -> _load_and_prepare_data() 得到 X/y
      -> KFoldRunner.run(X, y)
          -> build_splitter(KFold/StratifiedKFold)
          -> 每折: build pipeline -> train -> predict -> metrics
          -> 跨折聚合 overall_metrics/AUC mean/std
          -> 组装 KFoldRunResult (core.results)
      -> ModelStore.save_kfold_ensembles(KFoldRunResult)
      -> ReportWriter.write(KFoldRunResult)
      -> PlotComposer.render(KFoldRunResult)
```

## 4.2 `core` 在 KFold 中的关键作用

- `KFoldModelResult`：保存单模型的逐折 payload 与每折 estimator。
- `AggregatedModelResult`：保存跨折拼接预测与汇总指标。
- `KFoldRunResult`：统一承载 `models + aggregated + summary_rows`。
- `results` / `to_legacy_results()`：向旧代码暴露历史结构。

## 4.3 典型输出文件

- `models/<model_name>_ensemble_final.pkl`
- `ml_kfold_summary.csv`
- `ml_kfold_results.json`
- `kfold_*.pdf`

---

## 5. Predict（inference）详细数据流

## 5.1 时间线

```text
HoldoutWorkflow.run() with run_mode=predict
  -> HoldoutWorkflow.predict()
      -> InferenceRunner.run()
          -> 校验 pipeline_path / input path
          -> DataManager.load_inference_data()
          -> joblib.load(pipeline)
          -> predict / predict_proba
          -> (可选) calculate_metrics
          -> 组装 InferenceResult (core.results)
      -> ReportWriter.write(InferenceResult)
```

## 5.2 模块介入说明

- `workflows`：负责入口分发。
- `runners/inference`：负责预测全过程。
- `core`：用 `InferenceResult` 标准化预测输出。
- `reporting`：写 `prediction_results.csv`（有指标时写 `evaluation_metrics.csv`）。

## 5.3 典型输出文件

- `prediction_results.csv`
- `evaluation_metrics.csv`（仅当 `evaluate=True` 且能解析标签列）

---

## 6. Compare（多模型比较）数据流

`comparison_workflow.py` 是独立编排流，不走 `core` 结果契约。

## 6.1 时间线

```text
MLConfigurator.create_model_comparison()
  -> ModelComparison.setup()
      -> MultifileEvaluator.read_prediction_files()
      -> 合并数据与按 split 分组
  -> ModelComparison.run_evaluation()
      -> 绘图 (Plotter/PlotManager)
      -> DeLong 检验
      -> basic/youden/target 指标计算与阈值迁移
      -> MetricsStore 聚合
  -> ReportExporter.merge_and_save_metrics()
```

## 6.2 涉及模块

- `evaluation`：指标与阈值逻辑。
- `statistics`：DeLong 等统计检验。
- `visualization`：比较图输出。
- `reporting/report_exporter.py`：比较结果导出。

---

## 7. 为什么 `core` 对 train/kfold/predict 很关键

## 7.1 统一输出合同

- holdout、kfold、predict 三条执行路径都输出结构化 Result。
- reporting 不需要知道“这个结果来自哪种 runner”，只要识别类型即可处理。

## 7.2 兼容旧代码

- 新结构用于严格测试和后续演进。
- `to_legacy_results()` 提供历史 dict 结构，减少迁移成本。

## 7.3 降低耦合

- runner 只负责计算，不负责文件落盘。
- reporting 只负责落盘，不参与训练逻辑。
- workflow 只编排顺序，不承载大量业务细节。

---

## 8. 按模块看“输入 -> 输出”速查表


| 模块                  | 主要输入                   | 主要输出                                       |
| ------------------- | ---------------------- | ------------------------------------------ |
| `workflows`         | 配置对象                   | 调用 runner 与 reporting 的执行顺序                |
| `data_manager`      | 输入表格配置                 | 训练数据、切分数据、推理数据                             |
| `pipeline_utils`    | 模型名、模型参数、特征名           | sklearn Pipeline                           |
| `feature_selectors` | `X/y`、候选特征、selector 参数 | 保留特征列表                                     |
| `models`            | 模型名+参数                 | estimator                                  |
| `runners`           | context+plan(+X/y)     | `RunResult/KFoldRunResult/InferenceResult` |
| `core`              | runner 计算结果            | 标准化结果契约 + legacy 适配                        |
| `evaluation`        | `y_true/y_prob/y_pred` | 指标与阈值相关结果                                  |
| `statistics`        | 标签+概率                  | 显著性检验统计量与 `p` 值                            |
| `reporting`         | core 结果对象              | `*.pkl/*.csv/*.json`                       |
| `visualization`     | 预测 payload + 绘图配置      | `*.pdf` 图文件                                |


---

## 9. 维护时的实用判断

- 想改“训练逻辑/折内计算” -> 优先看 `runners`。
- 想改“字段结构/跨模块数据合同” -> 先改 `core`。
- 想改“落盘文件命名/格式” -> 看 `reporting`。
- 想改“图样式/图类型” -> 看 `visualization`。
- 想改“指标公式/阈值策略” -> 看 `evaluation`（必要时 `statistics`）。

