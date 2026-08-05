# workflows 子模块说明

`workflows` 是机器学习模块的“流程编排层”，负责组织运行顺序与阶段衔接，不负责模型细节实现。

## 为什么这个模块有必要

同一套机器学习能力通常要支持 `train/predict/kfold/compare` 多种入口。
`workflow` 的必要性在于把这些入口统一成稳定的对外 API，并保证执行顺序一致：
先跑计算，再做持久化与可视化。

## 主要职责

1. **统一入口与运行模式分发**
   - 按 `run_mode`（如 `train` / `predict`）分发执行路径。
   - 保持 CLI 与外部调用入口稳定。

2. **构建并注入运行上下文**
   - 在 `BaseWorkflow` 中完成配置校验、日志初始化、依赖对象创建（`DataManager`、`PipelineBuilder`、`Resampler` 等）。
   - 构建 `RunnerContext` 交给 runner 执行。

3. **串联执行与产物输出**
   - 调 runner 执行训练/验证/预测核心循环。
   - 调 reporting 层执行模型保存、报告写入、图形渲染。

4. **提供兼容层**
   - 保留历史类名（如 `MachineLearningWorkflow`）的兼容别名，降低迁移成本。

## 它是怎么起作用的

1. `BaseWorkflow` 初始化配置、日志和公共依赖（`DataManager`、`PipelineBuilder`、`Resampler`）。
2. 具体 workflow 组装 `WorkflowPlan` 和 runner。
3. 调 runner 产出结构化结果。
4. 顺序调用 `ModelStore`、`ReportWriter`、`PlotComposer` 产生产物。

## 具体例子

### 例子 1：HoldoutWorkflow 训练模式

- `run_mode=train` 时，`HoldoutWorkflow.fit()` 调 `HoldoutRunner.run()`。
- 拿到 `RunResult` 后，按顺序执行：
  `ModelStore.save()` -> `ReportWriter.write()` -> `PlotComposer.render()`。

### 例子 2：HoldoutWorkflow 预测模式

- `run_mode=predict` 时，`HoldoutWorkflow.predict()` 调 `InferenceRunner.run()`。
- runner 返回 `InferenceResult`，再由 `ReportWriter` 输出 `prediction_results.csv`。

## 关键文件与作用

- `base.py`
  - 定义 `BaseWorkflow`。
  - 负责共用基础设施与跨流程公共 helper（数据加载、配置验证等）。

- `holdout_workflow.py`
  - `HoldoutWorkflow`：标准 train/test 流程编排。
  - 支持训练与预测两种主路径。

- `kfold_workflow.py`
  - `KFoldWorkflow`：k-fold 交叉验证流程编排。
  - 串联折内执行、聚合结果、落盘与绘图。

- `comparison_workflow.py`
  - 多模型比较与阈值评估流程编排（多文件预测结果输入）。

## 输入与输出边界

- **输入**：验证后的配置对象（`MLConfig` 或兼容配置）。
- **输出**：结构化结果通过 `reporting` 生成文件产物，不直接在 workflow 中堆积导出逻辑。

## 与其他子模块关系

- 依赖 `runners` 执行算法循环。
- 依赖 `core` 进行计划与结果契约管理。
- 依赖 `reporting` 输出文件和图。
- 不应在 workflow 内直接实现复杂指标公式或模型算法。

## 维护建议

- workflow 代码应“薄”：更像 orchestration script，而不是算法实现文件。
- 新功能优先新增 runner 或 reporting 能力，再在 workflow 层接线。
- 若 workflow 出现大量指标或绘图细节，说明职责下沉不充分，应重构回下层模块。
