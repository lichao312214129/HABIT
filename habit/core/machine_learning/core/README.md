# core 子模块说明

`core` 是机器学习模块的“数据契约层”，专门定义各层之间交换的数据对象。

## 为什么这个模块有必要

如果没有 `core`，`workflow`、`runner`、`reporting` 往往会通过临时 `dict` 互传数据，
字段名容易漂移，改一个 key 会牵连多处代码。`core` 的作用就是把这些“口头约定”变成
显式的数据合同。

## 它是怎么起作用的

1. `WorkflowPlan` 固化一次运行的配置快照（配置、输出目录、随机种子）。
2. `Runner` 计算后返回结构化结果（`RunResult`、`KFoldRunResult`、`InferenceResult`）。
3. `ReportWriter`、`ModelStore`、`PlotComposer` 只依赖这些结果对象，不直接碰临时字典。
4. 需要兼容旧逻辑时，通过 `to_legacy_results()` 回落到历史字典结构。

## 具体例子

### 例子 1：Holdout 训练结果如何被消费

- `HoldoutRunner.run()` 产出 `RunResult`。
- `RunResult.models` 提供每个模型的 `train/test` 预测和指标。
- `ReportWriter` 用 `RunResult.summary_rows` 写 `*_summary.csv`，
  用 `RunResult.to_legacy_results()` 写 `*_results.json`。

### 例子 2：K-Fold 结果如何兼容历史代码

- 新结构里，`KFoldRunResult` 拆成 `models`（逐折）和 `aggregated`（跨折汇总）。
- 历史代码仍可通过 `KFoldRunResult.results` 或 `to_legacy_results()` 拿到旧 shape，
  所以旧绘图与旧脚本不需要立刻改。

## 对外提供的核心对象

- `DatasetSnapshot`
- `WorkflowPlan`
- `WorkflowResult`
- `ModelResult`
- `RunResult`
- `KFoldModelResult`
- `AggregatedModelResult`
- `KFoldRunResult`
- `InferenceResult`

## 输入与输出边界

- **输入**：来自 `workflow/runner` 的标准化配置与数据。
- **输出**：供 `reporting` 与可视化消费的结构化结果对象。

## 维护建议

- 新增结果字段时，优先修改 `core/results.py` 的 dataclass，并同步 legacy 适配方法。
- 不要在 `workflow/runner` 中随意新增临时 `dict` 结构。
- 新运行模式先定义新 Result 契约，再接入 `reporting`。