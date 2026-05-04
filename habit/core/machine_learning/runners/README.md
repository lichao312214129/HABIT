# runners 子模块说明

`runners` 是机器学习模块的“执行层”，负责真正的训练、验证、预测循环。它不负责 CLI、配置解析、报告落盘等外层流程控制。

## 为什么这个模块有必要

如果把训练细节直接写在 `workflow` 里，`workflow` 会变成“又编排又计算”的巨型类，
测试和维护都困难。`runners` 把“计算”单独拆出来，让 `workflow` 只保留流程编排职责。

## 主要职责

1. **实现不同运行场景的执行循环**
   - `HoldoutRunner`：单次 train/test 训练评估。
   - `KFoldRunner`：多折训练、逐折评估与折间聚合。
   - `InferenceRunner`：加载已训练模型并执行预测（可选评估）。

2. **产出结构化结果**
   - 输出 `core/results.py` 中定义的契约对象（如 `RunResult`、`KFoldRunResult`、`InferenceResult`）。
   - 结果对象可直接交给 reporting 层处理。

3. **隔离流程依赖**
   - 通过 `RunnerContext` 注入依赖对象，而不是反向访问 workflow 内部状态。
   - 提升可测试性与复用性。

## 它是怎么起作用的

1. `workflow` 先构建 `RunnerContext` 和 `WorkflowPlan`。
2. 按运行模式选择对应 runner（`HoldoutRunner` / `KFoldRunner` / `InferenceRunner`）。
3. runner 只做计算：切分数据、训练模型、生成预测、计算指标。
4. runner 返回 `core` 里的结果对象，交给 `reporting` 持久化。

## 具体例子

### 例子 1：HoldoutRunner

- 输入：`RunnerContext + WorkflowPlan`。
- 过程：`split_data()` -> 每个模型训练 -> 计算 train/test 指标。
- 输出：`RunResult`（含 `models`、`summary_rows`、`dataset`）。

### 例子 2：KFoldRunner

- 输入：`X`、`y`、`RunnerContext + WorkflowPlan`。
- 过程：按折循环训练 -> 收集每折预测 -> 聚合全折指标。
- 输出：`KFoldRunResult`（含 `models`、`aggregated`、`summary_rows`）。

## 关键文件与作用

- `context.py`
  - `RunnerContext`：封装 `DataManager`、`PipelineBuilder`、`Resampler`、`logger`、`config`。
  - 形成稳定依赖注入边界。

- `base.py`
  - `BaseRunner`：通用加载与共享行为（如统一读取数据入口）。

- `holdout.py`
  - 单模型训练、预测容器构造、指标计算、汇总行生成。

- `kfold.py`
  - 分层/非分层 k-fold 切分、每折训练评估、跨折统计聚合。

- `inference.py`
  - 管道加载、输入数据读取、预测结果表构建、可选评估。

## 输入与输出边界

- **输入**：
  - `RunnerContext`
  - `WorkflowPlan`
  - 部分 runner 需要显式 `X`/`y`（如 k-fold）
- **输出**：
  - 与执行场景对应的结构化 Result 对象

## 与其他子模块关系

- 向上被 `workflows` 调用。
- 向下依赖 `core` 契约、`evaluation` 指标计算、`pipeline_utils` 管道构建。
- 不应包含文件导出细节（应交给 `reporting`）。

## 维护建议

- 新增运行模式时，优先新建 runner，而不是在现有 runner 中塞大量分支。
- runner 方法应尽量函数化、可单元测试，避免隐式读取全局状态。
- 所有 runner 的输出应优先通过 `core` 契约对齐，避免“特殊字典格式”蔓延。
