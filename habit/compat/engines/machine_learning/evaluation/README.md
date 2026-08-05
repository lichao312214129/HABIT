# evaluation 子模块说明

`evaluation` 是机器学习模块的“评估计算层”，负责指标计算、阈值策略、预测数据标准化以及多文件评估支撑能力。

## 为什么这个模块有必要

评估逻辑如果散落在 runner/workflow，指标公式和阈值策略会出现不一致。
`evaluation` 把“怎么算好坏”集中到一个地方，保证训练、交叉验证、模型比较口径一致。

## 主要职责

1. **统一指标计算**
   - 计算分类任务常用指标（AUC、敏感度、特异度、准确率等）。
   - 提供一致的输入约束，避免各处重复实现指标逻辑。

2. **阈值策略管理**
   - 支持 Youden、目标敏感度/特异度等阈值选择策略。
   - 提供“训练集求阈值，测试集复用阈值”的标准流程支持。

3. **预测结果容器化**
   - 将 `y_true`、`y_prob`、`y_pred` 封装为统一容器，处理维度与缺失值细节。
   - 给 runner/workflow 提供稳定输入接口。

4. **多文件评估支持**
   - 在 comparison 流程中提供多来源预测文件读取、统一评估与统计检验入口。

## 它是怎么起作用的

1. runner 先用 `PredictionContainer` 规范化 `y_true/y_prob/y_pred`。
2. `metrics.py` 计算 AUC、敏感度、特异度、F1 等指标。
3. 需要阈值控制时，`threshold_manager.py` 负责阈值搜索和复用。
4. 比较流程中，`model_evaluation.py` 将多文件预测统一成可比较结果。

## 具体例子

### 例子 1：Holdout 指标计算

- `HoldoutRunner` 生成 train/test 的 `PredictionContainer`。
- 调 `calculate_metrics(container)` 得到指标字典。
- 指标被写入 `ModelResult.train_metrics` 和 `ModelResult.test_metrics`。

### 例子 2：预测模式可选评估

- `InferenceRunner` 在 `evaluate=True` 且能解析标签列时，
  调 `calculate_metrics(container)` 生成评估指标。
- 结果放入 `InferenceResult.metrics`，后续由 `ReportWriter` 输出 `evaluation_metrics.csv`。

## 关键文件与作用

- `metrics.py`
  - 指标计算主入口与阈值应用函数。

- `prediction_container.py`
  - 标准化预测数组输入，统一二分类/多分类概率处理。

- `threshold_manager.py`
  - 阈值查找、缓存和读取逻辑。

- `model_evaluation.py`
  - 多文件评估与比较流程的底层评估支持。

- `calculations.py`
  - 评估过程中的辅助计算逻辑。

## 输入与输出边界

- **输入**：模型预测数组、标签数组、阈值配置。
- **输出**：结构化指标字典、阈值信息、比较结果中间数据。

## 与其他子模块关系

- 被 `runners` 和 `workflows/comparison_workflow.py` 调用。
- 依赖 `statistics` 完成部分统计检验（如 DeLong）。
- 不应负责文件落盘和目录管理（由 `reporting` 负责）。

## 维护建议

- 指标公式与阈值策略要集中维护，避免在 workflow 中重复写逻辑。
- 新增指标时，优先在 `metrics.py` 增加单一实现，再由上层复用。
- 对外暴露的数据结构保持稳定，减少下游改动范围。
