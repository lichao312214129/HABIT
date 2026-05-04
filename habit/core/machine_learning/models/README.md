# models 子模块说明

`models` 是机器学习模块的“模型构建层”，负责将配置映射为统一可训练的模型对象，并对外提供稳定的模型创建入口。

## 为什么这个模块有必要

如果 workflow 直接 new 各种第三方模型，会出现大量 `if/elif`，并且难以统一参数和接口。
`models` 通过工厂和包装层把这些差异收敛成统一接口。

## 主要职责

1. **统一模型注册与创建**
   - 通过 `ModelFactory` 将模型名称映射到具体模型类。
   - 让 workflow/runner 不关心具体模型实现细节。

2. **封装不同算法实现**
   - 维护逻辑回归、随机森林、SVM、XGBoost 等模型包装类。
   - 每个模型类遵循一致接口（`fit`、`predict`、`predict_proba`）。

3. **隔离第三方依赖差异**
   - 通过模块内包装兼容可选依赖（如 AutoGluon）。
   - 避免在上层流程中写大量依赖判断。

4. **提供集成模型能力**
   - `ensemble.py` 提供多折模型集合推理（如 soft/hard voting）。

## 它是怎么起作用的

1. `PipelineBuilder` 根据模型名调用 `ModelFactory` 获取 estimator。
2. estimator 作为 pipeline 最后一步参与 `fit/predict/predict_proba`。
3. K-Fold 场景下，`ModelStore` 可将多折 estimator 封装为 `HabitEnsembleModel` 持久化。

## 具体例子

### 例子 1：配置驱动创建逻辑回归模型

- 配置中写 `model_name=LogisticRegression` 和参数字典。
- `ModelFactory` 返回对应 estimator。
- runner 训练后，模型被保存为 `LogisticRegression_final_pipeline.pkl`。

### 例子 2：K-Fold 集成推理

- 每折训练得到一个 estimator。
- `HabitEnsembleModel` 在推理时可执行 `soft` 平均概率或 `hard` 多数投票。

## 关键文件与作用

- `factory.py`
  - 模型注册中心与构建入口。

- `base.py`
  - 模型包装基类与公共行为约束。

- `*_model.py`
  - 各类具体模型实现（如 `logistic_regression_model.py`、`random_forest_model.py`）。

- `ensemble.py`
  - k-fold 场景模型集成推理支持。

## 输入与输出边界

- **输入**：模型名称、参数字典、特征矩阵。
- **输出**：sklearn 风格的可训练/可预测模型对象。

## 与其他子模块关系

- 被 `pipeline_utils.PipelineBuilder` 调用。
- 被 `runners` 间接使用（runner 通过 pipeline 触发模型训练）。
- 不应承担数据切分、指标计算、文件落盘职责。

## 维护建议

- 新增模型时，优先在 `models` 内增加包装类并注册到 `ModelFactory`。
- 保持各模型实现接口一致，避免上层出现特判分支。
- 对可选依赖模型做好降级逻辑，确保核心流程在依赖缺失时仍可运行。
