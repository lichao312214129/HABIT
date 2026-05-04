# feature_selectors 子模块说明

`feature_selectors` 是机器学习模块的“特征筛选层”，负责提供可插拔的特征选择算法，并统一接入训练流水线。

## 为什么这个模块有必要

特征筛选是高频变化点：不同项目会替换不同方法。把它独立成子模块，
可以让上层只依赖“选择器名称 + 参数”，而不依赖具体算法实现。

## 主要职责

1. **统一选择器注册机制**
   - 通过 `selector_registry.py` 管理选择器注册、检索、执行。
   - 上层只需要方法名和参数，不需要直接导入具体算法实现。

2. **提供多种特征选择方法**
   - 相关性、VIF、LASSO、ANOVA、Chi2、stepwise、mRMR 等。
   - 支持统计检验类方法与工程规则类方法混合使用。

3. **与 Pipeline 无缝集成**
   - `FeatureSelectTransformer`（位于 `pipeline_utils.py`）调用 `run_selector` 形成 sklearn Pipeline 步骤。
   - 支持在归一化前后分别执行筛选阶段。

4. **标准化输出与日志行为**
   - 各选择器统一输出“保留特征列表”，并可输出诊断信息和图。

## 它是怎么起作用的

1. 配置里声明选择器名称和参数（如 `lasso`、`mrmr`）。
2. `selector_registry.run_selector()` 按名称取到具体实现并执行。
3. 返回统一的保留特征列表。
4. `FeatureSelectTransformer` 将该列表应用到 pipeline 的后续步骤。

## 具体例子

### 例子 1：LASSO 选择器在训练中生效

- `PipelineBuilder` 构建 pipeline 时插入 `FeatureSelectTransformer`。
- transformer 调 `run_selector(..., method="lasso")`。
- 返回的特征子集用于后续 scaler 和 model 训练。

### 例子 2：新增一个选择器如何接入

- 在 `feature_selectors` 下实现函数并注册到 `selector_registry`。
- 配置中填新方法名即可被 pipeline 调用，不需要改 workflow/runner。

## 关键文件与作用

- `selector_registry.py`
  - 注册器核心：`register_selector`、`get_selector`、`run_selector`。

- `*_selector.py`
  - 各算法实现文件，例如：
  - `correlation_selector.py`
  - `lasso_selector.py`
  - `anova_selector.py`
  - `chi2_selector.py`
  - `variance_selector.py`
  - `vif_selector.py`

- `transformer.py`
  - 提供与 pipeline 交互的特征筛选转换逻辑（与上层流水线衔接）。

- `icc/`
  - ICC 相关的子模块实现与文档。

## 输入与输出边界

- **输入**：
  - 特征数据（通常为 `pd.DataFrame`）
  - 标签数据（`pd.Series` 或兼容结构）
  - 当前候选特征列表
  - 选择器参数
- **输出**：
  - 筛选后的特征名称列表（`List[str]`）

## 与其他子模块关系

- 被 `pipeline_utils.PipelineBuilder` 通过 transformer 调用。
- 与 `models` 解耦：不依赖具体模型实现（除部分算法自身必要模型依赖）。
- 不应承担训练评估、报告导出职责。

## 维护建议

- 新增选择器时，优先使用注册机制，不直接在上层写 `if/elif` 分支。
- 统一参数命名和返回格式，避免 pipeline 接入时做兼容分支。
- 若选择器依赖外部库，需提供清晰降级策略，防止主流程中断。
