# visualization 子模块说明

`visualization` 是机器学习模块的“绘图实现层”，负责生成模型评估相关图形产物，不负责指标计算和流程编排。

## 为什么这个模块有必要

绘图逻辑如果分散在 workflow/reporting，会导致图风格、命名和内容不一致。
`visualization` 把“怎么画图”集中管理，保证不同流程的图可比较、可复现。

## 主要职责

1. **统一图形绘制实现**
   - 输出 ROC、PR、校准曲线、决策曲线等图。
   - 统一图标题、文件命名、保存路径行为。

2. **封装绘图工具类**
   - `Plotter`：通用模型评估图绘制。
   - `KMSurvivalPlotter`：生存分析相关绘图（如 KM 曲线）。
   - `PlotManager`：作为 workflow/reporting 与底层绘图函数之间的桥梁。

3. **与结果对象解耦**
   - 接收已整理好的预测数据，不直接读取配置文件或训练对象。

## 它是怎么起作用的

1. `PlotComposer` 根据结果类型调度 `PlotManager`。
2. `PlotManager` 调用 `plotting.py` 中的具体绘图函数。
3. 将 ROC、PR、校准曲线、决策曲线等图写入输出目录。

## 具体例子

### 例子 1：Holdout 训练与测试图

- `PlotComposer.render(RunResult)` 被调用后，
  使用 `standard_train_` 与 `standard_test_` 前缀分别出图。
- 输出文件如 `standard_test_roc_curve.pdf`。

### 例子 2：K-Fold 聚合图

- `PlotComposer.render(KFoldRunResult)` 时，读取聚合后的预测 payload。
- 使用 `kfold_` 前缀生成聚合评估图。

## 关键文件与作用

- `plotting.py`
  - 核心图形绘制函数集合。

- `plot_manager.py`
  - 上层调度接口，统一处理不同流程下的绘图调用。

- `km_survival.py`
  - KM 生存分析图绘制实现。

- `__init__.py`
  - 暴露 `Plotter` 与 `KMSurvivalPlotter`。

## 输入与输出边界

- **输入**：结构化预测数据、模型名、图形配置参数。
- **输出**：图文件（通常为 PDF）写入目标目录。

## 与其他子模块关系

- 通常由 `reporting/plot_composer.py` 调用，不应直接由 CLI 层调用底层函数。
- 依赖 `evaluation` 提供的指标或预测结构，但不负责指标公式定义。

## 维护建议

- 绘图参数的默认值应集中管理，避免不同流程输出风格不一致。
- 图上文本保持英文（符合项目约束），日志与文档可用中文说明。
- 新增图类型时，优先在 `plotting.py` 增加单一能力，再由 `PlotManager` 暴露。
