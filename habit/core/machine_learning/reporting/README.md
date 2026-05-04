# reporting 子模块说明

`reporting` 是机器学习模块的“产物输出层”，负责把结构化结果对象转换为可交付文件（模型文件、JSON、CSV、图形触发）。

## 为什么这个模块有必要

如果导出逻辑散落在各个 workflow，文件命名、字段映射、输出路径会逐渐不一致。
`reporting` 把“如何落盘”集中管理，让不同 workflow 共用一套稳定导出规范。

## 主要职责

1. **模型持久化**
   - `ModelStore` 负责保存训练后模型。
   - 支持 holdout 的单模型保存与 k-fold 的集成模型保存。

2. **报告文件写入**
   - `ReportWriter` 负责将 `RunResult` / `KFoldRunResult` / `InferenceResult` 写成标准 CSV、JSON。
   - 统一输出命名规范，避免不同 workflow 重复写导出代码。

3. **图形渲染触发**
   - `PlotComposer` 根据结果类型触发 `PlotManager` 对应绘图流程。
   - 让 workflow 不直接依赖绘图细节。

4. **比较流程结果导出**
   - `report_exporter.py` 为模型比较流程提供指标汇总存储与导出支持。

## 它是怎么起作用的

1. 接收 `core` 层结果对象（`RunResult`、`KFoldRunResult`、`InferenceResult`）。
2. 根据结果类型分发到对应写入逻辑。
3. 统一产出模型文件、JSON/CSV 报告、绘图任务。

## 具体例子

### 例子 1：Holdout 导出

- `ModelStore.save(run_result)` 保存每个模型的 `*_final_pipeline.pkl`。
- `ReportWriter.write(run_result)` 写 `ml_standard_summary.csv` 与 `ml_standard_results.json`。
- `PlotComposer.render(run_result)` 调 `PlotManager` 生成 train/test 图。

### 例子 2：K-Fold 导出

- `ModelStore.save_kfold_ensembles(kfold_result)` 把每折 estimator 封装为 `HabitEnsembleModel` 并保存。
- `ReportWriter.write(kfold_result)` 输出 `ml_kfold_summary.csv` 与 `ml_kfold_results.json`。

## 关键文件与作用

- `model_store.py`
  - 模型保存入口，统一管理保存目录、命名与模型结构。

- `report_writer.py`
  - 将结构化结果对象落盘为文本报告（summary/results/predictions）。

- `plot_composer.py`
  - 统一图形渲染调度，不关心具体图形实现细节。

- `report_exporter.py`
  - 面向 comparison 工作流的指标导出与聚合工具。

## 输入与输出边界

- **输入**：`core` 层定义的结果对象。
- **输出**：
  - 模型：`*.pkl`
  - 报告：`*_summary.csv`、`*_results.json`
  - 预测：`prediction_results.csv` 等
  - 图：由绘图子系统产生的 PDF/图片文件

## 与其他子模块关系

- 被 `workflows` 调用。
- 依赖 `core` 契约对象与 `visualization` 绘图能力。
- 不应实现模型训练、特征选择、指标公式。

## 维护建议

- 新增输出格式优先在 reporting 层扩展，不在 workflow 中分散导出逻辑。
- 任何新增结果字段，先在 `core` 中定义，再在 reporting 中映射导出。
- 输出文件命名与目录结构应保持向后兼容，避免破坏下游脚本。
