# HABIT 机器学习模块改进建议 (ML Improvements)

基于对 `habit/core/machine_learning/machine_learning.py` 的分析，为了进一步提升框架的稳定性、专业性和科研可复现性，建议进行以下改进：

## 1. 核心流水线优化 (Pipeline Optimization)
*   **集成 `sklearn.pipeline`**: 建议将数据标准化 (Scaler)、特征选择 (FeatureSelector) 和模型 (Model) 封装进同一个 Pipeline 对象。
    *   **目的**: 确保预测 (Inference) 时使用的预处理参数（如均值/标准差）与训练时完全一致，消除数据泄露风险。
*   **解耦算法工厂**: 引入注册制 (Registry)，将不同的模型和特征选择算法插件化，避免 `if/else` 硬编码。

## 2. 配置与校验 (Configuration & Validation)
*   **引入 Pydantic**: 使用 Pydantic 定义配置模型，对 YAML 输入进行强类型校验。
    *   **目的**: 在程序启动阶段捕获拼写错误或非法参数值。
*   **默认配置补充**: 提供一个 `default_config.yaml`，确保用户未定义非核心参数时系统仍能运行。

## 3. 增强科研属性 (Academic/Scientific Features)
*   **实验元数据追踪**: 在模型保存目录中同步生成 `experiment_info.json`，记录：
    *   代码 Git Hash 值。
    *   训练集/测试集的样本 ID。
    *   完整的特征列表。
    *   所有库的版本号。
*   **自动超参搜索 (HPO)**: 集成 `Optuna` 或 `GridSearchCV` 模块，允许在配置中定义参数搜索空间。

## 4. 数据质量守卫 (Data Quality Guard)
*   **预处理自检**: 在进入训练前，自动执行：
    *   **常量特征过滤**: 剔除方差为 0 的特征。
    *   **标签平衡检查**: 若类别严重失衡，自动建议或启用 SMOTE/样本加权策略。
    *   **共线性检查**: 自动生成特征相关性矩阵，并对冗余特征进行预警。

## 5. 结果可视化 (Visualization Enhancement)
*   **标准化图表**: 自动生成特征重要性图 (Feature Importance)、混淆矩阵 (Confusion Matrix) 以及 PR 曲线。
*   **SHAP 解释性分析**: 集成 SHAP 库，自动生成特征贡献的解释图表。

## 6. TODO 待办事项
*   ** 机器学习没有保存图片**
*   **多分类支持**: 重构 `metrics.py` 以支持多分类任务（目前的 `calculate_metrics` 仅支持二分类）。
    *   引入 `macro/weighted` 平均指标。
    *   在 `roc_auc_score` 中增加 `multi_class='ovr'` 支持。
*   **Pipeline 导出与应用示例**: 编写示例代码展示如何加载导出的 `pipeline.pkl` 并对新数据进行预测。
*   **模型对比模块重构 (`model_comparison.py`)**: 
    *   **架构统一**: 继承 `BaseWorkflow` 并使用 `run_pipeline` 接口。
    *   **增加统计检验**: 引入 NRI (Net Reclassification Index) 和 IDI (Integrated Discrimination Improvement) 检验，用于评价新模型的改进贡献。
    *   **阈值管理增强**: 抽象 `ThresholdManager`，支持更灵活的阈值寻找与跨数据集应用逻辑。
