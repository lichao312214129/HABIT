# Habitat Analysis 模块设计问题与改进建议

## 当前状态：已重构完成 (2026-01-16)

本次重构主要解决了以下关键问题：
1. **模块拆分**：将庞大的 `HabitatAnalysis` 类拆分为三个独立的管理器：
   - `FeatureManager`: 负责特征提取和预处理
   - `ClusteringManager`: 负责聚类算法和验证
   - `ResultManager`: 负责结果保存和输出
2. **目录结构优化**：
   - `features/` -> `clustering_features/` (用于聚类前的特征)
   - `feature_extraction/` -> `habitat_feature_extraction/` (用于生境图的特征)
   - `pipeline.py` -> `strategies/clustering_pipeline.py`
3. **架构改进 (Facade + Strategy + Managers)**：
   - `HabitatAnalysis` 现在作为纯粹的协调器 (Facade) 和配置容器，不再包含处理逻辑。
   - `TwoStepStrategy` 和 `DirectPoolingStrategy` 直接调用 Managers 执行任务，逻辑内聚性更高。
   - 解决了并行计算时的序列化问题：将 `TwoStepStrategy` 的任务函数提取为模块级私有函数 `_process_subject_supervoxels`。
   - `HabitatAnalysis` 中冗余的代理方法（如 `process_all_subjects`）已被移除。
4. **Bug 修复**：
   - 修复了路径设置时的属性冲突。

现在系统架构如下：
- **L1 (Managers)**: 提供原子能力（提取、聚类、保存）。
- **L2 (Strategies)**: 编排业务流程，包含并行处理逻辑（模块级函数）。
- **L3 (Facade)**: 统一入口和配置管理。

---

## 原始问题记录

