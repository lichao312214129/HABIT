# 重构 comparison_workflow.py TODO List

本列表旨在通过分阶段重构，提升 `comparison_workflow.py` 及其相关模块的软件设计质量。

---

### 阶段一：提升代码的健壮性和可测试性 (P0 - 高优先级)

-   [ ] **1. 引入配置模型 (Config Schema)**
    -   [ ] 使用 `pydantic` 创建一个 `ModelComparisonConfig` 类，用来定义 `config` 的完整结构。
    -   [ ] 在 `ModelComparison` 的 `__init__` 中，用 `ModelComparisonConfig(**config_dict)` 来解析和验证传入的字典配置。
    -   [ ] 将所有 `self.config.get(...)` 的调用替换为对 `self.config.metrics.youden_metrics.enabled` 这种强类型对象的访问。
    *   **目的**：消除配置中的“魔术字符串”，提供类型安全和自动补全，让配置结构清晰化。

-   [ ] **2. 应用依赖注入 (Dependency Injection)**
    -   [ ] 修改 `ModelComparison` 的 `__init__` 方法，使其接收 `evaluator`, `reporter`, `plot_manager`, `threshold_manager` 的实例作为参数，而不是在内部创建它们。
    -   [ ] 在调用 `ModelComparison` 的地方（例如 CLI 命令或主脚本），先创建这些依赖的实例，然后将它们注入。
    *   **目的**：解耦，使 `ModelComparison` 不再依赖于具体实现，极大提升单元测试的可行性。

### 阶段二：拆分“上帝类”，遵循单一职责 (P1 - 中优先级)

-   [ ] **3. 提取数据处理逻辑 -> `DataManager`**
    -   [ ] 创建一个新的 `DataManager` 类。
    -   [ ] 将 `_add_split_columns` 和 `_create_split_groups` 方法从 `ModelComparison` 移动到 `DataManager` 中。
    -   [ ] `DataManager` 内部持有 `MultifileEvaluator` 实例，并负责所有与数据加载、合并、分组相关的逻辑。
    *   **目的**：将数据准备的职责分离出去。

-   [ ] **4. 提取指标计算逻辑 -> `MetricOrchestrator`**
    -   [ ] 创建一个新的 `MetricOrchestrator` 类。
    -   [ ] 将 `_calculate_all_basic_metrics`, `_calculate_youden_metrics`, `_calculate_target_metrics` 等所有指标计算方法移动到此类中。
    -   [ ] `MetricOrchestrator` 接收 `DataManager` 提供的数据和分组信息，并负责运行所有评估。
    *   **目的**：将核心的指标计算与评估流程的编排分离开。

-   [ ] **5. 重构 `ModelComparison` 为顶层协调器**
    -   [ ] 移除所有被移走的方法，让 `ModelComparison` 变得非常轻量。
    -   [ ] 其 `run` 方法现在只负责按顺序调用 `data_manager.prepare_data()`, `metric_orchestrator.run_evaluations()`, `plot_manager.generate_plots()`, `reporter.save_reports()`。
    *   **目的**：使主工作流的逻辑一目了然。

### 阶段三：代码精炼与复用 (P2 - 低优先级)

-   [ ] **6. 抽象通用的阈值计算流程**
    -   [ ] 在 `MetricOrchestrator` 中，创建一个通用的 `_run_threshold_based_evaluation` 方法。
    -   [ ] 这个方法接收 `metric_name`, `find_threshold_func`, `apply_threshold_func` 等函数作为参数。
    -   [ ] 重构 `_calculate_youden_metrics` 和 `_calculate_target_metrics`，让它们都调用这个通用的方法，只是传入不同的参数。
    *   **目的**：消除 `youden` 和 `target` 计算流程中的重复代码。

-   [ ] **7. 精简长方法**
    -   [ ] 审视重构后依然过长的方法（例如 `_run_threshold_based_evaluation`），将其内部逻辑再次拆分为更小的私有方法（例如 `_find_thresholds_on_train` 和 `_apply_thresholds_to_splits`）。
    *   **目的**：提升代码的可读性。