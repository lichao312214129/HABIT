# HABIT 架构优化与重构 TODO List

本文档包含从架构层面提出的优化方向，旨在提升整个 HABIT 项目的软件设计质量、可维护性和可扩展性。

---

## 🏗️ 架构层面优化方向 (Architecture-Level Improvements)

### 一、依赖注入与解耦 (Dependency Injection & Decoupling) - P0

#### 1.1 统一依赖注入模式
**问题**：当前代码中大量类在构造函数中直接创建依赖，导致紧耦合和难以测试。

**现状分析**：
- `HabitatAnalysis` 在 `__init__` 中直接创建 `FeatureManager`, `ClusteringManager`, `ResultManager`
- `BaseWorkflow` 直接创建 `DataManager`, `PlotManager`, `PipelineBuilder`
- `BatchProcessor` 在内部创建预处理器实例

**优化方向**：
- [ ] **引入依赖注入容器**：考虑使用轻量级 DI 框架（如 `dependency-injector`）或实现简单的服务定位器模式
- [ ] **重构核心类构造函数**：将依赖作为参数传入，而不是在内部创建
  ```python
  # 当前方式
  class HabitatAnalysis:
      def __init__(self, config):
          self.feature_manager = FeatureManager(config, logger)
  
  # 优化后
  class HabitatAnalysis:
      def __init__(self, config, feature_manager, clustering_manager, result_manager):
          self.feature_manager = feature_manager
  ```
- [ ] **创建工厂类**：为复杂对象的创建提供统一的工厂接口
- [ ] **定义接口抽象**：为 Manager 类定义抽象基类，支持接口隔离原则

**影响范围**：
- `habit/core/habitat_analysis/habitat_analysis.py`
- `habit/core/machine_learning/base_workflow.py`
- `habit/core/preprocessing/image_processor_pipeline.py`
- 所有 Manager 类

---

#### 1.2 配置管理统一化
**问题**：配置验证和访问方式不一致，部分使用 Pydantic，部分使用字典。

**现状分析**：
- `HabitatAnalysisConfig` 使用 Pydantic
- `MLConfig` 使用 Pydantic
- `PreprocessingConfig` 使用 Pydantic
- 但 `BaseWorkflow` 中有 fallback 到字典的逻辑
- 配置访问混用 `config.get()` 和 `config.field_name`

**优化方向**：
- [ ] **统一配置基类**：创建 `BaseConfig` 抽象基类，所有配置类继承它
- [ ] **配置验证中间件**：在配置加载阶段统一验证，避免运行时错误
- [ ] **配置访问器模式**：提供统一的配置访问接口，消除 `config.get()` 调用
- [ ] **配置版本管理**：支持配置文件的版本迁移和兼容性检查

**影响范围**：
- `habit/core/*/config_schemas.py`
- `habit/utils/config_utils.py`
- 所有使用配置的类

---

### 二、错误处理与异常管理 (Error Handling & Exception Management) - P0

#### 2.1 统一异常体系
**问题**：异常处理不统一，有些地方捕获所有异常，有些地方直接抛出。

**现状分析**：
- CLI 命令中有 `except Exception as e` 的通用捕获
- 并行处理中有 `ProcessingResult` 包装，但使用不一致
- 缺少自定义异常类型层次结构

**优化方向**：
- [ ] **定义异常层次结构**：
  ```python
  class HABITError(Exception): pass
  class ConfigurationError(HABITError): pass
  class DataError(HABITError): pass
  class ProcessingError(HABITError): pass
  class ValidationError(HABITError): pass
  ```
- [ ] **统一错误响应格式**：使用 `Result` 类型（类似 Rust 的 Result）包装操作结果
- [ ] **错误恢复策略**：为关键操作定义错误恢复和重试机制
- [ ] **错误上下文传播**：使用异常链（`raise ... from ...`）保留完整错误上下文

**影响范围**：
- 所有模块的错误处理代码
- `habit/utils/parallel_utils.py` 中的 `ProcessingResult`
- CLI 命令的错误处理

---

#### 2.2 日志管理标准化
**问题**：虽然有 `LoggerManager`，但使用方式不一致，部分代码直接使用 `logging`。

**优化方向**：
- [ ] **强制使用 LoggerManager**：禁止直接使用 `logging.getLogger()`
- [ ] **日志上下文管理**：使用上下文管理器自动添加日志上下文信息
- [ ] **结构化日志**：考虑使用结构化日志格式（JSON）便于后续分析
- [ ] **日志级别策略**：定义统一的日志级别使用规范

---

### 三、单一职责与关注点分离 (Single Responsibility & Separation of Concerns) - P1

#### 3.1 Manager 类职责细化
**问题**：部分 Manager 类承担过多职责，违反单一职责原则。

**现状分析**：
- `FeatureManager` 同时负责特征提取、预处理、路径管理
- `ClusteringManager` 包含聚类算法选择、验证、可视化
- `DataManager` 混合了数据加载、预处理、验证逻辑

**优化方向**：
- [ ] **拆分 FeatureManager**：
  - `FeatureExtractorOrchestrator`：协调特征提取流程
  - `FeaturePreprocessor`：特征预处理逻辑
  - `FeaturePathResolver`：路径解析和文件管理
- [ ] **拆分 ClusteringManager**：
  - `ClusteringExecutor`：执行聚类算法
  - `ClusterValidator`：聚类验证
  - `ClusterVisualizer`：聚类可视化
- [ ] **拆分 DataManager**：
  - `DataLoader`：数据加载
  - `DataValidator`：数据验证
  - `DataTransformer`：数据转换

**影响范围**：
- `habit/core/habitat_analysis/managers/`
- `habit/core/machine_learning/data_manager.py`

---

#### 3.2 工作流模式统一
**问题**：不同模块的工作流实现方式不一致。

**现状分析**：
- `BaseWorkflow` 提供了基础框架
- `HabitatAnalysis` 使用策略模式
- `BatchProcessor` 使用命令模式
- 缺少统一的工作流抽象

**优化方向**：
- [ ] **定义统一工作流接口**：`IWorkflow` 接口，所有工作流实现它
- [ ] **工作流步骤抽象**：将工作流步骤抽象为 `WorkflowStep`，支持步骤组合
- [ ] **工作流编排器**：创建 `WorkflowOrchestrator` 统一管理工作流执行
- [ ] **支持工作流暂停/恢复**：为长时间运行的工作流添加状态持久化

**影响范围**：
- `habit/core/machine_learning/workflows/`
- `habit/core/habitat_analysis/`
- `habit/core/preprocessing/`

---

### 四、接口抽象与多态 (Interface Abstraction & Polymorphism) - P1

#### 4.1 算法接口标准化
**问题**：虽然使用了工厂模式，但接口定义不够统一。

**优化方向**：
- [ ] **统一算法接口**：所有算法（聚类、特征提取、预处理）实现统一的 `IAlgorithm` 接口
- [ ] **算法注册表**：使用装饰器模式统一注册所有算法
- [ ] **算法元数据**：为每个算法定义元数据（名称、版本、参数、依赖）
- [ ] **算法插件系统**：支持动态加载外部算法插件

**影响范围**：
- `habit/core/habitat_analysis/algorithms/`
- `habit/core/habitat_analysis/extractors/`
- `habit/core/preprocessing/`
- 所有 Factory 类

---

#### 4.2 数据访问层抽象
**问题**：数据访问逻辑分散在各个模块中，缺少统一的数据访问抽象。

**优化方向**：
- [ ] **数据访问接口**：定义 `IDataRepository` 接口
- [ ] **数据源抽象**：支持多种数据源（文件系统、数据库、云存储）
- [ ] **数据缓存层**：为频繁访问的数据添加缓存机制
- [ ] **数据版本管理**：支持数据版本追踪和回滚

---

### 五、代码复用与DRY原则 (Code Reuse & DRY Principle) - P2

#### 5.1 通用组件提取
**问题**：多个模块中有重复的逻辑实现。

**现状分析**：
- 日志配置逻辑在多个类中重复
- 路径解析逻辑分散
- 进度条使用方式不统一（虽然有 `progress_utils`）

**优化方向**：
- [ ] **提取通用服务**：
  - `ConfigurationService`：统一配置管理
  - `LoggingService`：统一日志管理
  - `PathService`：统一路径解析
  - `ProgressService`：统一进度显示
- [ ] **工具类库**：将通用工具函数组织成工具类库
- [ ] **共享组件模块**：创建 `habit/core/common/` 存放共享组件

---

#### 5.2 模板方法模式应用
**问题**：相似的处理流程在多个地方重复实现。

**优化方向**：
- [ ] **提取模板方法**：为相似的处理流程定义模板方法
- [ ] **钩子方法**：使用钩子方法允许子类定制特定步骤
- [ ] **流程构建器**：使用构建器模式简化复杂流程的构建

---

### 六、测试性与可观测性 (Testability & Observability) - P1

#### 6.1 测试基础设施
**问题**：由于紧耦合，单元测试难以编写。

**优化方向**：
- [ ] **Mock 框架集成**：统一使用 `unittest.mock` 或 `pytest-mock`
- [ ] **测试夹具**：为常见测试场景创建可复用的测试夹具
- [ ] **测试数据管理**：创建测试数据生成和管理工具
- [ ] **集成测试框架**：建立端到端集成测试框架

---

#### 6.2 可观测性增强
**问题**：缺少运行时监控和性能分析工具。

**优化方向**：
- [ ] **性能指标收集**：添加关键操作的性能指标收集
- [ ] **健康检查端点**：为长时间运行的任务添加健康检查
- [ ] **操作追踪**：使用追踪ID追踪跨模块的操作流程
- [ ] **资源使用监控**：监控内存、CPU、磁盘使用情况

---

### 七、并发与并行处理 (Concurrency & Parallel Processing) - P1

#### 7.1 并行处理抽象
**问题**：并行处理逻辑分散，缺少统一抽象。

**现状分析**：
- `parallel_utils.py` 提供了基础工具
- 但不同模块的并行处理方式不一致
- 缺少任务调度和资源管理

**优化方向**：
- [ ] **任务执行器抽象**：定义 `ITaskExecutor` 接口
- [ ] **任务调度器**：实现任务调度和优先级管理
- [ ] **资源池管理**：统一管理计算资源（进程池、线程池）
- [ ] **异步支持**：考虑添加 `asyncio` 支持用于 I/O 密集型操作

**影响范围**：
- `habit/utils/parallel_utils.py`
- 所有使用并行处理的模块

---

### 八、数据模型与领域驱动设计 (Data Models & DDD) - P2

#### 8.1 领域模型定义
**问题**：缺少明确的领域模型定义，数据传递使用字典和 DataFrame。

**优化方向**：
- [ ] **领域实体定义**：定义核心领域实体（Subject, Image, Habitat, Feature 等）
- [ ] **值对象**：使用值对象封装不可变数据
- [ ] **领域服务**：将业务逻辑封装到领域服务中
- [ ] **仓储模式**：使用仓储模式管理数据持久化

---

### 九、配置与部署 (Configuration & Deployment) - P2

#### 9.1 环境管理
**优化方向**：
- [ ] **环境配置分离**：区分开发、测试、生产环境配置
- [ ] **配置模板系统**：提供配置模板和验证工具
- [ ] **配置热重载**：支持运行时配置更新（对于长时间运行的任务）

---

### 十、文档与API设计 (Documentation & API Design) - P2

#### 10.1 API 文档
**优化方向**：
- [ ] **API 文档生成**：使用 Sphinx 或 MkDocs 生成 API 文档
- [ ] **类型注解完善**：为所有公共 API 添加完整的类型注解
- [ ] **使用示例**：为每个主要功能提供使用示例

---

## 📋 具体模块重构 TODO

### 重构 comparison_workflow.py TODO List

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

---

## 🎯 实施建议 (Implementation Recommendations)

### 优先级说明
- **P0 (高优先级)**：影响代码质量和可维护性的核心问题，应优先解决
- **P1 (中优先级)**：重要的架构改进，可在 P0 完成后进行
- **P2 (低优先级)**：优化和增强，可在主要功能稳定后考虑

### 重构策略

#### 渐进式重构
1. **保持向后兼容**：重构时保持 API 兼容性，使用适配器模式过渡
2. **小步快跑**：每次重构一个模块或功能，确保测试通过后再继续
3. **测试驱动**：先编写测试，再重构代码，确保功能不变
4. **文档同步**：重构时同步更新文档和示例

#### 重构顺序建议
1. **第一阶段**（1-2个月）：
   - 统一配置管理（1.2）
   - 统一异常体系（2.1）
   - 日志管理标准化（2.2）
   - 依赖注入核心类（1.1 部分）

2. **第二阶段**（2-3个月）：
   - Manager 类职责细化（3.1）
   - 工作流模式统一（3.2）
   - 算法接口标准化（4.1）
   - 并行处理抽象（7.1）

3. **第三阶段**（3-4个月）：
   - 通用组件提取（5.1）
   - 测试基础设施（6.1）
   - 数据访问层抽象（4.2）
   - 领域模型定义（8.1）

### 风险控制

#### 重构风险
- **功能回归**：通过完善的测试覆盖降低风险
- **性能下降**：重构后进行性能基准测试
- **兼容性破坏**：使用版本控制和适配器模式

#### 缓解措施
- 建立 CI/CD 流程，自动化测试
- 使用特性开关（Feature Flags）控制新功能发布
- 保持重构分支与主分支同步
- 定期代码审查

---

## 📊 架构改进效果评估

### 预期收益

#### 代码质量
- ✅ **可测试性提升**：依赖注入后，单元测试覆盖率目标 80%+
- ✅ **可维护性提升**：单一职责后，代码复杂度降低 30%+
- ✅ **可扩展性提升**：接口抽象后，新功能开发时间减少 40%+

#### 开发效率
- ✅ **开发速度**：统一接口和模式后，新功能开发更快
- ✅ **Bug 修复**：清晰的错误处理，问题定位时间减少 50%+
- ✅ **代码审查**：清晰的架构，代码审查效率提升

#### 系统稳定性
- ✅ **错误处理**：统一的异常体系，系统更健壮
- ✅ **资源管理**：统一的资源管理，避免资源泄漏
- ✅ **性能监控**：可观测性增强，问题发现更及时

---

## 📝 注意事项

1. **渐进式改进**：不要一次性重构所有代码，采用渐进式方法
2. **保持功能**：重构过程中确保现有功能不受影响
3. **测试覆盖**：重构前确保有足够的测试覆盖
4. **文档更新**：重构后及时更新相关文档
5. **团队沟通**：重大架构变更需要团队讨论和评审

---

## 🔗 相关资源

- [SOLID 原则](https://en.wikipedia.org/wiki/SOLID)
- [设计模式](https://refactoring.guru/design-patterns)
- [依赖注入](https://martinfowler.com/articles/injection.html)
- [领域驱动设计](https://martinfowler.com/bliki/DomainDrivenDesign.html)