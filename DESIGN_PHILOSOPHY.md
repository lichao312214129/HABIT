# HABIT 包设计哲学与核心思想

本文档阐述 HABIT (Habitat Analysis: Biomedical Imaging Toolkit) 包的核心设计哲学、架构原则和实现思想，为开发者提供设计指导和最佳实践参考。

---

## 📐 核心设计原则

### 1. 配置驱动 (Configuration-Driven)

**哲学**：所有行为由配置决定，而非硬编码。

**实现方式**：
- **统一配置基类**：所有配置类继承 `BaseConfig`（Pydantic）
- **类型安全**：使用 Pydantic 进行配置验证和类型检查
- **统一访问**：通过 `ConfigAccessor` 提供一致的配置访问接口
- **向后兼容**：支持字典和 Pydantic 对象两种访问方式

**示例**：
```python
# 配置类统一继承 BaseConfig
class MLConfig(BaseConfig):
    input: List[InputFileConfig]
    output: str
    models: Dict[str, ModelConfig]

# 统一加载方式
config = MLConfig.from_file('config.yaml')

# 统一访问方式
output_dir = config.output  # 属性访问
models = config.models  # 类型安全
```

**设计优势**：
- ✅ 类型安全：IDE 自动补全和类型检查
- ✅ 验证保证：配置加载时自动验证
- ✅ 文档化：配置结构即文档
- ✅ 可维护性：配置变更集中管理

---

### 2. 依赖注入 (Dependency Injection)

**哲学**：依赖通过构造函数注入，而非在类内部创建。

**实现方式**：
- **ServiceConfigurator**：统一的服务创建和配置中心
- **工厂模式**：通过 `create_*` 方法创建服务实例
- **依赖解耦**：服务不直接创建依赖，由配置器负责

**示例**：
```python
# 使用 ServiceConfigurator 创建服务
configurator = ServiceConfigurator(
    config=config,
    logger=logger,
    output_dir=output_dir
)

# 统一的服务创建接口
habitat_analysis = configurator.create_habitat_analysis()
ml_workflow = configurator.create_ml_workflow()
```

**设计优势**：
- ✅ 可测试性：易于注入 Mock 对象
- ✅ 可扩展性：新增服务只需扩展配置器
- ✅ 一致性：所有服务创建方式统一
- ✅ 解耦：服务与依赖创建逻辑分离

---

### 3. 工作流模式 (Workflow Pattern)

**哲学**：复杂流程抽象为可复用的工作流。

**实现方式**：
- **BaseWorkflow**：所有工作流的抽象基类
- **模板方法**：定义标准流程，子类实现具体步骤
- **回调机制**：通过 CallbackList 支持扩展点

**示例**：
```python
class BaseWorkflow(ABC):
    def __init__(self, config, module_name):
        # 统一的基础设施初始化
        self.data_manager = DataManager(config, logger)
        self.pipeline_builder = PipelineBuilder(config, output_dir)
        self.callbacks = CallbackList([...])
    
    @abstractmethod
    def run_pipeline(self):
        """子类实现具体流程"""
        pass

class MachineLearningWorkflow(BaseWorkflow):
    def run_pipeline(self):
        # 1. Load Data
        X, y = self._load_and_prepare_data()
        # 2. Split Data
        X_train, X_test, y_train, y_test = self.data_manager.split_data()
        # 3. Process Models
        # ...
```

**设计优势**：
- ✅ 代码复用：基础设施统一管理
- ✅ 一致性：所有工作流遵循相同模式
- ✅ 可扩展性：通过继承轻松创建新工作流
- ✅ 可维护性：修改基础设施影响所有工作流

---

### 4. 策略模式 (Strategy Pattern)

**哲学**：算法和策略可互换，通过配置选择。

**实现方式**：
- **策略接口**：定义抽象策略基类
- **策略注册**：通过注册表管理可用策略
- **运行时选择**：根据配置动态选择策略

**示例**：
```python
# 聚类策略
class BaseStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        pass

class OneStepStrategy(BaseStrategy): ...
class TwoStepStrategy(BaseStrategy): ...
class DirectPoolingStrategy(BaseStrategy): ...

# 通过配置选择策略
strategy = create_strategy(config.clustering.strategy)
```

**设计优势**：
- ✅ 灵活性：运行时选择算法
- ✅ 可扩展性：新增策略无需修改现有代码
- ✅ 可测试性：每个策略独立测试
- ✅ 清晰性：策略职责明确

---

### 5. 容器模式 (Container Pattern)

**哲学**：复杂数据结构封装为容器，提供统一接口。

**实现方式**：
- **PredictionContainer**：统一处理预测结果
- **自动处理**：自动识别二分类/多分类
- **统一接口**：提供 `get_eval_probs()`, `get_binary_probs()` 等方法

**示例**：
```python
# 统一使用 PredictionContainer
container = PredictionContainer(
    y_true=y_true,
    y_prob=probs_raw,  # 原始概率
    y_pred=preds
)

# 自动处理二分类/多分类
probs = container.get_eval_probs()  # 1D for binary, 2D for multiclass
metrics = calculate_metrics(container)
```

**设计优势**：
- ✅ 一致性：所有预测结果处理方式统一
- ✅ 自动化：自动处理边界情况
- ✅ 可维护性：逻辑集中在一个类
- ✅ 可复用性：所有模块共享同一容器

---

### 6. 模块化设计 (Modular Design)

**哲学**：功能按职责划分到独立模块，模块间低耦合。

**目录结构**：
```
habit/
├── cli_commands/      # CLI 命令层
│   └── commands/      # 具体命令实现
├── core/              # 核心业务逻辑
│   ├── common/        # 共享组件（配置、服务配置器）
│   ├── habitat_analysis/  # 生境分析模块
│   ├── machine_learning/  # 机器学习模块
│   └── preprocessing/     # 预处理模块
└── utils/             # 工具函数
    ├── log_utils.py   # 日志工具
    ├── progress_utils.py  # 进度条工具
    └── io_utils.py    # IO 工具
```

**设计原则**：
- **单一职责**：每个模块只负责一个领域
- **高内聚**：相关功能组织在同一模块
- **低耦合**：模块间通过接口交互
- **可替换**：模块可独立替换和测试

---

### 7. 统一工具管理 (Unified Utilities)

**哲学**：通用功能统一管理，避免重复实现。

**实现方式**：
- **统一日志**：`habit/utils/log_utils.py` 提供 `LoggerManager`
- **统一进度条**：`habit/utils/progress_utils.py` 提供标准进度条
- **统一 IO**：`habit/utils/io_utils.py` 提供文件操作

**示例**：
```python
# 统一日志使用
from habit.utils.log_utils import setup_logger
logger = setup_logger(name='module', output_dir='./logs')

# 统一进度条使用（开发时）
from habit.utils.progress_utils import ProgressBar
with ProgressBar(total=100) as pbar:
    pbar.update(10)
```

**设计优势**：
- ✅ 一致性：所有模块使用相同的工具接口
- ✅ 可维护性：工具更新影响所有模块
- ✅ 可配置性：工具行为统一配置
- ✅ 可测试性：工具可独立测试

---

### 8. 向后兼容 (Backward Compatibility)

**哲学**：在架构演进过程中保持向后兼容，平滑迁移。

**实现方式**：
- **渐进式迁移**：新功能使用新架构，旧功能保持兼容
- **Fallback 机制**：新代码支持旧格式（如字典配置）
- **双重支持**：同时支持新旧两种方式

**示例**：
```python
# 支持 Pydantic 对象和字典两种格式
if hasattr(config, 'input'):
    # Pydantic object
    self.input_config = config.input
else:
    # Dict (backward compatibility)
    self.input_config = config['input']
```

**设计优势**：
- ✅ 平滑迁移：无需一次性重构所有代码
- ✅ 风险控制：逐步验证新架构
- ✅ 用户友好：现有配置和代码继续工作
- ✅ 灵活性：允许混合使用新旧方式

---

## 🏗️ 架构模式

### 1. 分层架构 (Layered Architecture)

```
┌─────────────────────────────────────┐
│      CLI Layer (cli_commands/)      │  ← 用户接口层
├─────────────────────────────────────┤
│      Core Layer (core/)             │  ← 业务逻辑层
│  ├── habitat_analysis/              │
│  ├── machine_learning/             │
│  └── preprocessing/                 │
├─────────────────────────────────────┤
│      Common Layer (core/common/)    │  ← 共享组件层
│  ├── config_base.py                 │
│  ├── config_validator.py            │
│  └── service_configurator.py        │
├─────────────────────────────────────┤
│      Utils Layer (utils/)           │  ← 工具层
│  ├── log_utils.py                   │
│  ├── progress_utils.py              │
│  └── io_utils.py                    │
└─────────────────────────────────────┘
```

**原则**：
- **上层依赖下层**：CLI 依赖 Core，Core 依赖 Utils
- **避免循环依赖**：Common 层不依赖业务层
- **接口抽象**：层间通过接口交互

---

### 2. 协调者模式 (Coordinator Pattern)

**哲学**：复杂流程由协调者类统一编排，而非分散在各处。

**实现**：
- **HabitatAnalysis**：协调特征提取、聚类、结果管理
- **BaseWorkflow**：协调数据管理、模型训练、评估
- **ServiceConfigurator**：协调服务创建和依赖注入

**示例**：
```python
class HabitatAnalysis:
    """协调者：编排整个生境分析流程"""
    def __init__(self, config, feature_manager, clustering_manager, result_manager):
        self.feature_manager = feature_manager
        self.clustering_manager = clustering_manager
        self.result_manager = result_manager
    
    def run(self):
        # 协调各个管理器完成流程
        features = self.feature_manager.extract()
        habitats = self.clustering_manager.cluster(features)
        self.result_manager.save(habitats)
```

---

### 3. 管理器模式 (Manager Pattern)

**哲学**：相关功能组织在管理器类中，提供统一接口。

**实现**：
- **FeatureManager**：管理特征提取相关功能
- **ClusteringManager**：管理聚类相关功能
- **ResultManager**：管理结果保存和报告
- **DataManager**：管理数据加载和分割
- **PlotManager**：管理可视化相关功能

**设计原则**：
- **单一职责**：每个管理器只负责一个领域
- **统一接口**：管理器提供一致的 API
- **可替换**：管理器实现可替换

---

### 4. 工厂模式 (Factory Pattern)

**哲学**：复杂对象创建逻辑封装在工厂中。

**实现**：
- **ServiceConfigurator**：服务工厂
- **ModelFactory**：模型工厂
- **PipelineBuilder**：管道构建器

**示例**：
```python
class ModelFactory:
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create_model(cls, model_name, config):
        return cls._registry[model_name](config)
```

---

## 🎯 设计目标

### 1. 可维护性 (Maintainability)

**原则**：
- **代码清晰**：命名明确，结构清晰
- **注释详细**：关键逻辑有英文注释
- **文档完善**：重要模块有 README
- **统一风格**：遵循一致的代码风格

**实现**：
- 使用类型注解：`def process(data: pd.DataFrame) -> Dict[str, Any]`
- 详细文档字符串：说明参数、返回值、异常
- 模块级 README：说明模块用途和使用方法

---

### 2. 可扩展性 (Extensibility)

**原则**：
- **开放扩展**：通过继承和接口扩展功能
- **封闭修改**：核心逻辑稳定，扩展不修改核心
- **插件化**：新功能以插件形式添加

**实现**：
- 抽象基类：`BaseWorkflow`, `BaseStrategy`, `BaseModel`
- 注册机制：`ModelFactory.register()`, `SelectorRegistry`
- 配置驱动：通过配置启用/禁用功能

---

### 3. 可测试性 (Testability)

**原则**：
- **依赖注入**：便于注入 Mock 对象
- **单一职责**：类职责单一，易于测试
- **接口抽象**：通过接口隔离，便于 Mock

**实现**：
- ServiceConfigurator 支持注入依赖
- 配置对象可序列化，便于测试
- 工具函数纯函数化，无副作用

---

### 4. 类型安全 (Type Safety)

**原则**：
- **类型注解**：所有函数有类型注解
- **配置验证**：使用 Pydantic 验证配置
- **运行时检查**：关键路径有类型检查

**实现**：
```python
def process_data(
    config: MLConfig,
    data: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """类型注解确保类型安全"""
    pass
```

---

## 📋 编码规范

### 1. 命名规范

- **类名**：PascalCase，如 `HabitatAnalysis`, `BaseWorkflow`
- **函数名**：snake_case，如 `extract_features()`, `run_pipeline()`
- **常量**：UPPER_SNAKE_CASE，如 `DEFAULT_CONFIG_PATH`
- **私有方法**：前缀 `_`，如 `_load_data()`, `_validate_config()`

### 2. 注释规范

- **代码注释**：使用英文，详细说明逻辑
- **文档字符串**：使用英文，说明参数、返回值、异常
- **类型注解**：所有函数参数和返回值必须有类型注解

### 3. 配置规范

- **配置类**：继承 `BaseConfig`，使用 Pydantic
- **配置访问**：优先使用属性访问 `config.field`，而非 `config.get('field')`
- **配置验证**：使用 `ConfigValidator` 统一验证

### 4. 错误处理

- **自定义异常**：定义领域特定的异常类
- **错误日志**：记录详细的错误信息和堆栈
- **用户友好**：向用户提供清晰的错误消息

---

## 🔄 设计演进

### 当前状态

- ✅ **配置管理统一化**：已完成 BaseConfig、ConfigValidator、ConfigAccessor
- ✅ **依赖注入模式**：ServiceConfigurator 已实现
- ✅ **工作流模式**：BaseWorkflow 已建立
- ✅ **容器模式**：PredictionContainer 已实现
- ⏳ **错误处理统一化**：进行中
- ⏳ **接口抽象**：部分完成

### 未来方向

1. **完全类型化**：所有配置访问使用 Pydantic 对象
2. **统一异常体系**：建立完整的异常层次结构
3. **接口抽象**：为 Manager 类定义抽象接口
4. **测试覆盖**：提高单元测试覆盖率
5. **性能优化**：优化大数据处理性能

---

## 📚 参考资源

### 设计模式

- **策略模式**：聚类策略、特征选择策略
- **工厂模式**：ServiceConfigurator、ModelFactory
- **模板方法模式**：BaseWorkflow、BaseStrategy
- **观察者模式**：CallbackList
- **容器模式**：PredictionContainer

### 架构原则

- **SOLID 原则**：单一职责、开闭原则、里氏替换、接口隔离、依赖倒置
- **DRY 原则**：不重复自己
- **KISS 原则**：保持简单
- **YAGNI 原则**：你不会需要它

### 相关文档

- `REFACTORING_TODO.md`：架构优化方向
- `habit/core/common/README.md`：配置管理文档
- `habit/core/habitat_analysis/README.md`：生境分析模块文档
- `habit/utils/LOG_SYSTEM_README.md`：日志系统文档

---

## 💡 最佳实践

### 1. 添加新功能

1. **定义配置类**：继承 `BaseConfig`，定义配置结构
2. **创建服务类**：实现业务逻辑
3. **扩展 ServiceConfigurator**：添加 `create_*` 方法
4. **创建 CLI 命令**：在 `cli_commands/commands/` 中添加命令
5. **编写文档**：更新相关 README

### 2. 修改现有功能

1. **保持向后兼容**：支持旧配置格式
2. **渐进式迁移**：逐步迁移到新架构
3. **更新文档**：同步更新相关文档
4. **测试验证**：确保修改不影响现有功能

### 3. 代码审查要点

- ✅ 是否使用统一的配置加载方式？
- ✅ 是否通过 ServiceConfigurator 创建服务？
- ✅ 是否使用 PredictionContainer 处理预测结果？
- ✅ 是否有适当的类型注解？
- ✅ 是否有详细的注释？
- ✅ 是否遵循命名规范？

---

## 🎓 总结

HABIT 包的设计哲学可以概括为：

1. **配置驱动**：一切行为由配置决定
2. **依赖注入**：通过配置器管理依赖
3. **工作流模式**：复杂流程抽象为工作流
4. **策略模式**：算法可互换
5. **容器模式**：数据结构统一封装
6. **模块化设计**：功能按职责划分
7. **统一工具管理**：通用功能集中管理
8. **向后兼容**：平滑演进

这些原则共同构成了 HABIT 包的设计基础，确保代码的可维护性、可扩展性和可测试性。

---

**文档版本**：v1.0  
**最后更新**：2024  
**维护者**：HABIT 开发团队
