# HABIT 统一配置管理系统

本文档介绍 HABIT 的统一配置管理系统，提供了类型安全、统一的配置加载和验证机制。

## 核心组件

### 1. BaseConfig - 配置基类

所有配置类都应继承自 `BaseConfig`，它提供了：

- **统一的加载方法**：`from_dict()`, `from_file()`
- **类型安全**：基于 Pydantic 的验证
- **灵活的访问方式**：支持属性访问和字典访问
- **错误处理**：统一的异常类型

```python
from habit.core.common.config_base import BaseConfig
from pydantic import Field

class MyConfig(BaseConfig):
    data_dir: str = Field(..., description="Data directory")
    out_dir: str = Field(..., description="Output directory")
    processes: int = Field(2, gt=0)
```

### 2. ConfigValidator - 配置验证器

统一的配置验证和加载入口：

```python
from habit.core.common.config_validator import ConfigValidator
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

# 从文件加载并验证
config = ConfigValidator.validate_and_load(
    config_path='config.yaml',
    config_class=HabitatAnalysisConfig
)

# 从字典验证
config = ConfigValidator.validate_dict(
    config_dict={'data_dir': '/path/to/data', ...},
    config_class=HabitatAnalysisConfig
)
```

### 3. ConfigAccessor - 配置访问器

提供统一的配置访问接口，支持字典和 Pydantic 模型：

```python
from habit.core.common.config_base import ConfigAccessor

# 创建访问器
accessor = ConfigAccessor(config)

# 访问配置值（支持点号访问）
value = accessor.get('data_dir')
nested_value = accessor.get('FeatureConstruction.voxel_level.method')

# 检查配置是否存在
if accessor.has('processes'):
    processes = accessor.get('processes', default=1)
```

## 使用示例

### 示例 1: 加载 Habitat 分析配置

```python
from habit.core.common.config_validator import load_and_validate_config
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

# 方式1: 使用便捷函数
config = load_and_validate_config(
    'demo_data/config_habitat.yaml',
    HabitatAnalysisConfig
)

# 方式2: 使用验证器
from habit.core.common.config_validator import ConfigValidator
config = ConfigValidator.validate_and_load(
    'demo_data/config_habitat.yaml',
    HabitatAnalysisConfig
)

# 访问配置（推荐使用属性访问）
print(config.data_dir)
print(config.FeatureConstruction.voxel_level.method)

# 向后兼容的字典访问
print(config.get('data_dir'))
print(config['out_dir'])
```

### 示例 2: 在类中使用配置

```python
from habit.core.common.config_base import ConfigAccessor
from habit.core.machine_learning.config_schemas import MLConfig

class MyWorkflow:
    def __init__(self, config: MLConfig):
        # 使用 ConfigAccessor 统一访问
        self.config_accessor = ConfigAccessor(config)
        
        # 获取配置值
        self.output_dir = self.config_accessor.get('output', './results')
        self.random_state = self.config_accessor.get('random_state', 42)
        
        # 获取配置节
        normalization_config = self.config_accessor.get_section('normalization')
```

### 示例 3: 错误处理

```python
from habit.core.common.config_validator import ConfigValidator
from habit.core.common.config_base import ConfigValidationError
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

try:
    config = ConfigValidator.validate_and_load(
        'config.yaml',
        HabitatAnalysisConfig
    )
except ConfigValidationError as e:
    print(f"配置验证失败: {e.message}")
    print(f"配置文件: {e.config_path}")
    print(f"详细错误: {e.errors}")
except FileNotFoundError as e:
    print(f"配置文件未找到: {e}")
```

## 迁移指南

### 从字典访问迁移到强类型访问

**旧方式**（字典访问）：
```python
config = load_config('config.yaml')
data_dir = config.get('data_dir')
processes = config.get('processes', 2)
```

**新方式**（强类型访问）：
```python
from habit.core.common.config_validator import load_and_validate_config
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

config = load_and_validate_config('config.yaml', HabitatAnalysisConfig)
data_dir = config.data_dir  # 类型安全，IDE 自动补全
processes = config.processes  # 有默认值，无需 get()
```

### 向后兼容

为了保持向后兼容，`BaseConfig` 仍然支持字典式访问：

```python
# 这些方式仍然可用
config.get('data_dir')
config['out_dir']
'data_dir' in config
```

但推荐使用属性访问，因为它提供：
- 类型安全
- IDE 自动补全
- 编译时检查

## 最佳实践

1. **优先使用属性访问**：`config.field_name` 而不是 `config.get('field_name')`
2. **使用 ConfigValidator**：统一使用 `ConfigValidator.validate_and_load()` 加载配置
3. **处理异常**：捕获 `ConfigValidationError` 提供友好的错误信息
4. **类型注解**：在函数签名中使用配置类型，而不是 `Dict[str, Any]`
5. **渐进式迁移**：可以逐步从字典访问迁移到强类型访问

## 相关文件

- `habit/core/common/config_base.py` - 配置基类和访问器
- `habit/core/common/config_validator.py` - 配置验证器
- `habit/core/habitat_analysis/config_schemas.py` - Habitat 分析配置
- `habit/core/machine_learning/config_schemas.py` - 机器学习配置
- `habit/core/preprocessing/config_schemas.py` - 预处理配置
