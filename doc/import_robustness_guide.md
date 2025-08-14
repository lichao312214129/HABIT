# HABIT Package Import Robustness Guide

## 概述

HABIT包现在包含了强大的导入容错机制，确保即使在部分模块导入失败的情况下，包仍然可以正常使用。这个机制提供了：

1. **优雅的错误处理** - 导入失败不会导致整个包崩溃
2. **状态跟踪** - 可以查询哪些模块可用，哪些失败
3. **警告系统** - 自动显示导入错误信息
4. **实用工具** - 提供检查和诊断功能

## 基本使用

### 导入包

```python
import habit

# 检查包版本
print(f"HABIT version: {habit.__version__}")

# 检查可用的模块
available_modules = habit.get_available_modules()
print(f"Available modules: {list(available_modules.keys())}")

# 检查导入错误
import_errors = habit.get_import_errors()
if import_errors:
    print(f"Import errors: {import_errors}")
```

### 检查模块可用性

```python
import habit

# 检查特定模块是否可用
if habit.is_module_available('HabitatAnalysis'):
    analyzer = habit.HabitatAnalysis()
    # 使用分析器...
else:
    print("HabitatAnalysis is not available")
    print(f"Error: {habit.get_import_errors().get('HabitatAnalysis')}")

if habit.is_module_available('Modeling'):
    model = habit.Modeling()
    # 使用模型...
else:
    print("Modeling is not available")
```

## 高级功能

### 使用ImportManager

`ImportManager`类提供了更高级的导入管理功能：

```python
from habit.utils.import_utils import ImportManager

# 创建导入管理器
manager = ImportManager()

# 安全导入模块
numpy = manager.safe_import('numpy', alias='np')
pandas = manager.safe_import('pandas', alias='pd')

# 安全导入类
rf_classifier = manager.safe_import('sklearn.ensemble', 'RandomForestClassifier', 'RFC')

# 批量导入
imports = [
    ('matplotlib.pyplot', None, 'plt'),
    ('seaborn', None, 'sns'),
    ('sklearn.metrics', 'accuracy_score', 'acc_score'),
]

results = manager.safe_import_multiple(imports)

# 检查导入状态
manager.print_import_status(verbose=True)

# 获取错误信息
errors = manager.get_import_errors()
warnings = manager.get_import_warnings()
```

### 依赖检查

```python
from habit.utils.import_utils import check_dependencies

# 检查必需和可选依赖
required_modules = ['numpy', 'pandas', 'sklearn']
optional_modules = ['matplotlib', 'seaborn', 'plotly']

status = check_dependencies(required_modules, optional_modules)

for module, available in status.items():
    if available:
        print(f"✓ {module} is available")
    else:
        print(f"✗ {module} is not available")
```

### 模块信息查询

```python
from habit.utils.import_utils import get_module_info

# 获取模块详细信息
modules_to_check = ['numpy', 'pandas', 'nonexistent_module']

for module in modules_to_check:
    info = get_module_info(module)
    print(f"\n{module}:")
    print(f"  Available: {info['available']}")
    print(f"  Version: {info['version']}")
    print(f"  Path: {info['path']}")
    if info['error']:
        print(f"  Error: {info['error']}")
```

### 装饰器使用

```python
from habit.utils.import_utils import safe_import_decorator

# 使用装饰器安全导入
@safe_import_decorator('matplotlib.pyplot', alias='plt', default_value=None)
def plot_data(data, plt):
    if plt is None:
        print("matplotlib.pyplot is not available, skipping plot")
        return
    
    plt.plot(data)
    plt.show()

# 调用函数
plot_data([1, 2, 3, 4, 5])
```

## 错误处理最佳实践

### 1. 检查模块可用性

```python
import habit

def safe_analysis():
    if not habit.is_module_available('HabitatAnalysis'):
        print("HabitatAnalysis is not available")
        return None
    
    try:
        analyzer = habit.HabitatAnalysis()
        return analyzer
    except Exception as e:
        print(f"Error creating analyzer: {e}")
        return None
```

### 2. 提供替代方案

```python
from habit.utils.import_utils import ImportManager

def get_plotting_backend():
    manager = ImportManager()
    
    # 尝试不同的绘图后端
    backends = [
        ('matplotlib.pyplot', None, 'plt'),
        ('plotly.express', None, 'px'),
        ('seaborn', None, 'sns'),
    ]
    
    for module_path, class_name, alias in backends:
        backend = manager.safe_import(module_path, class_name, alias)
        if backend is not None:
            print(f"Using {alias} for plotting")
            return backend
    
    print("No plotting backend available")
    return None
```

### 3. 条件功能启用

```python
import habit

class FeatureProcessor:
    def __init__(self):
        self.has_ml = habit.is_module_available('Modeling')
        self.has_analysis = habit.is_module_available('HabitatAnalysis')
    
    def process_features(self, data):
        if self.has_analysis:
            # 使用HabitatAnalysis处理
            analyzer = habit.HabitatAnalysis()
            return analyzer.process(data)
        else:
            # 使用基础处理
            return self.basic_process(data)
    
    def train_model(self, X, y):
        if self.has_ml:
            # 使用Modeling训练
            model = habit.Modeling()
            return model.train(X, y)
        else:
            print("Machine learning module not available")
            return None
```

## 测试导入健壮性

运行测试脚本来验证导入容错机制：

```bash
python scripts/test_import_robustness.py
```

这个脚本会测试：
- 基本导入功能
- 模块可用性检查
- 导入工具功能
- 优雅失败处理

## 常见问题

### Q: 如何知道哪些模块导入失败了？

A: 使用`habit.get_import_errors()`获取详细的错误信息：

```python
import habit

errors = habit.get_import_errors()
for module, error in errors.items():
    print(f"{module}: {error}")
```

### Q: 如何强制重新导入模块？

A: 使用Python的`importlib.reload()`：

```python
import importlib
import habit

# 重新加载模块
importlib.reload(habit)
```

### Q: 如何禁用导入警告？

A: 使用Python的warnings模块：

```python
import warnings
warnings.filterwarnings("ignore", category=ImportWarning)

import habit  # 不会显示导入警告
```

### Q: 如何检查特定版本的模块？

A: 使用`get_module_info()`函数：

```python
from habit.utils.import_utils import get_module_info

info = get_module_info('numpy')
if info['available'] and info['version']:
    print(f"NumPy version: {info['version']}")
```

## 总结

HABIT包的导入容错机制提供了：

1. **可靠性** - 包可以在部分模块缺失的情况下正常工作
2. **透明度** - 清楚了解哪些功能可用，哪些不可用
3. **灵活性** - 可以根据可用模块调整功能
4. **调试能力** - 详细的错误信息和状态跟踪

这个机制确保了HABIT包在各种环境中的稳定性和可用性。 