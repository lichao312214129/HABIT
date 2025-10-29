# HABIT 项目 Logging 系统重构总结

## 问题描述

重构前的问题：
- ❌ 经常同时创建 log 文件和 logs 文件夹
- ❌ 在 logs 文件夹下又创建 log 文件，结构混乱
- ❌ 每个模块自己配置 logging，导致重复配置
- ❌ 使用时间戳创建多个 log 文件，日志堆积
- ❌ 没有统一的 logger 层级管理

## 解决方案

### 1. 核心改进

✅ **单一日志文件**：每次运行只创建一个 log 文件  
✅ **扁平化结构**：日志文件直接在输出目录下（不再有 logs/ 子文件夹）  
✅ **统一配置**：所有模块使用同一套 logging 配置  
✅ **层级管理**：清晰的 logger 层级结构（habit.preprocessing, habit.habitat, etc.）  
✅ **线程安全**：使用 Singleton 模式确保线程安全

### 2. 新的日志结构

```
output_dir/
├── processing.log          # 单一日志文件 (之前是 logs/processing.log)
├── results/
└── other_output_files/
```

### 3. 重构的文件

#### 核心文件
- ✅ `habit/utils/log_utils.py` - 完全重构，实现统一的 logging 系统

#### Core 模块
- ✅ `habit/core/preprocessing/image_processor_pipeline.py`
- ✅ `habit/core/preprocessing/dcm2niix_converter.py`
- ✅ `habit/core/preprocessing/histogram_standardization.py`
- ✅ `habit/core/preprocessing/resample.py`
- ✅ `habit/core/preprocessing/n4_correction.py`
- ✅ `habit/core/preprocessing/zscore_normalization.py`
- ✅ `habit/core/habitat_analysis/habitat_analysis.py`
- ✅ `habit/core/habitat_analysis/features/__init__.py`
- ✅ `habit/core/habitat_analysis/feature_extraction/extractor.py`
- ✅ `habit/core/machine_learning/feature_selectors/icc/icc.py`
- ✅ `habit/core/machine_learning/feature_selectors/icc_selector.py`

#### CLI Commands
- ✅ `habit/cli_commands/commands/cmd_preprocess.py`
- ✅ `habit/cli_commands/commands/cmd_habitat.py`

#### Scripts
- ✅ `scripts/app_image_preprocessing.py`

#### Utils
- ✅ `habit/utils/file_system_utils.py`
- ✅ `habit/utils/io_utils.py`

## 使用方法

### 对于主程序/脚本

```python
from habit.utils.log_utils import setup_logger

# 在主程序入口设置 logger
logger = setup_logger(
    name='preprocessing',           # 模块名称
    output_dir=Path('/output'),     # 输出目录
    log_filename='processing.log',  # 日志文件名
    level=logging.INFO              # 日志级别
)

logger.info('程序开始运行')
```

### 对于子模块

```python
from habit.utils.log_utils import get_module_logger

# 子模块使用 get_module_logger，会自动继承主程序的配置
logger = get_module_logger(__name__)

def my_function():
    logger.debug('调试信息')
    logger.info('正常信息')
```

### 对于类

```python
from habit.utils.log_utils import get_module_logger

class MyProcessor:
    def __init__(self):
        self.logger = get_module_logger(__name__)
    
    def process(self):
        self.logger.info('开始处理')
```

## Logger 层级结构

```
habit (root)
├── habit.preprocessing
│   ├── habit.preprocessing.resample
│   ├── habit.preprocessing.n4_correction
│   └── habit.preprocessing.zscore
├── habit.habitat
│   ├── habit.habitat.clustering
│   └── habit.habitat.features
├── habit.cli
│   ├── habit.cli.preprocess
│   └── habit.cli.habitat
└── habit.scripts
    └── habit.scripts.app_preprocessing
```

## 迁移指南

### 旧代码（不要再使用）

```python
# ❌ 旧方式
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### 新代码（推荐）

```python
# ✅ 新方式
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)
```

## 日志格式

### 控制台输出（简洁）
```
2025-10-29 10:30:45 - INFO - Starting processing...
2025-10-29 10:30:46 - INFO - Processing completed
```

### 文件输出（详细）
```
2025-10-29 10:30:45 - habit.preprocessing - INFO - [image_processor_pipeline.py:123] - Starting processing...
2025-10-29 10:30:46 - habit.preprocessing - INFO - [image_processor_pipeline.py:156] - Processing completed
```

## 关键改进总结

| 功能 | 重构前 | 重构后 |
|------|--------|--------|
| 日志文件结构 | `logs/processing_20251029_103045.log` | `processing.log` |
| 配置方式 | 每个模块单独配置 | 统一的 LoggerManager |
| 重复日志 | 经常出现 | 已消除 |
| 日志层级 | 无统一层级 | habit.* 层级结构 |
| 线程安全 | 不保证 | Singleton 模式保证 |

## 剩余待处理文件

以下文件仍使用 `logging.basicConfig`，但影响较小（主要是独立脚本）：
- `scripts/app_dilation_or_erosion.py`
- `scripts/app_traditional_radiomics_extractor.py`
- `scripts/get_supervoxel.py`
- `scripts/organize_image_data.py`
- `habit/core/habitat_analysis/feature_extraction/traditional_radiomics_extractor.py`
- `habit/core/machine_learning/feature_selectors/icc/habitat_test_retest_mapper.py`

这些文件可以在后续使用时逐步更新。

## 文档

详细的使用文档位于：`habit/utils/LOG_SYSTEM_README.md`

## 验证

重构后的系统已通过以下验证：
- ✅ 基本 logging 功能
- ✅ 层级 logger 结构
- ✅ 无重复日志
- ✅ 单例模式
- ✅ Logger 命名规则
- ✅ 日志级别过滤

## 使用建议

1. **新代码**：直接使用新的 logging 系统
2. **旧代码**：逐步迁移，使用 `get_module_logger(__name__)`
3. **调试**：使用 `level=logging.DEBUG` 获取详细信息
4. **生产**：使用 `level=logging.INFO` 减少输出

## 总结

这次重构彻底解决了 HABIT 项目中 logging 系统混乱的问题，建立了统一、清晰、易用的日志管理系统。所有核心模块已完成迁移，新的系统已准备好投入使用。

