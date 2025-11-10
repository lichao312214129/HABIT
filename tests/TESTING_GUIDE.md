# HABIT Testing Guide

本文档介绍如何使用 HABIT 项目的测试套件。

## 测试文件结构

```
tests/
├── __init__.py                    # 测试包初始化
├── conftest.py                    # Pytest 共享fixtures
├── pytest.ini                     # Pytest 配置文件
├── README.md                      # 测试文档（英文）
├── TESTING_GUIDE.md              # 测试指南（中文）
│
├── Debug Scripts (调试脚本 - 快速测试)
│   ├── debug_preprocess.py       # 测试图像预处理
│   ├── debug_habitat.py          # 测试栖息地分析
│   ├── debug_extract_features.py # 测试特征提取
│   ├── debug_radiomics.py        # 测试影像组学
│   ├── debug_ml.py               # 测试机器学习
│   ├── debug_kfold.py            # 测试K折交叉验证
│   ├── debug_icc.py              # 测试ICC分析
│   ├── debug_test_retest.py      # 测试重测信度
│   └── debug_compare.py          # 测试模型比较
│
├── Unit Tests (单元测试 - 全面测试)
│   ├── test_preprocessing.py     # 预处理模块测试
│   ├── test_habitat_analysis.py  # 栖息地分析测试
│   ├── test_machine_learning.py  # 机器学习测试
│   ├── test_utils.py             # 工具函数测试
│   └── test_cli.py               # CLI命令测试
│
└── Test Runners (测试运行器)
    ├── run_all_tests.py          # 运行所有单元测试
    └── run_debug_menu.py         # 交互式调试菜单
```

## 使用方式

### 1. 快速调试（Debug Scripts）

Debug脚本模拟CLI命令，适用于快速测试单个模块：

```bash
# 直接运行单个调试脚本
cd tests
python debug_preprocess.py

# 或使用交互式菜单
python run_debug_menu.py
```

**注意：** 运行前需要：
1. 在 `demo_image_data` 目录下准备相应的配置文件（如 `config_image_preprocessing.yaml`）
2. 根据你的本地路径修改debug脚本中的配置文件路径

### 2. 单元测试（Unit Tests）

单元测试使用 pytest 框架，提供全面的功能测试：

```bash
# 安装pytest（如果还没安装）
pip install pytest pytest-cov

# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_preprocessing.py -v

# 运行特定测试类
pytest tests/test_preprocessing.py::TestN4Correction -v

# 运行特定测试函数
pytest tests/test_preprocessing.py::TestN4Correction::test_n4_correction_basic -v

# 使用标记过滤测试
pytest tests/ -m "unit" -v           # 只运行单元测试
pytest tests/ -m "not slow" -v       # 跳过慢速测试

# 生成覆盖率报告
pytest tests/ --cov=habit --cov-report=html

# 使用测试运行器（推荐）
python tests/run_all_tests.py
```

### 3. 测试标记（Markers）

可以使用标记来组织和筛选测试：

```python
import pytest

@pytest.mark.unit
@pytest.mark.preprocessing
def test_n4_correction():
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline():
    pass
```

运行特定标记的测试：

```bash
pytest -m "preprocessing" -v        # 只运行预处理测试
pytest -m "unit and not slow" -v   # 运行快速单元测试
```

## 编写测试

### Debug脚本模板

所有debug脚本遵循相同的模板：

```python
# debug_<module>.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments
    sys.argv = ['habit', '<command>', '-c', '<path/to/config.yaml>']
    cli()
```

### 单元测试模板

使用 pytest 编写单元测试：

```python
# test_<module>.py
import pytest
import numpy as np

class TestMyFeature:
    """Test description"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        input_data = np.array([1, 2, 3])
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result is not None
        assert len(result) == 3
    
    def test_edge_cases(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            my_function(None)
```

### 使用Fixtures

在 `conftest.py` 中定义共享的测试数据：

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_image_3d():
    """Create a sample 3D image"""
    return np.random.randn(64, 64, 32)

# 在测试中使用
def test_image_processing(sample_image_3d):
    result = process_image(sample_image_3d)
    assert result.shape == sample_image_3d.shape
```

## 配置文件准备

在运行debug脚本前，需要准备对应的YAML配置文件：

### 1. `config_image_preprocessing.yaml` (预处理)
```yaml
data_dir: path/to/images
out_dir: path/to/output
Preprocessing:
  resample:
    images: [T1, T2]
    target_spacing: [1.0, 1.0, 1.0]
```

### 2. `config_habitat_analysis.yaml` (栖息地分析)
```yaml
data_dir: path/to/data
out_dir: path/to/results
Clustering:
  method: kmeans
  n_clusters: 3
```

### 3. 其他配置文件
根据具体模块需求创建相应配置文件。

## 持续集成（CI）

可以在CI/CD流程中运行测试：

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=habit
```

## 最佳实践

1. **测试驱动开发（TDD）**：先编写测试，再实现功能
2. **保持测试独立**：每个测试应该独立运行
3. **使用有意义的测试名称**：清楚描述测试的目的
4. **测试边界条件**：包括正常情况和异常情况
5. **保持测试简洁**：一个测试只测试一个功能点
6. **使用fixtures**：复用测试数据和设置
7. **定期运行测试**：每次修改代码后都运行相关测试
8. **维护测试覆盖率**：关键模块保持80%以上覆盖率

## 故障排查

### 问题：找不到模块
```bash
# 确保安装了HABIT包
pip install -e .
```

### 问题：配置文件路径错误
```bash
# 检查debug脚本中的路径是否正确
# 使用绝对路径或正确的相对路径
```

### 问题：测试失败
```bash
# 查看详细的错误信息
pytest tests/ -v --tb=long

# 进入调试模式
pytest tests/ --pdb
```

## 参考资源

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## 常见问题

**Q: 如何跳过某个测试？**
```python
@pytest.mark.skip(reason="Not implemented yet")
def test_feature():
    pass
```

**Q: 如何测试异常？**
```python
def test_raises_error():
    with pytest.raises(ValueError):
        function_that_raises()
```

**Q: 如何测试警告？**
```python
def test_warning():
    with pytest.warns(UserWarning):
        function_that_warns()
```

**Q: 如何参数化测试？**
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    assert input * 2 == expected
```

