# 测试快速开始指南

## 5分钟快速上手

### 步骤1：安装测试依赖

```bash
pip install pytest pytest-cov
```

### 步骤2：验证安装

```bash
# 检查 pytest 是否安装成功
pytest --version

# 应该看到类似输出：
# pytest 7.x.x
```

### 步骤3：运行第一个测试

```bash
# 测试 CLI 帮助功能（最简单的测试）
cd tests
pytest test_cli.py::TestCLICommands::test_cli_help -v
```

### 步骤4：使用交互式菜单

```bash
# 运行交互式调试菜单
python tests/run_debug_menu.py
```

会看到如下菜单：

```
================================================================================
HABIT Debug Script Menu
================================================================================

1. Image Preprocessing Pipeline
2. Habitat Analysis
3. Feature Extraction
4. Radiomics Extraction
5. Machine Learning Pipeline
6. K-Fold Cross Validation
7. ICC Analysis
8. Test-Retest Reliability
9. Model Comparison
q. Quit

================================================================================
Select an option:
```

选择一个数字，按回车即可运行对应的调试脚本。

### 步骤5：运行所有测试

```bash
# 运行所有单元测试
python tests/run_all_tests.py

# 或直接使用 pytest
pytest tests/ -v
```

## 常用测试命令速查

```bash
# ============ 运行测试 ============
pytest tests/                              # 运行所有测试
pytest tests/test_cli.py                  # 运行单个文件
pytest tests/ -k "test_cli"               # 运行名称包含"cli"的测试
pytest tests/ -v                          # 详细输出
pytest tests/ -s                          # 显示print输出

# ============ 调试 ============
pytest tests/ --pdb                       # 失败时进入调试器
pytest tests/ -x                          # 遇到第一个失败就停止
pytest tests/ --lf                        # 只运行上次失败的测试
pytest tests/ --ff                        # 先运行上次失败的

# ============ 覆盖率 ============
pytest tests/ --cov=habit                 # 生成覆盖率报告
pytest tests/ --cov=habit --cov-report=html  # HTML覆盖率报告

# ============ 标记筛选 ============
pytest tests/ -m "unit"                   # 只运行单元测试
pytest tests/ -m "not slow"               # 跳过慢速测试
pytest tests/ -m "preprocessing"          # 只运行预处理测试

# ============ 并行运行 ============
pytest tests/ -n auto                     # 自动并行（需要 pytest-xdist）
```

## 调试脚本使用示例

### 示例1：测试图像预处理

1. 准备配置文件（`demo_image_data/config_image_preprocessing.yaml`）：

```yaml
data_dir: F:/work/research/radiomics_TLSs/habit_project/demo_image_data/nii/processed_images
out_dir: F:/work/research/radiomics_TLSs/habit_project/demo_image_data/nii/preprocessed
auto_select_first_file: true

Preprocessing:
  resample:
    images: [T1, T2]
    target_spacing: [1.0, 1.0, 1.0]
```

2. 运行调试脚本：

```bash
cd tests
python debug_preprocess.py
```

### 示例2：测试机器学习

1. 准备配置文件（`demo_image_data/config_ml.yaml`）：

```yaml
data_file: path/to/features.csv
label_col: target
out_dir: path/to/results

FeatureSelection:
  - method: variance
    threshold: 0.01
  - method: lasso
    alpha: 0.01

Model:
  name: logistic_regression
  params:
    max_iter: 1000

processes: 1
random_state: 42
```

2. 运行调试脚本：

```bash
python debug_ml.py
```

## 编写你的第一个测试

在 `tests/` 目录下创建新文件 `test_my_feature.py`：

```python
# test_my_feature.py
import pytest
import numpy as np

class TestMyFeature:
    """Test my custom feature"""
    
    def test_simple_case(self):
        """Test a simple case"""
        # Arrange (准备)
        input_value = 5
        
        # Act (执行)
        result = input_value * 2
        
        # Assert (断言)
        assert result == 10
    
    def test_with_fixture(self, sample_image_3d):
        """Test using a fixture from conftest.py"""
        # sample_image_3d 会自动从 conftest.py 注入
        assert sample_image_3d.ndim == 3
        assert sample_image_3d.shape == (64, 64, 32)
```

运行你的测试：

```bash
pytest tests/test_my_feature.py -v
```

## 下一步

- 阅读 `TESTING_GUIDE.md` 了解详细测试指南
- 阅读 `README.md` 了解测试文件结构
- 查看现有测试文件，学习更多测试技巧
- 为新功能编写测试

## 获取帮助

- Pytest文档：https://docs.pytest.org/
- HABIT项目文档：查看项目根目录的 README.md
- 遇到问题？查看 `TESTING_GUIDE.md` 的故障排查章节

