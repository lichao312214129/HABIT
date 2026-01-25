# Tests Directory

本目录包含所有测试文件、测试工具和测试文档。

## 目录结构

### 测试文件
- `test_*.py` - 各个模块的pytest测试文件
- `test_workflow_steps.py` - 工作流步骤测试脚本（可独立运行）
- `test_end_to_end_workflow.py` - 端到端工作流测试（pytest格式）

### 工具脚本
- `create_prediction_files.py` - 生成模拟预测结果文件的工具脚本
- `analyze_tests.py` - 测试文件分析脚本（分析测试文件中的潜在问题）

### 文档文件
- `WORKFLOW_TEST_AND_FIX_GUIDE.md` - 工作流测试和修复指南
- `WORKFLOW_TEST_ANALYSIS.md` - 工作流测试分析报告
- `TEST_FIXES_SUMMARY.md` - 测试修复总结
- `TEST_ANALYSIS_REPORT.md` - 测试分析报告
- `QUICKSTART.md` - 快速开始指南
- `pytest.ini` - Pytest配置文件

## 快速开始

### 运行所有测试
```bash
# 从项目根目录
pytest tests/ -v
```

### 运行工作流步骤测试
```bash
# 从项目根目录
python tests/test_workflow_steps.py

# 或从tests目录
cd tests
python test_workflow_steps.py
```

### 运行端到端测试
```bash
pytest tests/test_end_to_end_workflow.py -v
```

### 生成预测结果文件
```bash
# 从项目根目录
python tests/create_prediction_files.py

# 或从tests目录
cd tests
python create_prediction_files.py
```

### 分析测试文件
```bash
# 从项目根目录
python tests/analyze_tests.py

# 或从tests目录
cd tests
python analyze_tests.py
```

## 测试文件说明

### `test_workflow_steps.py`
独立的工作流测试脚本，可以逐个测试每个步骤并报告错误。
- 测试步骤: preprocess → get-habitat → extract → model → compare
- 输出详细的错误信息
- 可以独立运行，不依赖pytest

### `test_end_to_end_workflow.py`
使用pytest格式的端到端工作流测试。
- 使用pytest的测试类结构
- 可以集成到CI/CD流程
- 提供测试结果摘要

### `create_prediction_files.py`
工具脚本，用于生成模型比较所需的预测结果文件。
- 基于 `breast_cancer_dataset.csv` 生成模拟预测
- 为radiomics和clinical模型生成不同的预测结果
- 生成的文件格式符合 `config_model_comparison.yaml` 的要求

### `analyze_tests.py`
测试文件分析工具，用于分析测试文件中的潜在问题。
- 检查语法错误
- 检查导入问题
- 检查缺失的配置文件引用
- 分析测试文件结构

## 文档说明

### `WORKFLOW_TEST_AND_FIX_GUIDE.md`
详细的工作流测试和修复指南，包括：
- 每个步骤的配置分析
- 潜在问题和解决方案
- 测试执行命令
- 故障排除指南

### `WORKFLOW_TEST_ANALYSIS.md`
完整的工作流测试分析报告，包括：
- 已完成的修复
- 各步骤配置分析
- 预期错误和解决方案
- 测试执行建议

### `TEST_FIXES_SUMMARY.md`
测试修复总结文档，记录所有已修复的问题和修复方法。

### `TEST_ANALYSIS_REPORT.md`
测试分析报告，详细列出所有测试文件的状态和发现的问题。

## 注意事项

1. **路径引用**: 所有测试文件使用相对路径 `Path(__file__).parent.parent` 来访问 `demo_data` 目录
2. **数据依赖**: 某些测试需要实际数据，如果数据不存在，测试会被跳过或返回退出码1（这是正常的）
3. **环境要求**: 确保在正确的Python/Conda环境中运行测试

## 相关文件位置

- 配置文件: `../demo_data/config_*.yaml`
- 测试数据: `../demo_data/`
- 源代码: `../habit/`
