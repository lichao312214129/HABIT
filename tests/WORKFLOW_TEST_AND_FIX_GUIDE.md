# 工作流测试和修复指南

## 概述
本指南用于测试从预处理到模型比较的完整工作流，并根据报错修复配置文件和测试代码。

## 测试步骤

### 步骤1: Preprocess（预处理）

#### 配置文件: `../demo_data/config_preprocessing.yaml`
- **状态**: ✅ 配置文件存在
- **潜在问题**:
  1. `data_dir` 指向 `./files_preprocessing.yaml` - 需要确保文件存在
  2. `dcm2niix_path: ./dcm2niix.exe` - 需要确保可执行文件存在
  3. 如果DICOM数据不存在，预处理会失败（这是预期的）

#### 测试文件: `test_preprocess.py`
- **状态**: ✅ 测试文件存在
- **修复建议**: 
  - 测试应该能够处理数据不存在的情况（使用 `pytest.skip` 或接受退出码1）

#### 运行测试:
```bash
# 从项目根目录运行
pytest tests/test_preprocess.py -v

# 或使用测试脚本（从tests目录）
cd tests
python test_workflow_steps.py
```

---

### 步骤2: Get-Habitat（生境分割）

#### 配置文件: `../demo_data/config_habitat.yaml`
- **状态**: ✅ 配置文件存在
- **潜在问题**:
  1. `data_dir` 指向 `./file_habitat.yaml` - 需要确保文件存在
  2. `out_dir: ./results/habitats/two_step` - 输出目录会被自动创建
  3. 如果预处理数据不存在，会失败（这是预期的）

#### 测试文件: `test_habitat.py`
- **状态**: ✅ 测试文件存在
- **修复建议**: 
  - 测试应该能够处理数据不存在的情况

#### 运行测试:
```bash
pytest tests/test_habitat.py::TestGetHabitatCommand::test_get_habitat_with_config -v
```

---

### 步骤3: Extract Features（特征提取）

#### 配置文件: `../demo_data/config_extract_features.yaml`
- **状态**: ✅ 配置文件存在
- **潜在问题**:
  1. `raw_img_folder: ./preprocessed/processed_images` - 需要预处理步骤完成
  2. `habitats_map_folder: ./results/habitat` - 需要生境分割步骤完成
  3. `params_file_of_non_habitat: ./parameter.yaml` - 需要确保文件存在
  4. `params_file_of_habitat: ./parameter_habitat.yaml` - 需要确保文件存在

#### 测试文件: `test_extract_features.py`
- **状态**: ✅ 测试文件存在

#### 运行测试:
```bash
pytest tests/test_extract_features.py -v
```

---

### 步骤4: Model Train（模型训练）

#### 配置文件: `../demo_data/config_machine_learning_clinical.yaml`
- **状态**: ✅ 配置文件存在
- **潜在问题**:
  1. `input[0].path: ./ml_data/clinical_feature.csv` - 需要确保文件存在
  2. `train_ids_file: ./ml_data/train_ids.txt` - 需要确保文件存在
  3. `test_ids_file: ./ml_data/test_ids.txt` - 需要确保文件存在
  4. 如果特征文件不存在，会失败（这是预期的）

#### 测试文件: `test_ml.py`
- **状态**: ✅ 测试文件存在

#### 运行测试:
```bash
pytest tests/test_ml.py::TestModelCommand::test_model_train_with_config -v
```

---

### 步骤5: Compare（模型比较）

#### 配置文件: `../demo_data/config_model_comparison.yaml`
- **状态**: ✅ 已修复（文件名问题）
- **修复内容**:
  1. ✅ 修复了文件名中的中文和空格: `all_prediction_results - 副本.csv` → `all_prediction_results.csv`
  2. ✅ 创建了模拟的预测结果文件:
     - `../demo_data/ml_data/radiomics/all_prediction_results.csv`
     - `../demo_data/ml_data/clinical/all_prediction_results.csv`

#### 测试文件: `test_compare.py`
- **状态**: ✅ 测试文件存在
- **之前修复的问题**:
  1. ✅ `ModelComparison` 的 `visualization` 属性访问问题已修复
  2. ✅ `PlotManager` 的字典类型config处理已修复
  3. ✅ `ServiceConfigurator` 的config传递已修复

#### 运行测试:
```bash
pytest tests/test_compare.py -v
```

---

## 已知问题和修复

### 1. Compare命令的visualization错误 ✅ 已修复
- **错误**: `'dict' object has no attribute 'visualization'`
- **修复**: 
  - `ModelComparison.__init__` 中添加了字典到Pydantic模型的转换
  - `PlotManager.__init__` 中添加了字典类型config的处理
  - `ServiceConfigurator.create_plot_manager` 中修复了config传递

### 2. 配置文件路径问题 ✅ 已修复
- **问题**: `config_model_comparison.yaml` 中引用的文件路径包含中文和空格
- **修复**: 
  - 修改为: `all_prediction_results.csv`
  - 创建了模拟的预测结果文件

### 3. 预测结果文件缺失 ✅ 已创建
- **问题**: `all_prediction_results.csv` 文件不存在
- **修复**: 
  - 创建了简化的测试文件
  - 提供了 `create_prediction_files.py` 脚本用于生成完整文件

---

## 测试执行顺序

### 推荐测试顺序（如果数据完整）:
1. **Preprocess** → 生成预处理数据
2. **Get-Habitat** → 生成生境图
3. **Extract Features** → 提取特征
4. **Model Train** → 训练模型（生成预测结果）
5. **Compare** → 比较模型

### 如果数据不完整:
- 每个步骤的测试应该能够优雅地处理数据缺失
- 测试应该返回退出码1而不是崩溃
- 使用 `pytest.skip` 跳过需要数据的测试

---

## 快速测试命令

### 测试所有步骤（使用测试脚本）:
```bash
# 从项目根目录
cd tests
python test_workflow_steps.py

# 或从项目根目录
python tests/test_workflow_steps.py
```

### 测试单个步骤:
```bash
# Preprocess
pytest tests/test_preprocess.py -v

# Get-Habitat
pytest tests/test_habitat.py::TestGetHabitatCommand::test_get_habitat_with_config -v

# Extract
pytest tests/test_extract_features.py -v

# Model Train
pytest tests/test_ml.py::TestModelCommand::test_model_train_with_config -v

# Compare
pytest tests/test_compare.py -v
```

### 只测试帮助命令（不需要数据）:
```bash
pytest tests/ -v -k "help"
```

---

## 生成完整的预测结果文件

如果需要完整的预测结果文件（而不是简化的测试文件），运行:

```bash
# 从项目根目录
python tests/create_prediction_files.py

# 或从tests目录
cd tests
python create_prediction_files.py
```

这将基于 `breast_cancer_dataset.csv`、`train_ids.txt` 和 `test_ids.txt` 生成完整的预测结果文件。

---

## 预期结果

### 成功的测试:
- ✅ 所有帮助命令测试应该通过
- ✅ 配置文件验证应该通过
- ✅ 命令应该能够正确解析参数

### 可能失败的测试（数据依赖）:
- ⚠️ 需要实际数据的测试可能会失败（这是预期的）
- ⚠️ 这些测试会返回退出码1，但不会崩溃
- ⚠️ 测试应该使用 `pytest.skip` 或接受退出码1

---

## 故障排除

### 问题1: 配置文件找不到
- **检查**: 确保配置文件路径正确
- **修复**: 使用相对路径或绝对路径

### 问题2: 数据文件找不到
- **检查**: 确保前置步骤已运行
- **修复**: 按顺序运行步骤，或使用 `pytest.skip` 跳过

### 问题3: 列名不匹配
- **检查**: 确保CSV文件的列名与配置匹配
- **修复**: 更新配置文件或CSV文件

### 问题4: 路径问题
- **检查**: 确保所有路径使用正确的分隔符（Windows使用 `\`，但YAML中可以使用 `/`）
- **修复**: 使用相对路径，或确保路径正确

---

## 下一步

1. 在正确的Python环境中运行测试
2. 查看每个步骤的错误输出
3. 根据错误修复配置文件和代码
4. 重新运行测试直到所有步骤通过
