# Predict 模式使用指南

HABIT 支持两种主要的 predict 模式：
1. **生境分割 predict 模式** - 使用已训练的生境分析 pipeline 对新数据进行生境分割
2. **机器学习 predict 模式** - 使用已训练的机器学习模型对新数据进行预测

---

## 1. 生境分割 Predict 模式

### 概述

生境分割的 predict 模式允许您使用已训练的生境分析 pipeline 对新数据集进行生境分割，而无需重新训练模型。这对于：
- 在新数据上应用已训练好的生境分割模型
- 保持生境标签的一致性（特别是 two_step 和 direct_pooling 策略）
- 快速处理新数据，无需重新训练

### 配置文件

HABIT 为三种聚类策略都提供了 predict 模式的配置文件：

#### 1.1 Two-Step 策略 Predict 模式

**配置文件**: `demo_data/config_habitat.yaml`

```yaml
run_mode: predict
pipeline_path: ./results/habitat/train/habitat_pipeline.pkl  # 必需：指向已训练的 pipeline
data_dir: ./file_habitat.yaml
out_dir: ./results/habitat/predict/two_step
```

**使用方法**:
```bash
habit get-habitat --config demo_data/config_habitat.yaml --mode predict
```

#### 1.2 One-Step 策略 Predict 模式

**配置文件**: `demo_data/config_habitat_one_step_predict.yaml`

```yaml
run_mode: predict
pipeline_path: ./results/habitat_one_step/train/habitat_pipeline.pkl  # 必需
data_dir: ./file_habitat.yaml
out_dir: ./results/habitat_one_step/predict
```

**使用方法**:
```bash
habit get-habitat --config demo_data/config_habitat_one_step_predict.yaml --mode predict
```

#### 1.3 Direct Pooling 策略 Predict 模式

**配置文件**: `demo_data/config_habitat_direct_pooling_predict.yaml`

```yaml
run_mode: predict
pipeline_path: ./results/habitat_direct_pooling/train/habitat_pipeline.pkl  # 必需
data_dir: ./file_habitat.yaml
out_dir: ./results/habitat_direct_pooling/predict
```

**使用方法**:
```bash
habit get-habitat --config demo_data/config_habitat_direct_pooling_predict.yaml --mode predict
```

### 关键配置项说明

#### `pipeline_path` (必需)

- **作用**: 指定已训练的生境分析 pipeline 文件路径
- **格式**: `.pkl` 文件
- **位置**: 通常在训练模式的输出目录中，例如：
  - `./results/habitat/train/habitat_pipeline.pkl`
  - `./results/habitat_one_step/train/habitat_pipeline.pkl`
  - `./results/habitat_direct_pooling/train/habitat_pipeline.pkl`

#### `run_mode`

- **值**: `predict` 或 `train`
- **说明**: 设置为 `predict` 时，HABIT 会加载已训练的 pipeline 而不是重新训练

#### `data_dir`

- **格式**: 指向数据配置文件（YAML 格式），与训练时使用的格式相同
- **内容**: 包含新数据集的路径信息

### 命令行参数

您也可以通过命令行参数覆盖配置文件中的设置：

```bash
# 覆盖 mode
habit get-habitat --config config.yaml --mode predict

# 覆盖 pipeline_path
habit get-habitat --config config.yaml --mode predict --pipeline ./custom/pipeline.pkl
```

---

## 2. 机器学习 Predict 模式

### 概述

机器学习的 predict 模式允许您使用已训练的模型对新数据进行预测。这对于：
- 在新患者数据上应用已训练的预测模型
- 批量预测
- 模型部署

### 配置文件

**配置文件**: `demo_data/config_predict.yaml`

```yaml
# 必需字段
model_path: ./ml_data/clinical/models/LogisticRegression_final_pipeline.pkl
data_path: ./ml_data/clinical_feature.csv
output_dir: ./ml_data/clinical/predictions

# 可选字段
evaluate: true  # 是否评估性能（需要真实标签）
label_col: label  # 真实标签列名（如果 evaluate=true）
model_name: MyBestModel  # 模型名称（用于日志）
```

### 使用方法

```bash
habit model --config demo_data/config_predict.yaml --mode predict
```

### 关键配置项说明

#### `model_path` (必需)

- **作用**: 指定已训练的模型文件路径
- **格式**: `.pkl` 或 `.joblib` 文件
- **位置**: 通常在训练模式的输出目录中，例如：
  - `./ml_data/clinical/models/LogisticRegression_final_pipeline.pkl`

#### `data_path` (必需)

- **作用**: 新数据的 CSV 文件路径
- **要求**: 
  - 必须包含训练时使用的所有特征列
  - 特征列名必须与训练时一致
  - 标签列是可选的（如果 `evaluate=false`）

#### `output_dir` (必需)

- **作用**: 预测结果的输出目录
- **输出内容**:
  - `prediction_results.csv` - 预测结果（包含预测标签和概率）
  - `evaluation_metrics.csv` - 评估指标（如果 `evaluate=true`）

#### `evaluate` (可选)

- **默认值**: `false`
- **说明**: 
  - 如果 `true`，数据必须包含真实标签列
  - 会计算评估指标（AUC, Accuracy, Sensitivity, Specificity 等）
  - 如果 `false`，只进行预测，不评估

#### `label_col` (可选)

- **作用**: 指定真实标签列名
- **自动检测**: 如果未指定，会尝试从以下常见列名中自动检测：
  - `['label', 'Target', 'class', 'diagnosis', 'outcome', 'y']`

---

## 3. 完整工作流程示例

### 示例 1: 生境分割 Predict 模式

```bash
# Step 1: 训练生境分析模型（two-step 策略）
habit get-habitat --config demo_data/config_habitat.yaml --mode train

# Step 2: 使用训练好的模型对新数据进行预测
habit get-habitat --config demo_data/config_habitat.yaml --mode predict
```

### 示例 2: 机器学习 Predict 模式

```bash
# Step 1: 训练机器学习模型
habit model --config demo_data/config_machine_learning_clinical.yaml --mode train

# Step 2: 使用训练好的模型对新数据进行预测
habit model --config demo_data/config_predict.yaml --mode predict
```

---

## 4. 常见问题

### Q1: predict 模式需要哪些文件？

**生境分割**:
- 已训练的 pipeline 文件（`.pkl`）
- 新数据配置文件（与训练时格式相同）

**机器学习**:
- 已训练的模型文件（`.pkl` 或 `.joblib`）
- 新数据 CSV 文件（包含所有训练时使用的特征）

### Q2: 如何找到已训练的 pipeline/model 文件？

训练完成后，文件通常保存在：
- **生境分割**: `{out_dir}/train/habitat_pipeline.pkl`
- **机器学习**: `{output_dir}/models/{model_name}_final_pipeline.pkl`

### Q3: predict 模式的数据格式必须与训练时完全一致吗？

**是的**，非常重要：
- 特征列名必须完全一致
- 特征数量必须相同
- 数据格式（CSV）必须相同

### Q4: 可以在 predict 模式中修改特征提取参数吗？

**不建议**。predict 模式应该使用与训练时完全相同的参数。修改参数可能导致：
- 特征维度不匹配
- 特征分布不一致
- 预测结果不可靠

### Q5: predict 模式会重新训练模型吗？

**不会**。predict 模式只加载已训练的模型/pipeline，不会进行任何训练。

---

## 5. 配置文件列表

### 生境分割 Predict 配置文件

| 策略 | 配置文件 | 用途 |
|------|----------|------|
| Two-Step | `config_habitat.yaml` | 两步聚类策略的 predict 模式 |
| One-Step | `config_habitat_one_step_predict.yaml` | 一步聚类策略的 predict 模式 |
| Direct Pooling | `config_habitat_direct_pooling_predict.yaml` | 直接拼接策略的 predict 模式 |

### 机器学习 Predict 配置文件

| 配置文件 | 用途 |
|----------|------|
| `config_predict.yaml` | 机器学习模型的 predict 模式 |

---

## 6. 测试

所有 predict 模式的配置文件都包含在测试套件中：

```bash
# 测试生境分割 predict 模式
pytest tests/test_habitat.py::TestGetHabitatCommand::test_get_habitat_predict_mode_two_step
pytest tests/test_habitat.py::TestGetHabitatCommand::test_get_habitat_predict_mode_one_step
pytest tests/test_habitat.py::TestGetHabitatCommand::test_get_habitat_predict_mode_direct_pooling

# 测试机器学习 predict 模式
pytest tests/test_predict.py::TestPredictCommand::test_predict_with_config
```

---

## 7. 相关文档

- [环境设置指南](ENVIRONMENT_SETUP.md)
- [故障排除指南](TROUBLESHOOTING.md)
- [Conda 安装指南](CONDA_INSTALLATION.md)
