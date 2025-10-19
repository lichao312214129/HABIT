# HABIT 命令行界面使用指南

> **HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** 统一命令行界面文档

---

## 📖 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [使用方法](#使用方法)
- [所有命令](#所有命令)
- [常见问题](#常见问题)
- [完整示例](#完整示例)

---

## ⚡ 快速开始

### 三步上手

```bash
# 1. 安装
pip install -e .

# 2. 测试
python -m habit --help

# 3. 使用
python -m habit preprocess -c config/config_image_preprocessing.yaml
```

---

## 🔧 安装

### 前提条件

- Python >= 3.8
- 已安装项目依赖

### 安装步骤

```bash
# 步骤 1: 清理缓存（如果之前安装过）
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# 步骤 2: 安装依赖
pip install -r requirements.txt

# 步骤 3: 安装 HABIT 包
pip install -e .
```

### 验证安装

```bash
# 方法 1: 使用 habit 命令
habit --help

# 方法 2: 使用 Python 模块（推荐，更可靠）
python -m habit --help
```

**✅ 看到命令列表说明安装成功！**

---

## 🚀 使用方法

### 三种运行方式

#### 方式 1: `habit` 命令（需要正确安装）

```bash
habit --help
habit preprocess -c config.yaml
```

#### 方式 2: Python 模块（推荐⭐）

```bash
python -m habit --help
python -m habit preprocess -c config.yaml
```

#### 方式 3: 直接运行脚本（开发调试）

```bash
python habit/cli.py --help
python habit/cli.py preprocess -c config.yaml
```

### 基本语法

```bash
habit <命令> [选项]
```

**常用选项：**
- `--config, -c`: 指定配置文件（必需）
- `--help`: 显示帮助信息
- `--version`: 显示版本号

---

## 📋 所有命令

### 命令列表

| 命令 | 说明 | 对应原脚本 |
|------|------|-----------|
| `preprocess` | 图像预处理 | `app_image_preprocessing.py` |
| `habitat` | 生成 Habitat 地图 | `app_getting_habitat_map.py` |
| `extract-features` | 提取 Habitat 特征 | `app_extracting_habitat_features.py` |
| `ml` | 机器学习（训练/预测） | `app_of_machine_learning.py` |
| `kfold` | K折交叉验证 | `app_kfold_cv.py` |
| `compare` | 模型比较 | `app_model_comparison_plots.py` |
| `icc` | ICC 分析 | `app_icc_analysis.py` |
| `radiomics` | 传统影像组学 | `app_traditional_radiomics_extractor.py` |
| `test-retest` | Test-Retest 分析 | `app_habitat_test_retest_mapper.py` |

### 1. 图像预处理

```bash
python -m habit preprocess -c config/config_image_preprocessing.yaml
```

**功能**: 对医学图像进行重采样、配准、标准化处理

### 2. Habitat 分析

```bash
python -m habit habitat -c config/config_getting_habitat.yaml

# 启用调试模式
python -m habit habitat -c config/config_getting_habitat.yaml --debug
```

**功能**: 通过聚类分析生成 Habitat 地图

### 3. 提取特征

```bash
python -m habit extract-features -c config/config_extract_features.yaml
```

**功能**: 从聚类后的图像提取 Habitat 特征

### 4. 机器学习

#### 训练模型

```bash
python -m habit ml -c config/config_machine_learning.yaml -m train
```

#### 预测（使用训练好的模型）

```bash
python -m habit ml \
  -c config/config_machine_learning.yaml \
  -m predict \
  --model ./ml_data/ml/rad/model_package.pkl \
  --data ./ml_data/breast_cancer_dataset.csv \
  -o ./ml_data/predictions/
```

**参数说明：**
- `-m, --mode`: 模式（`train` 或 `predict`）
- `--model`: 模型文件路径（.pkl）
- `--data`: 数据文件路径（.csv）
- `-o, --output`: 输出目录
- `--model-name`: 指定使用的模型名称
- `--evaluate/--no-evaluate`: 是否评估性能（默认评估）

### 5. K折交叉验证

```bash
python -m habit kfold -c config/config_machine_learning_kfold.yaml
```

**功能**: 对模型进行 K 折交叉验证

**输出文件**:
- `kfold_cv_results.json` - 详细的交叉验证结果
- `kfold_performance_summary.csv` - 性能摘要表
- `all_prediction_results.csv` - **兼容格式的预测结果**（可用于模型比较）
- `kfold_roc_curves.pdf` - ROC曲线（如果启用可视化）
- `kfold_calibration_curves.pdf` - 校准曲线（如果启用可视化）
- `kfold_dca_curves.pdf` - DCA决策曲线（如果启用可视化）
- `kfold_confusion_matrix_*.pdf` - 混淆矩阵（如果启用可视化）

**可视化配置**:
```yaml
is_visualize: true  # 在配置文件中启用可视化
```

**提示**: 
- K折验证完成后，会自动生成 `all_prediction_results.csv` 文件，该文件与标准 ml 命令的输出格式完全兼容
- 启用 `is_visualize` 后，会自动生成 ROC、DCA、校准曲线等可视化图表
- 可视化基于所有 fold 的聚合预测生成，全面反映模型性能

### 6. 模型比较

```bash
python -m habit compare -c config/config_model_comparison.yaml
```

**功能**: 生成多个模型的比较图表和统计数据

**兼容性**: 
- ✅ 支持标准 ml 命令的输出（`all_prediction_results.csv`）
- ✅ 支持 kfold 交叉验证的输出（`all_prediction_results.csv`）
- 可以比较来自不同训练方式的模型结果

### 7. ICC 分析

```bash
python -m habit icc -c config/config_icc_analysis.yaml
```

**功能**: 执行组内相关系数（ICC）分析

### 8. 传统影像组学

```bash
python -m habit radiomics -c config/config_traditional_radiomics.yaml
```

**功能**: 提取传统影像组学特征

### 9. Test-Retest 分析

```bash
python -m habit test-retest -c config/config_habitat_test_retest.yaml
```

**功能**: 执行 test-retest 重复性分析

---

## ❓ 常见问题

### Q1: `habit` 命令找不到

**原因**: 环境变量未配置或未正确安装

**解决方案**: 使用 Python 模块方式（更可靠）

```bash
python -m habit --help
```

### Q2: ImportError 或模块找不到

**原因**: Python 缓存问题或安装不完整

**解决方案**:

```bash
# 1. 清理缓存
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# 2. 重新安装
pip uninstall HABIT -y
pip install -e .
```

### Q3: 配置文件路径错误

**解决方案**: 使用绝对路径或确保在正确的目录下

```bash
# 相对路径（推荐在项目根目录运行）
python -m habit preprocess -c ./config/config_image_preprocessing.yaml

# 绝对路径
python -m habit preprocess -c F:/work/research/.../config.yaml
```

### Q4: 原有脚本还能用吗？

**答**: 完全可以！新 CLI 不影响原有脚本。

```bash
# 旧方式仍然可用
python scripts/app_image_preprocessing.py --config config.yaml

# 新方式（更简洁）
python -m habit preprocess -c config.yaml
```

---

## 💡 完整示例

### 完整的 Habitat 分析工作流

```bash
# 步骤 1: 图像预处理
python -m habit preprocess -c config/config_image_preprocessing.yaml

# 步骤 2: 生成 Habitat 地图
python -m habit habitat -c config/config_getting_habitat.yaml

# 步骤 3: 提取 Habitat 特征
python -m habit extract-features -c config/config_extract_features.yaml

# 步骤 4: 训练机器学习模型（两种方式任选其一）

## 方式 A: 标准训练/测试集分割
python -m habit ml -c config/config_machine_learning.yaml -m train

## 方式 B: K折交叉验证（推荐用于小样本）
python -m habit kfold -c config/config_machine_learning_kfold.yaml

# 步骤 5: 模型比较（支持两种方式的结果）
python -m habit compare -c config/config_model_comparison.yaml

# 提示：compare 命令会自动读取 all_prediction_results.csv 文件
# 无论是来自 ml 命令还是 kfold 命令，格式完全兼容
```

### 使用训练好的模型预测新数据

```bash
python -m habit ml \
  -c config/config_machine_learning.yaml \
  -m predict \
  --model ./ml_data/ml/rad/model_package.pkl \
  --data ./ml_data/new_patient_data.csv \
  --output ./ml_data/predictions/ \
  --model-name XGBoost \
  --evaluate
```

---

## 📚 相关文档

### 中文文档
- **详细使用手册**: `doc/CLI_USAGE.md`
- **原功能文档**: `doc/app_*.md`

### 英文文档
- **Usage Manual**: `doc_en/CLI_USAGE.md`
- **Feature Docs**: `doc_en/app_*.md`

### 其他文档
- **主 README**: `README.md`
- **安装指南**: `INSTALL.md`
- **快速入门**: `QUICKSTART.md`

---

## 🎓 新旧方式对比

### 旧方式（脚本）

```bash
python scripts/app_image_preprocessing.py --config config.yaml
python scripts/app_getting_habitat_map.py --config config.yaml --debug
python scripts/app_of_machine_learning.py --config config.yaml --mode train
python scripts/app_kfold_cv.py --config config.yaml
```

### 新方式（CLI）

```bash
python -m habit preprocess -c config.yaml
python -m habit habitat -c config.yaml --debug
python -m habit ml -c config.yaml -m train
python -m habit kfold -c config.yaml
```

**优势**:
- ✅ 更简洁、更直观
- ✅ 统一的命令风格
- ✅ 自动生成帮助文档
- ✅ 参数验证和错误提示
- ✅ 支持短选项（`-c` 代替 `--config`）

---

## 💡 使用技巧

### 1. 使用短选项

```bash
# --config 可以简写为 -c
python -m habit preprocess -c config.yaml

# --mode 可以简写为 -m
python -m habit ml -c config.yaml -m train

# --output 可以简写为 -o
python -m habit ml -c config.yaml -m predict --model m.pkl --data d.csv -o ./out/
```

### 2. 查看命令帮助

```bash
# 每个命令都有详细帮助
python -m habit --help              # 所有命令列表
python -m habit preprocess --help   # 预处理命令帮助
python -m habit ml --help           # 机器学习命令帮助
```

### 3. 推荐的命令别名

如果你经常使用，可以在 PowerShell 配置文件中添加别名：

```powershell
# 编辑 PowerShell 配置文件
notepad $PROFILE

# 添加以下内容
function habit { python -m habit $args }

# 之后就可以直接使用
habit --help
habit preprocess -c config.yaml
```

---

## 📧 技术支持

- **邮箱**: lichao19870617@163.com
- **问题反馈**: 请详细描述问题和错误信息

---

## 📝 版本信息

- **版本**: 0.1.0
- **更新日期**: 2025-10-19
- **状态**: ✅ 稳定可用

---

**祝使用愉快！** 🎉

