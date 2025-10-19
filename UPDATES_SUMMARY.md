# 更新总结 / Updates Summary

## ✨ 主要更新 / Key Updates

### 1. 📚 文档链接集成 / Documentation Links Integration

**所有功能都添加了文档链接**：
- ✅ 主要功能表格：每个功能旁边有 📖 文档链接
- ✅ 核心命令示例：每个命令后都有详细文档路径
- ✅ 快速参考表：新增"文档"列，一键跳转
- 点击 📖 图标即可直接跳转到对应的详细文档

### 2. 🌏 多语言支持 / Multilingual Support

**README顶部添加语言切换**：
- 🇨🇳 简体中文 | 🇬🇧 English
- 一键切换，方便用户查看不同语言版本

### 3. 🧬 生境分析支持两种模式 / Habitat Analysis Two Modes

**一步法 (One-Step)**:
- 个体水平聚类
- 自动确定最佳聚类数
- 适合个性化异质性分析

**二步法 (Two-Step)**:
- Supervoxel + Habitat聚类
- 识别跨患者共通生境
- 适合队列研究

**配置方式**:
```yaml
HabitatsSegmention:
  clustering_mode: one_step  # 或 two_step
```

### 4. 🎯 CLI统一命令行界面 / Unified CLI

所有功能通过 `habit` 命令访问：
```bash
habit habitat --config config/config_getting_habitat.yaml
habit ml --config config/config_machine_learning.yaml
habit kfold --config config/config_machine_learning_kfold.yaml
```

---

## 📚 当前文档结构 / Current Documentation Structure

### 根目录主要文档 / Root Level Main Docs

```
✅ README.md                # 中文主页
✅ README_en.md             # 英文主页
✅ INSTALL.md               # 中文安装指南
✅ INSTALL_en.md            # 英文安装指南
✅ QUICKSTART.md            # 快速入门
✅ HABIT_CLI.md             # CLI使用指南（双语）
```

### 详细功能文档 / Detailed Feature Docs

**中文文档 (doc/)**:
- app_habitat_analysis.md       - Habitat分析（一步法/二步法）
- app_extracting_habitat_features.md  - 特征提取
- app_of_machine_learning.md    - 机器学习
- app_kfold_cross_validation.md - K折交叉验证
- app_model_comparison_plots.md - 模型比较
- app_icc_analysis.md           - ICC分析
- app_image_preprocessing.md    - 图像预处理
- app_habitat_test_retest.md    - 测试-重测
- app_dcm2nii.md               - DICOM转换
- import_robustness_guide.md    - 导入鲁棒性

**英文文档 (doc_en/)**:
- 与中文文档一一对应

---

## 🗑️ 已删除的冗余文档 / Deleted Redundant Docs

- ❌ LANGUAGE_GUIDE.md
- ❌ HOW_TO_SWITCH_LANGUAGE.md
- ❌ MULTILINGUAL_IMPLEMENTATION_SUMMARY.md
- ❌ ONE_STEP_TWO_STEP_IMPLEMENTATION_SUMMARY.md
- ❌ README_CLI_UPDATE_SUMMARY.md
- ❌ DOCS_STRUCTURE.md
- ❌ doc/CLI_USAGE.md
- ❌ doc_en/CLI_USAGE.md
- ❌ doc/DOCUMENTATION_UPDATES_SUMMARY.md
- ❌ doc_en/DOCUMENTATION_UPDATES_SUMMARY.md
- ❌ doc/FONT_CONFIGURATION.md
- ❌ doc_en/FONT_CONFIGURATION.md
- ❌ doc/app_machine_learning_models.md
- ❌ doc_en/app_machine_learning_models.md

**原则**: 精简文档，只保留用户真正需要的核心文档

---

## 🚀 快速开始 / Quick Start

### 1. 查看文档 / View Documentation

**中文用户**:
- 访问 README.md
- 详细文档在 `doc/` 目录

**English Users**:
- Click "🇬🇧 English" at top
- Detailed docs in `doc_en/` directory

### 2. 使用Habitat分析 / Use Habitat Analysis

```bash
# 一步法（个性化）
# 修改配置: clustering_mode: one_step
habit habitat --config config/config_getting_habitat.yaml

# 二步法（队列研究）
# 修改配置: clustering_mode: two_step
habit habitat --config config/config_getting_habitat.yaml
```

### 3. 机器学习 / Machine Learning

```bash
# 训练模型
habit ml --config config/config_machine_learning.yaml --mode train

# K折交叉验证
habit kfold --config config/config_machine_learning_kfold.yaml

# 模型比较
habit compare --config config/config_model_comparison.yaml
```

---

## 📖 文档原则 / Documentation Principles

1. ✅ **简洁明了**: 只保留核心文档
2. ✅ **双语支持**: 中英文完整覆盖
3. ✅ **易于查找**: 清晰的目录结构
4. ✅ **用户友好**: 从用户角度组织文档
5. ✅ **避免冗余**: 一个主题一份文档

---

*最后更新 / Last Updated: 2025-10-19*

