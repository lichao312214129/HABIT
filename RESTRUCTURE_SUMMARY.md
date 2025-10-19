# 配置文件重组总结 / Configuration Restructure Summary

## ✅ 完成的工作 / Completed Work

成功将配置文件体系重组为清晰的双层结构。

## 📁 新的目录结构 / New Directory Structure

```
habit_project/
├── config/                          # 精简配置文件（日常使用）
│   ├── config_getting_habitat.yaml
│   ├── config_extract_features.yaml
│   ├── config_machine_learning.yaml
│   ├── config_machine_learning_kfold.yaml
│   ├── config_model_comparison.yaml
│   ├── config_icc_analysis.yaml
│   ├── config_image_preprocessing.yaml
│   ├── config_traditional_radiomics.yaml
│   ├── ... (其他配置文件)
│   └── README_CONFIG.md             # 配置文件索引
│
└── config_templates/                # 详细注释模板（学习参考）
    ├── config_getting_habitat_annotated.yaml
    ├── config_extract_features_annotated.yaml
    ├── config_machine_learning_annotated.yaml
    ├── config_machine_learning_kfold_annotated.yaml
    ├── config_model_comparison_annotated.yaml
    ├── config_icc_analysis_annotated.yaml
    ├── config_image_preprocessing_annotated.yaml
    ├── config_traditional_radiomics_annotated.yaml
    └── README.md                    # 模板使用说明
```

## 🎯 设计理念 / Design Philosophy

### config/ 目录
- **用途**: 日常使用的精简配置
- **特点**: 
  - 简洁明了，便于快速修改
  - CLI命令默认路径
  - 直接可用的示例配置

### config_templates/ 目录
- **用途**: 详细注释的学习模板
- **特点**:
  - 完整的英文注释（每个参数都有说明）
  - 包含使用示例和最佳实践
  - 参数说明、数据类型、可选值
  - 适合学习和深入理解

## 📊 模板文件统计 / Template Files Statistics

| 模板文件 | 大小 | 行数估计 |
|---------|------|---------|
| `config_getting_habitat_annotated.yaml` | 17.4 KB | ~350 |
| `config_extract_features_annotated.yaml` | 18.4 KB | ~380 |
| `config_machine_learning_annotated.yaml` | 34.9 KB | ~700 |
| `config_machine_learning_kfold_annotated.yaml` | 15.5 KB | ~320 |
| `config_model_comparison_annotated.yaml` | 20.5 KB | ~410 |
| `config_icc_analysis_annotated.yaml` | 15.1 KB | ~310 |
| `config_image_preprocessing_annotated.yaml` | 17.6 KB | ~360 |
| `config_traditional_radiomics_annotated.yaml` | 14.9 KB | ~300 |
| **总计** | **~154 KB** | **~3130 行** |

## 🔄 更新的文档 / Updated Documentation

### 1. 配置文件索引
- ✅ `config/README_CONFIG.md` - 更新了所有链接指向 `config_templates/`

### 2. 功能文档（中文）
- ✅ `doc/app_habitat_analysis.md`
- ✅ `doc/app_of_machine_learning.md`
- ✅ `doc/app_image_preprocessing.md`
- ✅ `doc/app_extracting_habitat_features.md`
- ✅ `doc/app_kfold_cross_validation.md`
- ✅ `doc/app_model_comparison_plots.md`
- ✅ `doc/app_icc_analysis.md`

### 3. 功能文档（英文）
- ✅ `doc_en/app_habitat_analysis.md`
- ✅ `doc_en/app_of_machine_learning.md`
- ✅ `doc_en/app_image_preprocessing.md`
- ✅ `doc_en/app_extracting_habitat_features.md`
- ✅ `doc_en/app_kfold_cross_validation.md`
- ✅ `doc_en/app_model_comparison_plots.md`
- ✅ `doc_en/app_icc_analysis.md`

## 💡 用户使用指南 / User Guide

### 快速开始 / Quick Start
```bash
# 使用默认配置（位于 config/ 目录）
habit habitat --config config/config_getting_habitat.yaml
```

### 学习和自定义 / Learning & Customization
```bash
# 1. 查看详细模板
cat config_templates/config_getting_habitat_annotated.yaml

# 2. 复制模板到 config/ 目录
cp config_templates/config_getting_habitat_annotated.yaml config/my_config.yaml

# 3. 删除不需要的注释，保留需要的参数

# 4. 使用自定义配置
habit habitat --config config/my_config.yaml
```

### 查看帮助 / Get Help
```bash
# 查看配置文件列表
cat config/README_CONFIG.md

# 查看模板使用说明
cat config_templates/README.md
```

## ✨ 优势 / Advantages

1. **清晰分离**: 日常配置和学习材料分开
2. **保持简洁**: config/ 目录保持干净整洁
3. **易于学习**: 完整的注释模板便于理解
4. **向后兼容**: 所有现有脚本和CLI命令无需修改
5. **文档完整**: 所有链接已更新，指向正确位置

## 📝 命名规范 / Naming Convention

- **精简配置**: `config_xxx.yaml`
- **详细模板**: `config_xxx_annotated.yaml`
- **目录**: 
  - `config/` - 当前使用的配置
  - `config_templates/` - 详细注释模板

## 🔮 未来可扩展 / Future Extensibility

如需添加新模块的配置模板：

1. 在 `config/` 创建精简配置 `config_new_module.yaml`
2. 在 `config_templates/` 创建详细模板 `config_new_module_annotated.yaml`
3. 更新 `config/README_CONFIG.md` 添加新条目
4. 在相关文档中添加配置链接

---

**完成日期**: 2025-10-19  
**重组文件数**: 8个模板文件  
**更新文档数**: 15个文档文件  
**总注释行数**: ~3130行

