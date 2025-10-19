# 如何创建详细注释的配置文件 / How to Create Detailed Config Files

## 📝 已完成的配置文件 / Completed Config Files

✅ **Habitat Analysis** - 已完成 / Completed
- `config_getting_habitat_CN.yaml` - 详细中文注释
- `config_getting_habitat_EN.yaml` - 详细英文注释

## 📋 待完成的配置文件 / Pending Config Files

以下配置文件需要创建详细注释版本：

### 1. Machine Learning - 机器学习
- [ ] `config_machine_learning_CN.yaml`
- [ ] `config_machine_learning_EN.yaml`

### 2. K-Fold Cross-Validation - K折交叉验证
- [ ] `config_machine_learning_kfold_CN.yaml`
- [ ] `config_machine_learning_kfold_EN.yaml`

### 3. Image Preprocessing - 图像预处理
- [ ] `config_image_preprocessing_CN.yaml`
- [ ] `config_image_preprocessing_EN.yaml`

### 4. Feature Extraction - 特征提取
- [ ] `config_extract_features_CN.yaml`
- [ ] `config_extract_features_EN.yaml`

### 5. Model Comparison - 模型比较
- [ ] `config_model_comparison_CN.yaml`
- [ ] `config_model_comparison_EN.yaml`

### 6. ICC Analysis - ICC分析
- [ ] `config_icc_analysis_CN.yaml`
- [ ] `config_icc_analysis_EN.yaml`

## 🔨 创建步骤 / Creation Steps

### 步骤1: 复制模板
使用 `config_getting_habitat_CN.yaml` 作为格式模板

### 步骤2: 添加详细注释
对每个配置项添加：
- 中文/英文说明
- 可选值列表
- 使用示例
- 注意事项

### 步骤3: 添加文档头部
包含：
- 文件用途说明
- YAML格式要求
- 快速开始命令
- 相关文档链接

### 步骤4: 添加使用示例
在文件末尾添加常见使用场景的示例配置

### 步骤5: 更新文档链接
在对应的 `doc/app_xxx.md` 中添加配置文件链接

## 📐 格式规范 / Format Standards

### 必需元素：
1. ✅ 文件头部说明（包含框线装饰）
2. ✅ 分节标题（使用分隔线）
3. ✅ 每个参数的详细注释
4. ✅ 可选值说明
5. ✅ 示例配置
6. ✅ 获取帮助部分

### 注释风格：
- 使用清晰的中英文对照
- 添加emoji增强可读性 📖 🔧 ⚠️ 💡
- 使用框线和分隔线组织内容
- 保持2空格缩进

## 🎯 示例参考 / Example Reference

查看 `config_getting_habitat_CN.yaml` 了解完整格式。

关键部分：
- 文件头部（第1-15行）
- 分节标题（使用 ─ 线条）
- 参数说明（包含可选值和示例）
- 使用示例（文件末尾）
- 帮助信息（最后一节）

---

*创建时间 / Created: 2025-10-19*

