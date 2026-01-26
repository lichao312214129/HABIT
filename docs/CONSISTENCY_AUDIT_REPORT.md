# HABIT 文档一致性审查报告

**审查日期**: 2026-01-26

**审查范围**: docs 目录 vs habit 包 vs demodata 配置文件

---

## 📋 发现的不一致问题汇总

### 🔴 严重问题（已修复）

#### 1. 字段命名不一致：`out_dir` vs `output`

**问题描述**：
- **docs 中**: 使用 `out_dir` 字段
- **MLConfig (config_schemas.py L152)**: 使用 `output` 字段
- **ModelComparisonConfig (config_schemas.py L134)**: 使用 `output_dir` 字段
- **demodata 配置**: 使用 `output` 字段

**影响**：用户按照 docs 配置会导致配置解析失败

**修复状态**：✅ 已修复
- 更新 [machine_learning_modeling_zh.rst](file:///f:\work\habit_project\docs\source\user_guide\machine_learning_modeling_zh.rst)
- 将所有 `out_dir` 替换为 `output`

**建议**：
- `MLConfig`: 使用 `output`
- `ModelComparisonConfig`: 使用 `output_dir`
- 保持术语一致性

---

#### 2. 模型对比配置结构不一致

**问题描述**：
- **docs 中描述**:
  ```yaml
  out_dir: ./results/comparison
  models:
    - name: Clinical_Model
      file: ./results/clinical/predictions.csv
      label_col: Label
      prob_col: Probability
  ```

- **实际 config_model_comparison.yaml**:
  ```yaml
  output_dir: ./ml_data/model_comparison
  files_config:
    - path: ./ml_data/radiomics/all_prediction_results.csv
      model_name: radiomics
      subject_id_col: subject_id
      label_col: label
      prob_col: LogisticRegression_prob
      pred_col: LogisticRegression_pred
      split_col: dataset
  ```

- **ModelComparisonConfig (config_schemas.py L131-140)**:
  ```python
  class ModelComparisonConfig(BaseConfig):
      output_dir: str
      files_config: List[ComparisonFileConfig] = Field(default_factory=list)
  ```

**影响**：字段完全不匹配，用户无法正确使用

**修复状态**：✅ 已修复
- 更新 [model_comparison_zh.rst](file:///f:\work\habit_project\docs\source\user_guide\model_comparison_zh.rst)
- 添加完整的配置示例，包括 `files_config`、`merged_data`、`visualization` 等

---

#### 3. ICC 配置结构不一致

**问题描述**：
- **docs 中描述**:
  ```yaml
  input:
    - path: ./data/test_scan.csv
      name: test_
      subject_id_col: PatientID
  output:
    path: ./results/icc_analysis.json
  ```

- **实际 config_icc.yaml**:
  ```yaml
  input:
    type: "files"
    file_groups:
      - [./ml_data/dataset1.csv, ./ml_data/dataset2.csv]
  output:
    path: ./ml_data/icc_radiomics.json
  metrics:
    - icc2
    - icc3
    - cohen
  ```

**影响**：ICC 工具无法正常工作

**修复状态**：✅ 已修复
- 更新 [app_icc_analysis_zh.rst](file:///f:\work\habit_project\docs\source\app_icc_analysis_zh.rst)
- 添加 `type`、`file_groups`、`metrics` 等完整配置

---

#### 4. Test-Retest 使用方式不一致

**问题描述**：
- **docs 中描述**: 使用配置文件的 `input` 列表方式
- **实际实现**: 使用命令行参数方式
- **CLI 实现**: 调用 `scripts/app_habitat_test_retest_mapper.py`

**影响**：用户无法按照文档使用工具

**修复状态**：✅ 已修复
- 更新 [app_habitat_test_retest_zh.rst](file:///f:\work\habit_project\docs\source\app_habitat_test_retest_zh.rst)
- 明确说明使用命令行参数
- 添加配置文件的使用方式（作为参数传递）

---

## 📊 一致性检查清单

### ✅ 已确认一致的模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 机器学习配置 - models 字段 | ✅ 一致 | 已从 ModelTraining 迁移到 models |
| 数据输入格式 | ✅ 一致 | CSV 和 Excel 格式 |
| 多分类支持 | ✅ 一致 | 文档已说明 |
| 输出目录结构 | ✅ 一致 | 已添加详细说明 |
| 模型类型列表 | ✅ 一致 | LogisticRegression, RandomForest, XGBoost, SVM, KNN, AutoGluon |

### ⚠️ 仍需关注的潜在问题

| 问题 | 模块 | 严重程度 | 建议 |
|------|------|----------|------|
| 字段命名不统一 | 多个模块 | 中 | 制定命名规范 |
| 脚本 vs CLI 混用 | 工具命令 | 中 | 统一 CLI 接口 |
| 配置验证缺失 | 所有配置 | 低 | 添加配置验证 |

---

## 🔧 建议的改进措施

### 1. 统一字段命名规范

```
输出目录:
  - MLConfig: output
  - ModelComparisonConfig: output_dir
  - PredictionConfig: output_dir
  - 建议统一为: output_dir
```

### 2. 添加配置验证

建议在 config_schemas 中添加更严格的验证，确保：
- 必填字段检查
- 字段类型检查
- 依赖字段检查

### 3. 文档与代码同步机制

- 在代码中直接生成配置示例
- 使用 doctest 或示例测试验证文档
- 添加 CI 检查确保配置一致性

### 4. 工具命令统一

- 所有工具使用统一的 CLI 接口模式
- 支持 `--config` 参数
- 支持命令行参数覆盖配置

---

## 📝 修复文件清单

**已修改文件 (6个)**:
1. [machine_learning_modeling_zh.rst](file:///f:\work\habit_project\docs\source\user_guide\machine_learning_modeling_zh.rst)
2. [model_comparison_zh.rst](file:///f:\work\habit_project\docs\source\user_guide\model_comparison_zh.rst) (完全重写)
3. [app_icc_analysis_zh.rst](file:///f:\work\habit_project\docs\source\app_icc_analysis_zh.rst) (完全重写)
4. [app_habitat_test_retest_zh.rst](file:///f:\work\habit_project\docs\source\app_habitat_test_retest_zh.rst) (完全重写)
5. [cli_zh.rst](file:///f:\work\habit_project\docs\source\cli_zh.rst)
6. [customization/index_zh.rst](file:///f:\work\habit_project\docs\source\customization\index_zh.rst)

**新建文件 (4个)**:
1. [app_merge_csv_zh.rst](file:///f:\work\habit_project\docs\source\app_merge_csv_zh.rst)
2. [model_comparison_zh.rst](file:///f:\work\habit_project\docs\source\user_guide\model_comparison_zh.rst)

---

## 📌 后续建议

1. **定期审查**: 建议每季度进行一次文档一致性审查
2. **配置模板**: 为每个模块提供标准配置文件模板
3. **集成测试**: 添加配置解析的集成测试
4. **示例数据集**: 完善 demodata 中的示例配置
5. **变更日志**: 记录配置变更历史

---

**审查人**: AI Assistant

**下次审查**: 建议 2026-04-26
