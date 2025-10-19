# ICC分析模块使用文档

## 功能概述

ICC分析模块用于计算组内相关系数（Intraclass Correlation Coefficient，ICC）。该模块支持对生境特征的测试-重测一致性、观察者间一致性和各种可靠性评估进行分析。ICC是评估定量测量可靠性的标准统计方法，在放射组学研究中具有重要意义。

## 🚀 快速开始

### 使用CLI（推荐）✨

```bash
# 使用默认配置
habit icc

# 使用指定配置文件
habit icc --config config/config_icc_analysis.yaml

# 简写形式
habit icc -c config/config_icc_analysis.yaml
```

### 使用传统脚本（兼容旧版）

```bash
python scripts/app_icc_analysis.py --config <config_file_path>
```

## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | YAML配置文件路径（必需） |

## 📋 配置文件

**📖 配置文件链接**：
- 📄 [当前配置文件](../config/config_icc_analysis.yaml) - 实际使用的配置文件
- 🇨🇳 详细中文配置（待创建）- 包含完整的中文注释和使用说明
- 🇬🇧 详细英文配置（待创建）- Complete English comments and instructions

> 💡 **提示**: 详细注释版配置文件正在准备中。目前请参考下方的配置说明。

## 配置文件格式

`app_icc_analysis.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据路径
input:
  type: <输入类型>  # "files" 或 "directory"
  path: <输入路径>  # 文件列表或目录
  pattern: <文件匹配模式>  # 当type为directory时使用

# 输出配置
output:
  dir: <输出目录>
  report_name: <报告名称>

# ICC分析配置
icc:
  type: <ICC类型>
  confidence_level: <置信水平>
  outlier_removal: <异常值处理方法>
```

### ICC分析类型配置

```yaml
icc:
  # ICC类型，支持以下类型之一：
  # - "test_retest": 测试-重测一致性
  # - "inter_observer": 观察者间一致性
  # - "intra_observer": 观察者内一致性
  # - "multi_reader": 多读者多病例
  type: "test_retest"
  
  # ICC模型配置
  model: <模型类型>  # "oneway", "twoway"
  unit: <单位>  # "single", "average"
  effect: <效应>  # "random", "fixed", "mixed"
  
  # 一致性/绝对一致配置
  definition: <一致性定义>  # "consistency", "absolute_agreement"
  
  # 置信水平
  confidence_level: 0.95
  
  # 异常值处理
  outlier_removal:
    method: <方法>  # "none", "zscore", "iqr", "modified_zscore"
    threshold: <阈值>  # 方法特定的阈值值
```

### 分组和特征配置

```yaml
# 数据分组配置
grouping:
  method: <分组方法>  # "filename_pattern", "explicit_mapping", "column"
  pattern: <文件名模式>  # 对于filename_pattern方法
  mapping_file: <映射文件>  # 对于explicit_mapping方法
  id_column: <ID列名>  # 对于column方法
  group_column: <分组列名>  # 对于column方法

# 特征配置
features:
  # 要包含的特征列
  include: <包含的特征列表>  # 可以是列表或 "*" 表示全部
  
  # 要排除的特征列
  exclude: <排除的特征列表>
  
  # 特征分类
  categories:
    - name: <类别1>
      features: <类别1特征列表>
    - name: <类别2>
      features: <类别2特征列表>
```

## 支持的ICC类型

ICC分析支持以下类型：

1. **测试-重测一致性 (test_retest)**：评估同一受试者在不同时间点测量的一致性
2. **观察者间一致性 (inter_observer)**：评估不同观察者测量同一对象的一致性
3. **观察者内一致性 (intra_observer)**：评估同一观察者在不同时间点测量同一对象的一致性
4. **多读者多病例 (multi_reader)**：多读者对多个病例进行评估的一致性分析

## ICC模型参数

### 模型类型 (model)

- **oneway**：单向随机效应模型，适用于每个受试者只有一个评分者的情况
- **twoway**：双向模型，适用于同一组评分者评估所有受试者的情况

### 单位 (unit)

- **single**：评估单个评分的可靠性
- **average**：评估平均评分的可靠性

### 效应 (effect)

- **random**：评分者被视为随机样本
- **fixed**：评分者被视为固定因素
- **mixed**：混合效应模型

### 一致性定义 (definition)

- **consistency**：评估评分的相对一致性
- **absolute_agreement**：评估评分的绝对一致性

## 异常值处理方法

- **none**：不进行异常值处理
- **zscore**：基于Z分数识别和处理异常值
- **iqr**：基于四分位距识别和处理异常值
- **modified_zscore**：使用修正Z分数方法

## 执行流程

1. 加载配置文件
2. 读取输入数据
3. 按配置的分组方法对数据进行分组
4. 对选定的特征计算ICC
5. 生成ICC分析报告和可视化结果
6. 保存结果到输出目录

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `icc_results.csv`：所有特征的ICC值及其置信区间
2. `icc_summary.csv`：按特征类别汇总的ICC结果
3. `icc_plots/`：ICC可视化图表，包括：
   - ICC柱状图
   - Bland-Altman图
   - 散点相关图
   - 热图
4. `icc_report.pdf`：完整的ICC分析报告

## 完整配置示例

### 测试-重测ICC分析

```yaml
# 基本配置
input:
  type: "directory"
  path: "./data/test_retest_features"
  pattern: "*.csv"

output:
  dir: "./results/icc_analysis"
  report_name: "test_retest_icc_report"

# ICC分析配置
icc:
  type: "test_retest"
  model: "twoway"
  unit: "single"
  effect: "random"
  definition: "absolute_agreement"
  confidence_level: 0.95
  outlier_removal:
    method: "iqr"
    threshold: 1.5

# 分组配置
grouping:
  method: "filename_pattern"
  pattern: "features_{subject_id}_{timepoint}.csv"

# 特征配置
features:
  include: "*"
  exclude: ["patient_id", "scan_date", "study_id"]
  categories:
    - name: "形状特征"
      features: ["shape_volume", "shape_surface_area", "shape_sphericity"]
    - name: "一阶特征"
      features: ["firstorder_mean", "firstorder_std", "firstorder_entropy"]
    - name: "纹理特征"
      features: ["glcm_*", "glrlm_*", "glszm_*"]
```

### 观察者间ICC分析

```yaml
# 基本配置
input:
  type: "files"
  path: 
    - "./data/observer1/features.csv"
    - "./data/observer2/features.csv"
    - "./data/observer3/features.csv"

output:
  dir: "./results/inter_observer_icc"
  report_name: "inter_observer_icc_report"

# ICC分析配置
icc:
  type: "inter_observer"
  model: "twoway"
  unit: "single"
  effect: "random"
  definition: "absolute_agreement"
  confidence_level: 0.95
  outlier_removal:
    method: "none"

# 分组配置
grouping:
  method: "column"
  id_column: "patient_id"
  group_column: "observer"

# 特征配置
features:
  include: ["intensity_*", "texture_*", "shape_*"]
  exclude: []
```

## 结果解释

ICC值的解释通常遵循以下标准：

- **< 0.50**: 较差的可靠性
- **0.50-0.75**: 中等可靠性
- **0.75-0.90**: 良好可靠性
- **> 0.90**: 优秀可靠性

## 注意事项

1. 确保数据格式正确，特别是测试-重测或观察者数据的配对关系
2. 选择适合研究设计的ICC类型和模型参数
3. 对于特征数量较多的数据集，考虑按特征类别分组进行分析
4. ICC分析结果应结合临床意义和研究目的进行解释
5. 当存在严重异常值时，使用适当的异常值处理方法可提高结果可靠性 