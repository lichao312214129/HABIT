# ICC 分析模块使用文档

## 1. 功能概述

ICC (Intraclass Correlation Coefficient, 组内相关系数) 分析模块是一个用于评估定量测量可靠性的专用工具。在影像组学等研究中，它通常用于：

- **测试-重测信度 (Test-Retest Reliability)**: 评估同一样本在不同时间点扫描或处理后，提取的特征是否一致。
- **观察者间一致性 (Inter-Observer Agreement)**: 评估不同观察者（或不同软件、不同参数）对同一样本进行勾画或分析后，提取的特征是否一致。

本工具通过计算特征的ICC值，帮助用户筛选出在不同条件下稳定、可靠的特征，为后续的模型构建提供高质量的数据基础。

## 2. 快速开始

### 使用CLI（推荐）✨

```bash
# 使用配置文件运行ICC分析
habit icc --config config/config_icc_analysis.yaml
```

### 使用传统脚本

```bash
python scripts/app_icc_analysis.py --config config/config_icc_analysis.yaml
```

## 3. 输入数据准备

正确准备输入数据是成功运行分析的关键。

- **文件格式**: 支持`.csv`和`.xlsx`格式。
- **数据结构**:
    - **第一列必须是受试者ID**，并作为索引列。
    - 后续的每一列代表一个独立的特征。
    - 每一行代表一个受试者。
- **数据对齐**:
    - 工具会自动找出同一组文件中**共有的受试者ID**和**共有的特征列**。
    - ICC分析将**只在这些共有的受试者和特征上进行**。
    - 因此，请确保需要比较的文件中，受试者ID的命名方式是一致的。

**示例 `test.csv`**:
```csv
SubjectID,feature_A,feature_B,feature_C
Patient_01,10.5,0.98,150
Patient_02,11.2,0.95,165
Patient_03,9.8,0.99,140
```

**示例 `retest.csv`**:
```csv
SubjectID,feature_A,feature_B,feature_D
Patient_01,10.8,0.97,5.5
Patient_02,11.1,0.96,5.8
Patient_04,12.0,0.92,6.1
```
> 在这个例子中，工具只会对 `Patient_01` 和 `Patient_02` 的 `feature_A` 和 `feature_B` 计算ICC值。

## 4. 配置文件 (`config.yaml`) 详解

工具的行为由一个YAML配置文件驱动。

### 完整配置示例
```yaml
# 输入配置
input:
  # 输入模式: "files" 或 "directories"
  type: "files"

  # 文件组: 当 type 为 "files" 时使用
  # 每个子列表是一个独立的比较组，通常包含2个或以上的文件
  file_groups:
    - [/path/to/test_features.csv, /path/to/retest_features.csv]
    - [/path/to/observer1_features.csv, /path/to/observer2_features.csv]

  # 目录列表: 当 type 为 "directories" 时使用
  # 工具会自动匹配这些目录下同名的文件，并将其归为一组进行比较
  # dir_list:
  #   - /path/to/test_data_dir
  #   - /path/to/retest_data_dir

# 输出配置
output:
  # 输出JSON文件的完整路径
  path: ./results/icc_analysis/icc_results.json

# 并行进程数 (null 表示使用所有可用的CPU核心)
processes: 6

# 调试模式 (True/False)，启用后会输出更详细的日志
debug: false
```

### 参数说明

- **`input.type`**: 定义输入模式。
  - `files`: 直接指定要比较的文件组。更灵活，推荐使用。
  - `directories`: 指定多个目录，工具会自动寻找这些目录下的同名文件进行配对比较。适用于文件结构非常规整的情况。

- **`input.file_groups`**: 当`type`为`files`时使用。这是一个列表，每个元素是另一个列表，代表一个独立的比较组。例如，你可以同时进行测试-重测分析和观察者间分析。

- **`input.dir_list`**: 当`type`为`directories`时使用。提供一个目录路径的列表。

- **`output.path`**: 定义结果输出路径。分析结果会以JSON格式保存在这里。

- **`processes`**: 设置用于并行计算的CPU核心数。设置为`null`或不设置此项，将默认使用所有可用的核心。

- **`debug`**: 设置为`true`会启用调试模式，日志将记录更多细节，便于排查问题。

## 5. 输出结果解读

### JSON 输出文件

分析结果是一个JSON文件，其结构如下：

```json
{
    "test_features_vs_retest_features": {
        "feature_A": 0.92,
        "feature_B": 0.85,
        ...
    },
    "observer1_features_vs_observer2_features": {
        "feature_A": 0.78,
        "feature_B": 0.88,
        ...
    }
}
```
- JSON的主键是根据比较组的文件名自动生成的组名。
- 每个组名下是一个字典，包含了该组内所有公共特征的ICC值。

### ICC 模型说明

- 本工具固定使用 **ICC3** 模型进行计算，它对应于**双向随机效应模型 (Two-Way Random) 和绝对一致性 (Absolute Agreement)**。这在评估不同时间点或不同观察者之间的可靠性时是一种常用且严格的标准。

### ICC 值解读标准

ICC值介于0和1之间，通常可按以下标准解读其可靠性：
- **< 0.50**: 差 (Poor)
- **0.50 - 0.75**: 中等 (Moderate)
- **0.75 - 0.90**: 良 (Good)
- **> 0.90**: 优 (Excellent)

在特征筛选时，通常选择ICC值大于0.75的特征用于后续建模。

### 控制台总结

脚本运行结束后，会在控制台打印一个总结信息，告知每个组别的平均ICC、以及达到“良好”标准（ICC >= 0.75）的特征数量和比例，帮助您快速评估整体的一致性。