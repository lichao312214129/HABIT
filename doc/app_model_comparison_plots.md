# app_model_comparison_plots.py 功能文档

## 功能概述

`app_model_comparison_plots.py` 是HABIT工具包中用于比较和评估多个机器学习模型性能的专用工具。它能读取多个模型的预测结果，合并评估数据，生成多种性能评估图表和指标，并支持按数据集分组进行分析（如训练集、测试集）。

## 用法

```bash
python scripts/app_model_comparison_plots.py --config <config_file_path>
```

## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | YAML配置文件路径（必需） |

## 配置文件格式

`app_model_comparison_plots.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 输出目录配置
output_dir: "./results/model_comparison"  # 所有比较结果将保存到的目录
```

### 模型预测文件配置

```yaml
# 每个条目定义要包含在比较中的模型预测文件
files_config:
  - path: "path/to/model_a_predictions.csv"  # 包含模型预测的CSV文件路径
    model_name: "ModelA"                     # 模型的显示名称
    subject_id_col: "subjid"                 # 包含受试者标识符的列名
    label_col: "true_label"                  # 包含真实结果标签的列名(0/1)
    prob_col: "prediction_probability"       # 包含预测概率的列名
    pred_col: "prediction_class"             # 包含离散预测的列名(可选)
    split_col: "split"                       # 指示数据分割的列名(例如"train"或"test")
```

### 合并数据配置

```yaml
# 控制如何合并来自不同模型的预测数据
merged_data:
  enabled: true                           # 是否将不同模型的预测合并到单个数据集中
  save_name: "combined_predictions.csv"   # 保存的合并数据集文件名
```

### 分割配置

```yaml
# 控制是否分别分析训练集和测试集
split:
  enabled: true                           # 是否为不同的数据分割生成单独的分析
```

### 可视化配置

```yaml
# 控制生成哪些性能图表及其属性
visualization:
  # ROC曲线配置
  roc:
    enabled: true                         # 是否生成ROC曲线图
    save_name: "roc_curves.pdf"           # 保存的ROC曲线图文件名
    title: "ROC Curves Comparison"        # ROC曲线图上显示的标题
  
  # 决策曲线分析(DCA)配置
  dca:
    enabled: true                         # 是否生成决策曲线分析图
    save_name: "decision_curves.pdf"      # 保存的决策曲线图文件名
    title: "Decision Curve Analysis"      # 决策曲线图上显示的标题
  
  # 校准曲线配置
  calibration:
    enabled: true                         # 是否生成校准曲线图
    save_name: "calibration_curves.pdf"   # 保存的校准曲线图文件名
    n_bins: 10                            # 用于校准曲线计算的分箱数
    title: "Calibration Curves"           # 校准曲线图上显示的标题
  
  # 精确率-召回率曲线配置
  pr_curve:
    enabled: true                         # 是否生成精确率-召回率曲线图
    save_name: "precision_recall_curves.pdf"  # 保存的精确率-召回率曲线图文件名
    title: "Precision-Recall Curves"      # 精确率-召回率曲线图上显示的标题
```

### DeLong检验配置

```yaml
# 控制使用DeLong检验进行ROC曲线之间的统计比较
delong_test:
  enabled: true                           # 是否执行DeLong检验以比较AUCs
  save_name: "delong_results.json"        # 保存DeLong检验结果的文件名
```

### 指标配置

```yaml
# 指标计算配置
metrics:
  # 基本指标配置
  basic_metrics:
    enabled: true                         # 是否计算基本指标
  
  # Youden指数指标配置
  youden_metrics:
    enabled: true                         # 是否计算Youden指数指标
  
  # 目标指标配置
  target_metrics:
    enabled: true                         # 是否计算目标指标
    targets:                              # 要计算的目标指标
      sensitivity: 0.9                    # 敏感性目标
      specificity: 0.8                    # 特异性目标
```

## 功能模块

### 1. 数据合并和分组

工具能从多个文件读取模型预测，合并为单个数据集，并可选择按照指定的分割列（如train/test）进行分组分析。

**核心功能**：
- 读取多个模型预测文件
- 将不同模型的预测合并到一个统一的数据集
- 根据split列将数据分组(例如训练集vs测试集)
- 保存合并后的数据集供进一步分析

### 2. 生成可视化图表

为每个模型和每个数据分组（如果启用）生成以下可视化图表：

**ROC曲线**：
- 显示每个模型的受试者工作特征曲线
- 包括AUC值作为模型性能指标

**决策曲线分析(DCA)**：
- 显示在不同阈值概率下各模型的临床净获益
- 包括"Treat All"和"Treat None"基线

**校准曲线**：
- 评估预测概率的校准程度（即，预测概率与实际结果的匹配程度）
- 支持自定义分箱数量

**精确率-召回率曲线**：
- 展示不同阈值下每个模型的精确率和召回率权衡
- 适用于处理不平衡数据集

### 3. 计算性能指标

计算并比较多种性能指标，支持不同阈值的指标计算：

**基本指标**：
- 准确率(Accuracy)、精确率(Precision)、召回率(Recall)、F1分数
- 特异性(Specificity)、敏感性(Sensitivity)
- AUC、log loss等

**Youden指数指标**：
- 在Youden指数(敏感性+特异性-1)最大的阈值下计算指标
- 对于训练集/测试集分析，使用训练集确定的阈值应用于所有数据集

**目标指标**：
- 基于用户指定的敏感性/特异性目标计算最佳阈值
- 计算达到目标指标的阈值下的综合性能指标
- 支持同时满足多个目标的阈值搜索

### 4. 模型比较

**DeLong检验**：
- 执行DeLong检验比较不同模型的ROC曲线
- 生成p值矩阵表示模型之间的显著性差异

**指标汇总**：
- 将所有模型和数据分组的指标汇总到一个JSON文件
- 支持不同指标间的比较和分析

## 执行流程

1. 解析命令行参数获取配置文件路径
2. 创建ModelComparisonTool实例并读取配置
3. 读取预测文件并准备数据
   - 读取每个模型的预测文件
   - 合并数据并添加分割信息
   - 按分组创建数据子集(如果启用)
4. 保存合并的预测数据
5. 执行模型评估
   - 对每个数据分组生成可视化图表
   - 计算和比较性能指标
   - 执行DeLong检验比较ROC曲线
6. 保存所有计算得到的指标到JSON文件

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `combined_predictions.csv`: 包含所有模型预测的合并数据集
2. 每个分组的子目录(如"train"、"test"等)，每个子目录中包含：
   - ROC曲线图表
   - 决策曲线分析图表
   - 校准曲线图表
   - 精确率-召回率曲线图表
   - DeLong检验结果
3. `metrics/metrics.json`: 包含所有模型和分组的性能指标

## 注意事项

1. 确保所有预测文件都包含必需的列(subject_id_col, label_col, prob_col)
2. 对于分组分析，训练集上确定的阈值将应用于所有数据集
3. 确保所有文件中的受试者ID格式一致，以便正确合并数据
4. 该工具主要设计用于二分类问题，不支持多分类情况
5. 在同一分析中的所有模型应该针对相同的目标变量 