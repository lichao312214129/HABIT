# app_habitat_test_retest_mapper.py 功能文档

## 功能概述

`app_habitat_test_retest_mapper.py` 是HABIT工具包中用于评估生境分析可重复性的专用工具。该模块通过对测试-重测数据（同一患者的多次扫描）进行分析，评估生境分割的稳定性和可靠性。这对于建立生境分析的临床适用性和验证其作为生物标志物的潜力至关重要。

## 用法

```bash
python scripts/app_habitat_test_retest_mapper.py --config <config_file_path>
```

## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | YAML配置文件路径（必需） |

## 配置文件格式

`app_habitat_test_retest_mapper.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据路径
test_dir: <测试数据目录路径>
retest_dir: <重测数据目录路径>
out_dir: <输出目录路径>

# 文件匹配
file_patterns:
  test: <测试文件匹配模式>
  retest: <重测文件匹配模式>
  mapping: <测试-重测映射文件>

# 分析配置
analysis:
  metrics: <评估指标列表>
  visualization: <可视化设置>
  statistics: <统计分析设置>
```

### 文件匹配配置

配置文件中的`file_patterns`部分允许灵活定义测试和重测文件的匹配方式：

```yaml
file_patterns:
  # 文件匹配模式
  test: "*.nrrd"  # 测试数据文件匹配模式
  retest: "*.nrrd"  # 重测数据文件匹配模式
  
  # 测试-重测映射方式，可以是以下之一：
  # 1. 映射文件路径
  mapping: "path/to/mapping.csv"  
  
  # 2. 文件名匹配规则
  prefix_pattern:
    test: "patient_{id}_scan1"
    retest: "patient_{id}_scan2"
```

### 分析配置

分析配置部分定义了测试-重测评估的具体内容：

```yaml
analysis:
  # 评估指标
  metrics:
    - "dice"  # 骰子系数
    - "jaccard"  # Jaccard索引
    - "hausdorff"  # Hausdorff距离
    - "volume_ratio"  # 体积比
    - "habitat_stability"  # 生境稳定性指数
  
  # 可视化设置
  visualization:
    overlay_images: true  # 是否生成叠加图
    difference_maps: true  # 是否生成差异图
    colormap: "jet"  # 颜色映射
  
  # 统计分析设置
  statistics:
    confidence_level: 0.95  # 置信水平
    permutation_tests: 1000  # 置换测试次数
```

## 支持的评估指标

测试-重测分析支持以下评估指标：

1. **骰子系数 (dice)**：测量两个生境分割之间的空间重叠度
2. **Jaccard索引 (jaccard)**：另一种衡量空间重叠度的指标
3. **Hausdorff距离 (hausdorff)**：测量两个生境边界之间的最大距离
4. **体积比 (volume_ratio)**：测试与重测体积的比率
5. **体积差异百分比 (volume_diff_percent)**：体积差异的百分比
6. **质心距离 (centroid_distance)**：生境质心之间的距离
7. **生境稳定性指数 (habitat_stability)**：综合评价生境稳定性的指标
8. **表面距离 (surface_distance)**：平均表面距离
9. **重叠体积比例 (overlap_fraction)**：重叠体积占总体积的比例

## 执行流程

1. 加载配置文件和测试-重测映射信息
2. 读取测试和重测数据
3. 配对测试和重测样本
4. 计算各种评估指标
5. 生成可视化结果
6. 执行统计分析
7. 生成报告

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `metrics/`: 存储各评估指标的CSV文件
2. `visualization/`: 生境叠加图和差异图
3. `statistics/`: 统计分析结果
4. `summary_report.pdf`: 测试-重测分析总结报告

## 完整配置示例

```yaml
# 基本配置
test_dir: ./data/test_scans
retest_dir: ./data/retest_scans
out_dir: ./results/test_retest_analysis

# 文件匹配配置
file_patterns:
  test: "*_habitats.nrrd"
  retest: "*_habitats.nrrd"
  mapping: "./data/test_retest_mapping.csv"

# 分析配置
analysis:
  metrics:
    - "dice"
    - "jaccard"
    - "hausdorff"
    - "volume_ratio"
    - "volume_diff_percent"
    - "centroid_distance"
    - "habitat_stability"
    - "surface_distance"
    - "overlap_fraction"
  
  visualization:
    overlay_images: true
    difference_maps: true
    colormap: "jet"
    save_formats: ["png", "pdf"]
    slice_view: "axial"
  
  statistics:
    confidence_level: 0.95
    permutation_tests: 1000
    bland_altman: true
    icc_analysis: true
```

## 映射文件格式

如果使用CSV文件来定义测试-重测映射，格式应为：

```csv
test_id,retest_id
patient001_scan1,patient001_scan2
patient002_scan1,patient002_scan2
...
```

## 注意事项

1. 确保测试和重测数据使用相同的预处理和生境分析方法
2. 建议使用相同扫描仪和扫描参数获取的测试-重测数据
3. 体积较小的生境可能出现较大的变异性
4. 评估结果应与临床意义相结合进行解释
5. 生境稳定性可能受患者状态、扫描条件和分析参数的影响 