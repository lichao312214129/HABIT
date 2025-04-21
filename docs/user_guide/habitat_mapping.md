# 生境地图生成

生境地图是HABIT工作流程的第一步，也是最关键的步骤之一。本节将详细介绍如何使用HABIT生成生境地图。

## 工作原理

生境地图生成基于两级聚类算法：

1. **第一级聚类（超体素级别）**：将每个肿瘤分割成多个超体素
2. **第二级聚类（生境级别）**：将所有样本的超体素聚类，得到能够代表所有样本中相似区域的生境

这种方法的优势在于可以识别出在不同患者之间具有相似特征的组织区域，便于后续的分析和比较。

## 数据准备

在生成生境地图之前，您需要准备以下数据：

1. **原始医学图像**：如CT、MRI等，可以是单序列或多序列
2. **感兴趣区域（ROI）分割结果**：通常是肿瘤的分割掩码

数据应当按照以下结构组织：

```
dataset/
├── images/              # 原始图像目录
│   ├── patient001/      # 患者001
│   │   ├── T1/          # T1序列
│   │   │   └── img.nii.gz  # 图像文件
│   │   ├── T2/          # T2序列
│   │   │   └── img.nii.gz  # 图像文件
│   ├── patient002/      # 患者002
│   │   ├── ...
├── masks/               # 分割掩码目录
│   ├── patient001/      # 患者001
│   │   ├── T1/          # T1序列对应的掩码
│   │   │   └── mask.nii.gz  # 掩码文件
│   │   ├── T2/          # T2序列对应的掩码
│   │   │   └── mask.nii.gz  # 掩码文件
│   ├── patient002/      # 患者002
│   │   ├── ...
```

## 配置文件

生成生境地图需要准备配置文件，通常使用YAML格式。以下是一个典型的配置示例：

```yaml
# 基本参数
data_dir: /path/to/data_directory  # 数据根目录
out_dir: /path/to/output_directory  # 输出目录
processes: 4  # 并行处理进程数
plot_curves: true  # 是否生成评估曲线图
random_state: 42  # 随机种子，确保结果可重复

# 特征提取参数
FeatureConstruction:
  method: simple  # 特征提取方法: simple, kinetic, custom
  image_names: [T1, T2]  # 图像序列名称
  params: {}  # 特征提取额外参数

# 生境分割参数
HabitatsSegmention:
  supervoxel:  # 超体素聚类参数
    algorithm: kmeans  # 可选: kmeans, gmm
    n_clusters: 50  # 超体素聚类数量
  
  habitat:  # 生境聚类参数
    algorithm: kmeans  # 可选: kmeans, gmm
    max_clusters: 10  # 最大生境数量
    min_clusters: 2  # 最小生境数量
    habitat_cluster_selection_method: silhouette  # 最优聚类数评估方法，可选: silhouette, calinski_harabasz, elbow
```

## 命令行使用

使用以下命令生成生境地图：

```bash
python scripts/generate_habitat_map.py --config config/your_config.yaml
```

您也可以启用调试模式以获取更详细的日志信息：

```bash
python scripts/generate_habitat_map.py --config config/your_config.yaml --debug
```

## 输出结果

生成的生境地图将保存在指定的输出目录中，主要包括以下文件：

1. **habitats.csv**：包含每个超体素的特征和生境标签
2. **patient001_habitats.nrrd**：每个患者的生境地图
3. **habitat_clustering_scores.png**：聚类评估曲线图
4. **habitats_stats.csv**：各生境的统计信息

## 高级选项

### 特征提取方法

HABIT支持多种特征提取方法：

- **simple**：使用原始图像强度值作为特征
- **kinetic**：适用于动态增强图像，提取动态特征
- **custom**：自定义特征提取，需实现特定接口

### 聚类算法

HABIT支持多种聚类算法：

- **kmeans**：K均值聚类，适合大多数场景
- **gmm**：高斯混合模型，适合处理复杂分布
- **spectral**：谱聚类，适合复杂形状的聚类

### 最优聚类数选择

HABIT提供多种评估方法来确定最优的生境数量：

- **silhouette**：轮廓系数，衡量聚类紧密度和分离度
- **calinski_harabasz**：方差比准则，衡量聚类间和聚类内方差比
- **elbow**：肘部法则，通过惯性下降确定合适的聚类数

## 注意事项

1. 对于大型数据集，建议增加进程数以加速处理
2. 调整超体素聚类数量可以控制结果的粒度
3. 生境地图生成是一个计算密集型操作，可能需要较长时间
4. 建议为不同数据集使用不同的配置文件 