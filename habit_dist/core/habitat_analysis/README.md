# Habitat 聚类分析模块

Habitat 聚类分析模块是一个用于医学图像栖息地分析的 Python 库。它提供了一套完整的工具，用于从医学图像中提取特征并执行聚类分析，以发现数据中的内在模式。

## 功能特点

- 支持灵活的特征提取配置
- 提供多种聚类算法（K-Means、GMM）
- 自动确定最佳聚类数量
- 并行处理以提高性能
- 生成详细的评估指标和可视化结果
- 支持多种聚类验证方法
- 支持批处理处理大规模数据
- 可扩展的特征提取器和聚类算法框架

## 模块结构

```
habitat_analysis/
├── __init__.py             # 包初始化文件
├── run_habitat_analysis.py # 命令行接口
├── habitat_analysis.py     # 主要分析类
├── features/               # 特征提取模块
│   ├── __init__.py               # 特征提取器包初始化
│   ├── base_feature_extractor.py # 特征提取器基类和注册机制
│   ├── feature_extractor_factory.py # 特征提取器工厂
│   ├── kinetic_feature_extractor.py # 动态特征提取器
│   ├── simple_feature_extractor.py  # 简单特征提取器
│   └── custom_feature_extractor_template.py # 自定义特征提取器模板
├── clustering/             # 聚类算法模块
│   ├── __init__.py            # 聚类算法包初始化
│   ├── base_clustering.py     # 聚类算法基类和注册机制
│   ├── kmeans_clustering.py   # K均值聚类实现
│   ├── gmm_clustering.py      # 高斯混合模型聚类实现
│   ├── cluster_validation_methods.py # 聚类验证方法
│   └── custom_clustering_template.py # 自定义聚类算法模板
└── utils/                  # 工具函数模块
    ├── __init__.py         # 工具包初始化
    ├── io_utils.py         # 输入输出工具函数
    ├── visualization.py    # 可视化工具函数
    ├── progress_utils.py   # 进度条工具类
    └── config_utils.py     # 配置工具函数
```

## 使用方法

### 命令行使用

可以通过命令行直接运行分析：

```bash
python -m habitat_analysis.run_habitat_analysis --config config_example.yaml
```

### 配置文件格式

创建配置文件 (YAML 格式)：

```yaml
# 数据路径
data_dir: "/path/to/data"
out_dir: "/path/to/output"

# 预处理
Preprocessing:
  N4BiasCorrection: true
    images: [pre_contrast, LAP, PVP, delay_3min]
    yourParams:
  
  resample:
    images: [pre_contrast, LAP, PVP, delay_3min]
    yourParams:

  registration:
    images: [pre_contrast, LAP, PVP, delay_3min]
    fixedImage: PVP
    movingImage: [pre_contrast, LAP, delay_3min]

# 特征构建
FeatureConstruction:
  # 特征提取方法
  method: kinetic  # 可选: kinetic, original, 持续更新
  # 时间序列特征
  timestamps: /path/to/timestamps.xlsx

# 栖息地分割
HabitatsSegmention:
  method: 2step  # 目前只支持二步聚类法
  # 超像素方法配置  
  supervoxel:
    algorithm: gmm  # 可选: kmeans, gmm
    n_clusters: 50
  
  # Habitat方法配置
  habitat:
    algorithm: gmm  # 可选: kmeans, gmm
    max_clusters: 10
    min_clusters: 2
    validation_methods: [aic, bic, silhouette]  # 可选验证方法
    
# 其他参数
processes: 4
plot_curves: true
random_state: 42
batch_size: 10
save_intermediate_results: false
```

### 通过 Python API 使用

```python
from habitat_analysis.habitat_analysis import HabitatAnalysis

# 创建分析实例
analysis = HabitatAnalysis(
    root_folder="/path/to/data",
    out_folder="/path/to/output",
    image_names=["pre_contrast", "LAP", "PVP", "delay_3min"],
    supervoxel_clustering_method="kmeans",
    n_clusters_supervoxel=50,
    habitat_clustering_method="gmm",
    n_clusters_habitats_max=10,
    n_clusters_habitats_min=2,
    habitat_cluster_selection_method=["aic", "bic", "silhouette"],
    feature_extractor="kinetic",
    feature_params={
        "timestamps": "/path/to/timestamps.xlsx",
        "normalize": True
    },
    n_processes=4,
    batch_size=10,
    save_intermediate_results=False
)

# 运行分析
results = analysis.run()
```

## 特征提取器

### 动态特征提取器 (Kinetic Feature Extractor)

动态特征提取器用于从动态增强序列中提取时间序列特征。目前提取的特征包括：

1. **洗入速率 (Wash-in Rate)**: 从基线到峰值的增强速率
2. **洗出速率 (Wash-out Rate)**: 从峰值到最终时间点的下降速率
3. **峰值增强 (Peak Enhancement)**: 峰值与基线之间的差值

动态特征提取器需要时间戳信息，可以通过Excel文件提供。

### 简单特征提取器 (Simple Feature Extractor)

简单特征提取器直接从图像中提取基本特征，如：
- 像素值
- 统计特征（均值、标准差等）
- 纹理特征

## 聚类算法

### K-Means 聚类

K-Means 是一种基于距离的聚类算法，通过最小化簇内平方和来划分数据。

### 高斯混合模型 (GMM)

GMM 是一种基于概率的聚类算法，假设数据是由多个高斯分布混合生成的。

### 聚类验证方法

支持多种聚类验证方法：
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

## 扩展功能

### 添加新的特征提取器

1. 在 `features/` 目录下创建新的特征提取器类
2. 继承 `BaseFeatureExtractor` 类
3. 实现 `extract_features` 方法，接受 `image_data` 和 `**kwargs`
4. 使用 `@register_feature_extractor` 装饰器注册

示例：

```python
from habitat_clustering.features.base_feature_extractor import BaseFeatureExtractor, register_feature_extractor

@register_feature_extractor("my_extractor")
class MyFeatureExtractor(BaseFeatureExtractor):
    # 声明是否需要时间戳
    requires_timestamp = True  # 或 False
    
    def __init__(self, timestamps=None, param1=1, param2=2, **kwargs):
        super().__init__(timestamps=timestamps, **kwargs)
        self.param1 = param1
        self.param2 = param2
        self.feature_names = ["feature1", "feature2", ...]
        
    def extract_features(self, image_data, **kwargs):
        # 从kwargs中获取参数
        timestamps = kwargs.get('timestamps', None)
        subject = kwargs.get('subject', None)
        
        # 实现特征提取逻辑
        features = ...
        
        return features
```

### 添加新的聚类算法

1. 在 `clustering/` 目录下创建新的聚类算法类
2. 继承 `BaseClustering` 类
3. 使用 `@register_clustering` 装饰器注册

示例：

```python
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering

@register_clustering("my_clustering")
class MyClustering(BaseClustering):
    def __init__(self, n_clusters=5, random_state=None):
        super().__init__(n_clusters, random_state)
        
    def fit(self, X):
        # 实现拟合逻辑
        ...
        
    def predict(self, X):
        # 实现预测逻辑
        ...
        
    def find_optimal_clusters(self, X, min_clusters=2, max_clusters=10, random_state=None):
        # 实现最佳聚类数确定方法
        ...
```

## 数据结构要求

程序期望以下数据结构：

```
dataset/
├── images/
│   ├── subj001/
│   │   ├── img1/
|   |   |   ├── img.nii.gz (或 img.nrrd)
│   │   ├── img2/
|   |   |   ├── img.nii.gz (或 img.nrrd)
│   ├── subj002/
│   │   ├── img1/
|   |   |   ├── img.nii.gz (或 img.nrrd)
│   │   ├── img2/
|   |   |   ├── img.nii.gz (或 img.nrrd)
├── masks/
│   ├── subj001/
│   │   ├── img1/
|   |   |   ├── mask.nii.gz (或 mask.nrrd)
│   │   ├── img2/
|   |   |   ├── mask.nii.gz (或 mask.nrrd)
│   ├── subj002/
│   │   ├── img1/
|   |   |   ├── mask.nii.gz (或 mask.nrrd)
│   │   ├── img2/
|   |   |   ├── mask.nii.gz (或 mask.nrrd)
```

时间戳文件格式 (Excel)：

```
| Name   | img1         | img2     |
|--------|--------------|----------|
| sub001 | 00-06-09     | 00-07-39 |
| sub002 | 06-50-33     | 06-51-23 |
```

## 性能优化

- 支持多进程并行处理
- 批处理处理大规模数据
- 可配置的中间结果保存
- 进度条显示处理进度

## 依赖项

- Python 3.7+
- SimpleITK
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- PyYAML 