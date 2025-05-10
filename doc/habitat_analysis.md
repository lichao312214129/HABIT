# Habitat Analysis Documentation

## Overview

HABIT (HAbitat Based Imaging & Texture-analysis) 是一个用于医学影像分析的综合工具，可进行两步聚类分析：
1. 个体级聚类：将每个肿瘤分割成超体素(supervoxels)
2. 群体级聚类：在群体水平上对超体素进行分类，将其划分为栖息地(habitats)

### 系统要求
- Python 3.8+
- 依赖包: numpy, pandas, scikit-learn, SimpleITK, matplotlib
- 最低内存: 8GB (推荐 16GB+)
- 存储空间: 取决于数据集大小 (推荐 50GB+ 可用空间)

### 性能考虑
- 处理时间取决于图像大小和特征数量
- 内存使用随图像尺寸和超体素数量而变化
- 特征提取和聚类支持并行处理

## 主要组件

### 1. HabitatAnalysis 类

HabitatAnalysis是主要的分析类，负责处理从特征提取到栖息地聚类的整个流程。

#### 参数

| 参数 | 类型 | 描述 | 默认值 | 验证规则 |
|-----------|------|-------------|---------|-----------------|
| root_folder | str | 数据根目录 | 必填 | 必须存在且可读 |
| out_folder | str | 输出目录 | root_folder/habitats_output | 如不存在将被创建 |
| feature_config | dict | 特征配置字典 | 必填 | 必须遵循指定结构 |
| supervoxel_clustering_method | str | 超体素聚类方法 | "kmeans" | 必须是: ["kmeans", "hierarchical", "spectral", "dbscan", "mean_shift", "gmm", "affinity_propagation"] 之一 |
| habitat_clustering_method | str | 栖息地聚类方法 | "kmeans" | 必须是: ["kmeans", "hierarchical", "spectral", "dbscan", "mean_shift", "gmm", "affinity_propagation"] 之一 |
| n_clusters_supervoxel | int | 超体素聚类数量 | 50 | 必须 > 0 且 < 1000 |
| n_clusters_habitats_max | int | 最大栖息地聚类数量 | 10 | 必须 > n_clusters_habitats_min |
| n_clusters_habitats_min | int | 最小栖息地聚类数量 | 2 | 必须 > 0 且 < n_clusters_habitats_max |
| habitat_cluster_selection_method | str/list | 选择栖息地聚类数量的方法 | None | 必须是: ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin", "aic", "bic", "gap", "cophenetic"] 之一或多个组合 |
| best_n_clusters | int | 直接指定最佳聚类数量 | None | 必须在 n_clusters_habitats_min 和 n_clusters_habitats_max 之间 |
| n_processes | int | 并行处理数量 | 1 | 必须 > 0 且 <= CPU核心数 |
| random_state | int | 随机种子 | 42 | 任意整数 |
| verbose | bool | 是否输出详细信息 | True | 布尔值 |
| images_dir | str | 图像目录名 | "images" | 必须存在于root_folder中 |
| masks_dir | str | 掩码目录名 | "masks" | 必须存在于root_folder中 |
| plot_curves | bool | 是否绘制评估曲线 | True | 布尔值 |
| progress_callback | callable | 进度回调函数 | None | 如果提供，必须是可调用的 |
| save_intermediate_results | bool | 是否保存中间结果 | False | 布尔值 |

#### 特征配置

特征配置字典应具有以下结构:

```yaml
FeatureConstruction:
  voxel_level:  # 体素级特征提取
    image_names: [T1, T2, T2FLAIR]  # 图像名称列表
    method: func1(func2(T1, params_func2), func3(T2, params_func3), params_func1)  
    params:  # 特征提取参数
      params_func1: your_params_value1
      params_func2: your_params_value2
      params_func3: your_params_value3
  
  supervoxel_level:  # 超体素级特征提取（可选）
    supervoxel_file_keyword: '*_supervoxel.nrrd'  # 超体素文件关键字
    method: func1(func2(T1, params_func2), func3(T2, params_func3), params_func1)  
    params:
      params_func1: your_params_value1
      params_func2: your_params_value2
      params_func3: your_params_value3
      
  preprocessing:  # 特征预处理配置（可选）
    methods:
      - method: minmax
        global_normalize: true
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: true
```

##### 配置示例

1. 基本动态对比增强(DCE)特征配置:
```yaml
FeatureConstruction:
  voxel_level:
    image_names: [pre_contrast, LAP, PVP, delay_3min]
    method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
    params:
      timestamps: path/to/scan_time_of_phases.xlsx
  
  preprocessing:
    methods:
      - method: minmax
        global_normalize: true
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: true
```

2. 多模态放射组学特征配置:
```yaml
FeatureConstruction:
  voxel_level:
    image_names: [T1, T2, T2FLAIR]
    method: concat(voxel_radiomics(T1, params_file), voxel_radiomics(T2, params_file), voxel_radiomics(T2FLAIR, params_file))
    params:
      params_file: ./config/params_voxel_radiomics.yaml
      kernelRadius: 2
  
  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params:
      aggregation: "mean"
```

##### 支持的特征构建函数:

1. **原始特征**:
   - `raw`: 从感兴趣区域提取原始图像强度
   - 参数: image_data (SimpleITK图像或路径), mask_data (SimpleITK图像或路径)
   - 返回: 包含原始强度值的DataFrame
   - 性能: 快速，内存占用最小

2. **体素放射组学特征**:
   - `voxel_radiomics`: 提取基于体素的放射组学特征
   - 参数: image_data, mask_data, params_file, kernelRadius
   - 返回: 包含基于体素的放射组学特征的DataFrame
   - 性能: 中等，内存占用取决于内核大小
   - 支持的特征类别: firstorder, shape, glcm, glrlm, glszm, ngtdm, gldm

3. **超体素放射组学特征**:
   - `supervoxel_radiomics`: 为每个超体素提取放射组学特征
   - 参数: image_data, supervoxel_map, params_file
   - 返回: 包含基于超体素的放射组学特征的DataFrame
   - 性能: 中高，取决于超体素数量

4. **均值体素特征**:
   - `mean_voxel_features`: 计算每个超体素的均值
   - 参数: image_data, supervoxel_map
   - 返回: 包含每个超体素平均特征的DataFrame
   - 性能: 快速，低内存占用

5. **动态对比增强特征**:
   - `kinetic`: 提取动态对比增强(DCE)序列的动态特征
   - 参数: pre_contrast, early_phase, late_phase, delay_phase, timestamps
   - 返回: 包含动态特征的DataFrame
   - 性能: 中等，取决于时间点数量

6. **特征预处理**:
   - `process_features_pipeline`: 应用一系列预处理方法
   - 支持的方法:
     - minmax: 最小-最大归一化 (快速，内存高效)
     - standard: 标准化 (中等，内存高效)
     - robust: 稳健缩放 (中等，内存高效)
     - quantile: 分位数变换 (慢，高内存占用)
     - kmeans: K-means离散化 (中等，内存高效)
     - winsorize: 温瑟方法处理离群值 (快速，内存高效)
     - log: 对数变换 (快速，内存高效)

7. **特征合并**:
   - `concat`: 合并多个特征提取器的结果
   - 参数: 多个特征提取器的结果
   - 返回: 合并的特征DataFrame
   - 性能: 快速，内存占用取决于输入特征数量

#### 聚类方法

##### 超体素聚类
1. **K-means**
   - 优点: 快速，可扩展，适用于大型数据集
   - 缺点: 对初始化敏感，需要指定聚类数量
   - 最适用于: 具有明确聚类边界的大型数据集

2. **层次聚类 (Hierarchical)**
   - 优点: 无需指定聚类数量，可处理非球形聚类
   - 缺点: 计算成本高，内存密集
   - 最适用于: 具有复杂聚类形状的小型到中型数据集

3. **谱聚类 (Spectral)**
   - 优点: 可处理非球形聚类，对噪声具有鲁棒性
   - 缺点: 计算成本高，对参数敏感
   - 最适用于: 需要降噪的复杂聚类形状

4. **高斯混合模型 (GMM)**
   - 优点: 基于概率模型，提供聚类概率
   - 缺点: 对初始化敏感，计算成本较高
   - 最适用于: 遵循高斯分布的数据

5. **DBSCAN**
   - 优点: 无需指定聚类数量，可识别噪声点，可处理任意形状的聚类
   - 缺点: 需要仔细调整参数
   - 最适用于: 非球形聚类和噪声数据

6. **Mean Shift**
   - 优点: 无需指定聚类数量，可处理非球形聚类
   - 缺点: 计算复杂度高
   - 最适用于: 需要自动检测聚类数量的复杂分布

7. **亲和力传播 (Affinity Propagation)**
   - 优点: 自动确定聚类数量，可处理非均匀大小的聚类
   - 缺点: 计算复杂度高，不适合大规模数据
   - 最适用于: 小型到中型数据集

## 聚类验证方法

### 验证策略

#### 单一评估方法
- **选择标准**:
  1. 对于轮廓系数(Silhouette)和Calinski-Harabasz指数:
     - 选择得分最高的聚类数量
     - 考虑局部最大值以提高稳健性
  2. 对于惯性(Inertia)、BIC和AIC:
     - 使用肘部法则
     - 计算一阶和二阶差分
     - 选择具有最大二阶差分点之后的聚类数量
     - 考虑多个潜在的肘部点
  3. 对于其他方法:
     - 默认选择得分最高的聚类数量
     - 如可能，考虑置信区间

#### 多评估方法
- **组合得分计算**:
  1. 拆分组合方法名称
  2. 对于每种评估方法:
     - Silhouette和Calinski-Harabasz: 对较高分数进行归一化
     - Inertia、BIC、AIC: 对较低分数进行反转和归一化
  3. 累积归一化得分
  4. 选择具有最高组合得分的聚类数量
  5. 考虑置信区间和稳定性

### 验证方法

1. **inertia**
   - 衡量到聚类中心的平方距离之和
   - 较低值表示定义更好的聚类
   - 最适用于: K-means聚类
   - 局限性: 对尺度敏感，不适用于非球形聚类
   - 支持算法: kmeans

2. **silhouette**
   - 衡量聚类的内聚性和分离性
   - 较高值表示定义更好的聚类
   - 最适用于: 球形聚类，中等数据集大小
   - 局限性: 大型数据集计算成本高
   - 支持算法: kmeans, hierarchical, spectral, dbscan, mean_shift, gmm, affinity_propagation

3. **calinski_harabasz**
   - 比较聚类间和聚类内方差
   - 较高值表示更好的聚类分离
   - 最适用于: K-means聚类，大型数据集
   - 局限性: 假设聚类为球形
   - 支持算法: kmeans, hierarchical, spectral, dbscan, mean_shift, gmm, affinity_propagation

4. **davies_bouldin**
   - 衡量聚类相似性
   - 较低值表示定义更好的聚类
   - 最适用于: 球形聚类，中等数据集大小
   - 局限性: 对噪声敏感
   - 支持算法: kmeans, hierarchical, spectral, mean_shift, affinity_propagation

5. **aic** (Akaike信息准则)
   - 估计统计模型的质量
   - 较低值表示更好的模型拟合
   - 最适用于: 模型选择，大型数据集
   - 局限性: 假设正态分布
   - 支持算法: gmm

6. **bic** (贝叶斯信息准则)
   - 估计模型复杂性和拟合优度
   - 较低值表示更好的模型选择
   - 最适用于: 模型选择，大型数据集
   - 局限性: 假设正态分布
   - 支持算法: gmm

7. **gap** (间隙统计量)
   - 比较实际数据与随机数据的聚类
   - 较高值表示更好的聚类结构
   - 最适用于: 评估聚类有效性
   - 局限性: 计算复杂度高
   - 支持算法: kmeans

8. **cophenetic** (共树系数)
   - 评估层次聚类的质量
   - 较高值表示更好的层次表示
   - 最适用于: 仅层次聚类
   - 局限性: 仅适用于层次聚类
   - 支持算法: hierarchical

## 使用示例

### 基本使用
```python
from habit.core.habitat_analysis import HabitatAnalysis

# 初始化分析
analysis = HabitatAnalysis(
    root_folder="path/to/data",
    feature_config={
        "voxel_level": {
            "image_names": ["T1", "T2"],
            "method": "voxel_radiomics",
            "params": {
                "kernelRadius": 2,
                "params_file": "./config/params_voxel_radiomics.yaml"
            }
        }
    }
)

# 运行分析
analysis.run()
```

### 高级使用
```python
from habit.core.habitat_analysis import HabitatAnalysis

# 使用自定义设置初始化分析
analysis = HabitatAnalysis(
    root_folder="path/to/data",
    feature_config={
        "voxel_level": {
            "image_names": ["pre_contrast", "LAP", "PVP", "delay_3min"],
            "method": "kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)",
            "params": {
                "timestamps": "path/to/scan_time_of_phases.xlsx"
            }
        },
        "preprocessing": {
            "methods": [
                {"method": "minmax", "global_normalize": True},
                {"method": "winsorize", "winsor_limits": [0.05, 0.05]}
            ]
        }
    },
    n_processes=4,
    supervoxel_clustering_method="kmeans",
    habitat_clustering_method="kmeans",
    n_clusters_supervoxel=50,
    habitat_cluster_selection_method=["silhouette", "calinski_harabasz"]
)

# 使用进度回调运行分析
def progress_callback(progress):
    print(f"进度: {progress}%")

analysis.run(progress_callback=progress_callback)
```

### 使用YAML配置文件
```python
import yaml
from habit.core.habitat_analysis import HabitatAnalysis
from habit.utils.config_utils import load_config

# 加载配置文件
config = load_config("./config/config_getting_habitat.yaml")

# 初始化分析
analysis = HabitatAnalysis(
    root_folder=config["data_dir"],
    out_folder=config["out_dir"],
    feature_config=config["FeatureConstruction"],
    supervoxel_clustering_method=config["HabitatsSegmention"]["supervoxel"]["algorithm"],
    n_clusters_supervoxel=config["HabitatsSegmention"]["supervoxel"]["n_clusters"],
    habitat_clustering_method=config["HabitatsSegmention"]["habitat"]["algorithm"],
    n_clusters_habitats_max=config["HabitatsSegmention"]["habitat"]["max_clusters"],
    habitat_cluster_selection_method=config["HabitatsSegmention"]["habitat"]["habitat_cluster_selection_method"],
    best_n_clusters=config["HabitatsSegmention"]["habitat"]["best_n_clusters"],
    n_processes=config.get("processes", 1),
    random_state=config.get("random_state", 42),
    plot_curves=config.get("plot_curves", True)
)

# 运行分析
analysis.run()
```

## 常见问题和解决方案

1. **内存问题**
   - 问题: 特征提取过程中内存不足
   - 解决方案: 减少特征数量，使用较小的内核大小，增加交换空间，减少并行处理数量

2. **性能问题**
   - 问题: 处理时间过长
   - 解决方案: 启用并行处理，减少特征数量，使用更快的聚类方法，减小图像分辨率

3. **聚类质量**
   - 问题: 聚类分离不佳
   - 解决方案: 尝试不同的聚类方法，调整聚类数量，使用不同的特征集，优化特征预处理

4. **验证问题**
   - 问题: 聚类选择不一致
   - 解决方案: 使用多种验证方法，考虑稳定性分析，增加最小/最大聚类范围

## 性能优化

1. **特征提取**
   - 为您的数据选择适当的特征类型
   - 考虑特征重要性和冗余性
   - 尽可能使用并行处理
   - 使用更小的ROI或降低图像分辨率

2. **聚类**
   - 选择合适的聚类方法
   - 优化聚类数量
   - 使用高效的实现
   - 对于大型数据集，考虑使用更快的算法如K-means

3. **内存管理**
   - 分块处理数据
   - 清除中间结果
   - 使用适当的数据类型
   - 启用save_intermediate_results=False以减少内存占用

## 错误处理

1. **输入验证**
   - 检查文件是否存在
   - 验证参数范围
   - 验证数据格式
   - 确保图像和掩码维度匹配

2. **运行时错误**
   - 处理内存错误
   - 管理并行处理问题
   - 记录错误以进行调试
   - 使用verbose=True获取详细日志

3. **输出验证**
   - 验证结果一致性
   - 检查输出文件完整性
   - 验证聚类质量
   - 检查栖息地图是否覆盖整个ROI

