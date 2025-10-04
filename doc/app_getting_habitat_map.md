# app_getting_habitat_map.py 功能文档

## 功能概述

`app_getting_habitat_map.py` 是HABIT工具包的主要入口，用于执行肿瘤生境分析。该模块通过特征提取、超体素聚类和生境聚类，实现了医学影像生境分析的完整流程。这个脚本支持命令行和图形界面两种交互方式，可以根据用户指定的配置文件执行生境分析过程。

## 用法

```bash
python scripts/app_getting_habitat_map.py --config <config_file_path>
```


## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | 配置文件路径 |
| `--debug` | 启用调试模式（可选） |

## 配置文件格式

`app_getting_habitat_map.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据路径
data_dir: <数据目录路径>
out_dir: <输出目录路径>

# 一般设置
processes: <并行进程数>
plot_curves: <是否生成曲线图>
random_state: <随机种子>
debug: <是否启用调试模式>
```

### 特征提取配置 (FeatureConstruction)

特征提取部分分为体素级和超体素级两个部分：

```yaml
FeatureConstruction:
  voxel_level:
    method: <特征提取方法表达式>
    params:
      <方法特定参数>
  
  supervoxel_level:
    supervoxel_file_keyword: <超体素文件关键字>
    method: <超体素级特征提取方法>
    params:
      <方法特定参数>

  preprocessing:
    methods:
      - method: <预处理方法1>
        <方法1参数>
      - method: <预处理方法2>
        <方法2参数>
```

### 生境分割配置 (HabitatsSegmention)

```yaml
HabitatsSegmention:
  # 超体素聚类设置
  supervoxel:
    algorithm: <聚类算法>
    n_clusters: <超体素数量>
    random_state: <随机种子>
    max_iter: <最大迭代次数>
    n_init: <初始化次数>
  
  # 生境聚类设置
  habitat:
    mode: <模式>  # training或testing
    algorithm: <聚类算法>
    max_clusters: <最大生境数量>
    min_clusters: <最小生境数量>  # 可选，默认为2
    habitat_cluster_selection_method: <聚类数量选择方法>
    best_n_clusters: <生境数量>  # 设置为null进行自动选择
    random_state: <随机种子>
    max_iter: <最大迭代次数>
    n_init: <初始化次数>
```

## 支持的特征提取方法

### 体素级特征提取 (voxel_level)

体素级特征提取支持以下方法：

#### 1. kinetic - 动态增强特征

提取基于时间序列的动态增强特征，如wash-in斜率、wash-out斜率等。

```yaml
method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
params:
  timestamps: <时间戳文件路径>
```

**参数说明：**
- `timestamps`: Excel文件路径，包含每个患者在各期相的扫描时间

**输出特征：**
- `wash_in_slope`: 洗入斜率
- `wash_out_slope_lap_pvp`: 动脉期到门静脉期的洗出斜率
- `wash_out_slope_pvp_dp`: 门静脉期到延迟期的洗出斜率

#### 2. voxel_radiomics - 体素级影像组学特征

使用PyRadiomics提取体素级影像组学特征。

```yaml
method: concat(voxel_radiomics(image_name))
params:
  params_voxel_radiomics: <参数文件路径>
  kernelRadius: <核半径>
```

**参数说明：**
- `params_voxel_radiomics`: PyRadiomics参数文件路径
- `kernelRadius`: 用于提取局部特征的内核半径，默认为1

**输出特征：**
根据PyRadiomics参数文件中的设置，可能包括：
- 一阶统计特征（如均值、标准差等）
- 形状特征
- 灰度共生矩阵(GLCM)特征
- 灰度游程矩阵(GLRLM)特征
- 灰度大小区域矩阵(GLSZM)特征
- 等等

#### 3. local_entropy - 局部熵特征

计算每个体素周围区域的局部熵，作为组织异质性的度量。

```yaml
method: local_entropy(raw(image_name))
params:
  kernel_size: <局部区域大小>
  bins: <直方图分箱数>
```

**参数说明：**
- `kernel_size`: 局部区域的大小，表示体素周围的立方体边长（默认为3）
- `bins`: 计算熵时使用的直方图分箱数（默认为32）

**输出特征：**
- `local_entropy`: 每个体素的局部熵值

**示例：**
```yaml
method: concat(local_entropy(raw(PVP)), voxel_radiomics(raw(PVP)))
params:
  kernel_size: 5
  bins: 32
```

**应用场景：**
- 肿瘤异质性量化
- 微环境复杂度分析
- 组织边界和过渡区识别

#### 4. concat - 特征连接

连接多个特征提取方法的结果。

```yaml
method: concat(method1(params), method2(params), ...)
```

**示例：**
```yaml
method: concat(voxel_radiomics(raw(pre_contrast)), voxel_radiomics(raw(PVP)))
```

#### 5. raw - 原始图像数据

提取原始图像数据，通常作为其他方法的输入。

```yaml
method: raw(image_name)
```

### 超体素级特征提取 (supervoxel_level)

超体素级特征提取支持以下方法：

#### 1. supervoxel_radiomics - 超体素级影像组学特征

为每个超体素提取影像组学特征。

```yaml
method: supervoxel_radiomics(image_name, params_file)
params:
  params_file: <参数文件路径>
```

**参数说明：**
- `params_file`: PyRadiomics参数文件路径

#### 2. mean_voxel_features - 体素特征平均

计算每个超体素内体素特征的平均值。

```yaml
method: mean_voxel_features()
```

### 特征预处理方法 (preprocessing)

支持的预处理方法包括：

#### 1. minmax - 最小-最大归一化

```yaml
method: minmax
global_normalize: <是否全局归一化>
```

#### 2. standard - 标准化

```yaml
method: standard
global_normalize: <是否全局归一化>
```

#### 3. robust - 鲁棒归一化

```yaml
method: robust
global_normalize: <是否全局归一化>
```

#### 4. winsorize - Winsorize变换

```yaml
method: winsorize
winsor_limits: [<下限>, <上限>]
global_normalize: <是否全局归一化>
```

#### 5. binning - 分箱离散化

将连续特征值离散化为指定数量的分箱，有助于减少噪声和异常值的影响。

```yaml
method: binning
n_bins: <分箱数量>
strategy: <分箱策略>
global_normalize: <是否全局归一化>
```

**参数说明：**
- `n_bins`: 分箱数量，默认为5
- `strategy`: 分箱策略，支持以下选项：
  - `uniform`: 等宽分箱（默认）
  - `quantile`: 等频分箱（分位数分箱）
  - `kmeans`: 基于K-均值聚类的分箱
- `global_normalize`: 是否在所有样本上进行全局分箱，默认为true

**应用场景：**
- 减少特征噪声
- 处理异常值
- 简化特征分布
- 提高模型稳定性

**示例：**
```yaml
method: binning
n_bins: 8
strategy: quantile
global_normalize: true
```

## 支持的聚类算法

### 超体素聚类 (supervoxel)

- `kmeans`: K-均值聚类
- `gmm`: 高斯混合模型
- `spectral`: 谱聚类
- `hierarchical`: 层次聚类
- `mean_shift`: Mean Shift聚类
- `dbscan`: DBSCAN密度聚类
- `affinity_propagation`: 亲和传播聚类

### 生境聚类 (habitat)

- `kmeans`: K-均值聚类（最常用）
- `gmm`: 高斯混合模型

### 聚类数量选择方法 (habitat_cluster_selection_method)

- `inertia`: 惯性（适用于K-均值）
- `aic`: Akaike信息准则（适用于GMM）
- `bic`: 贝叶斯信息准则（适用于GMM）
- `silhouette`: 轮廓系数（适用于所有算法）
- `calinski_harabasz`: Calinski-Harabasz指数（适用于所有算法）
- `davies_bouldin`: Davies-Bouldin指数（适用于所有算法）

## 完整配置示例

```yaml
# 数据路径
data_dir: your_data_dir
out_dir: your_output_dir

# 特征提取设置
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min))
    params:
      params_voxel_radiomics: ./config/params_voxel_radiomics.yaml
      kernelRadius: 2
      timestamps: F:\work\research\radiomics_TLSs\data\scan_time_of_phases.xlsx
      kernel_size: 5
      bins: 32

  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params:
      params_file: ./config/parameter.yaml

  preprocessing:
    methods:
      - method: minmax
        global_normalize: true
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: true
      - method: binning
        n_bins: 8
        strategy: quantile
        global_normalize: true

# 生境分割设置
HabitatsSegmention:
  supervoxel:
    algorithm: kmeans
    n_clusters: 50
    random_state: 42
    max_iter: 300
    n_init: 10
  
  habitat:
    mode: training  # training或testing，training表示训练新模型，testing表示使用已有模型
    algorithm: kmeans
    max_clusters: 10
    min_clusters: 2  # 可选，最小聚类数量，默认为2
    habitat_cluster_selection_method: inertia
    best_n_clusters: null  # 设置为具体数字可指定聚类数，设置为null则自动选择
    random_state: 42
    max_iter: 300
    n_init: 10

# 一般设置
processes: 2
plot_curves: true
random_state: 42
debug: false
```

## 执行流程

1. 解析命令行参数或通过图形界面选择配置文件
2. 加载配置文件和数据
3. 初始化特征提取器和聚类算法
4. 执行体素级特征提取
5. 执行超体素聚类
6. 执行生境聚类
7. 保存结果和分析图表

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. 超体素聚类结果（每个样本的超体素分割图）
2. 生境聚类结果（全部样本的共同生境）
3. 特征数据表格
4. 分析报告和聚类评估图

## 注意事项

1. 确保数据目录结构正确
2. 如果使用动态增强特征提取，必须提供正确的timestamps文件
3. 参数文件路径可以是相对于脚本运行目录的相对路径
4. 建议根据实际数据特点调整聚类算法和参数 