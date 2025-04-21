# 特征提取

特征提取是在生成生境地图后的关键步骤，它可以量化各个生境的特征，为后续分析提供数据基础。本节将详细介绍如何使用HABIT提取生境特征。

## 工作原理

HABIT支持从生境地图中提取多种类型的特征：

1. **传统放射组学特征**：从原始图像的不同生境区域提取组学特征
2. **非放射组学特征**：包括生境体积百分比、断开区域数量等
3. **整体生境特征**：将整个生境地图作为一个整体进行分析
4. **各生境特征**：针对每个生境单独提取特征
5. **多尺度图像 (MSI) 特征**：分析生境之间的空间关系特征

## 数据准备

特征提取需要两类输入数据：

1. **原始医学图像**：通常与生成生境地图时使用的图像相同
2. **生境地图**：从前一步骤生成的生境地图（*.nrrd文件）

数据结构示例：

```
dataset/
├── raw_images/            # 原始图像目录
│   ├── images/            # 图像子目录
│   │   ├── patient001/    # 患者001
│   │   │   ├── T1/        # T1序列
│   │   │   │   └── img.nii.gz  # 图像文件
│   │   │   ├── ...
│   ├── masks/             # 分割掩码子目录
│   │   ├── patient001/    # 患者001
│   │   │   ├── ...
├── habitats_maps/         # 生境地图目录
│   ├── patient001_habitats.nrrd  # 患者001的生境地图
│   ├── patient002_habitats.nrrd  # 患者002的生境地图
│   ├── ...
│   └── habitats.csv       # 生境信息表
```

## 配置文件

特征提取可以通过命令行参数或配置文件进行设置。以下是一个典型的配置示例：

```yaml
# 文件路径参数
params_file_of_non_habitat: parameter.yaml  # 原始图像特征提取参数文件
params_file_of_habitat: parameter_habitat.yaml  # 生境图特征提取参数文件
raw_img_folder: /path/to/raw_images  # 原始图像目录
habitats_map_folder: /path/to/habitats_maps  # 生境地图目录
out_dir: /path/to/output  # 输出目录

# 处理参数
n_processes: 4  # 并行处理进程数
habitat_pattern: '*_habitats.nrrd'  # 生境文件匹配模式
  # 可选: '*_habitats.nrrd', '*_habitats_remapped.nrrd'

# 特征提取参数
feature_types:  # 要提取的特征类型列表
  - traditional  # 传统放射组学特征
  - non_radiomics  # 非放射组学特征
  - whole_habitat  # 整体生境特征
  - each_habitat  # 各生境特征
  - msi  # 多尺度图像特征

n_habitats: 0  # 生境数量 (0表示自动检测)
mode: both  # 操作模式: extract, parse, both
```

## 命令行使用

使用以下命令提取特征：

```bash
python scripts/extract_features.py --config config/feature_extraction.yaml
```

或者直接使用命令行参数：

```bash
python scripts/extract_features.py \
  --params_file_of_non_habitat parameter.yaml \
  --params_file_of_habitat parameter_habitat.yaml \
  --raw_img_folder /path/to/raw_images \
  --habitats_map_folder /path/to/habitats_maps \
  --out_dir /path/to/output \
  --feature_types traditional non_radiomics whole_habitat each_habitat msi
```

## 输出结果

特征提取结果将保存在指定的输出目录中，主要包括以下文件：

1. **feature_<timestamp>.npy**：包含所有提取特征的二进制文件
2. **parsed_features/**：包含解析后的特征CSV文件
   - **traditional_features.csv**：传统放射组学特征
   - **non_radiomics_features.csv**：非放射组学特征
   - **whole_habitat_features.csv**：整体生境特征
   - **each_habitat_features.csv**：各生境特征
   - **msi_features.csv**：多尺度图像特征

## 特征类型详解

### 传统放射组学特征

从原始图像的各个生境区域提取的标准放射组学特征，包括：

- 一阶特征（直方图特征）
- 形状特征
- 纹理特征（GLCM、GLRLM、GLSZM等）

### 非放射组学特征

描述生境形态学和分布特征，如：

- 各生境所占体积百分比
- 生境区域连通性（断开区域数量）
- 生境边界特征

### 整体生境特征

将生境地图作为一个整体进行分析，提取：

- 生境地图的纹理特征
- 生境分布模式特征
- 生境整体形状特征

### 各生境特征

针对每个独立的生境区域提取特征，可以分析：

- 各生境的形状特征
- 各生境在肿瘤中的相对位置
- 各生境的边界特性

### 多尺度图像 (MSI) 特征

分析生境之间的空间关系，包括：

- 生境邻接关系
- 生境过渡区特性
- 生境空间分布模式

## 高级选项

### 自定义参数文件

HABIT使用PyRadiomics作为特征提取引擎，支持自定义参数文件来控制特征提取过程。参数文件示例：

```yaml
# 图像类型设置
imageType:
  Original: {}  # 使用原始图像
  LoG: 
    sigma: [2.0, 3.0, 4.0]  # 不同尺度的高斯拉普拉斯滤波

# 特征类别设置
featureClass:
  shape: []
  firstorder: []
  glcm: []
  glrlm: []
  glszm: []
  ngtdm: []
  gldm: []

# 常规设置
setting:
  normalize: true
  normalizeScale: 100
  interpolator: 'sitkBSpline'
  resampledPixelSpacing: [1, 1, 1]
```

### 操作模式

HABIT支持三种操作模式：

- **extract**：仅提取特征，生成.npy文件
- **parse**：仅解析已提取特征，生成CSV文件
- **both**：提取并解析特征（默认行为）

## 注意事项

1. 特征提取是一个计算密集型操作，特别是处理大量数据时
2. 根据计算机配置适当调整并行进程数
3. 对于大型数据集，建议分批处理
4. 特征文件可能很大，确保有足够的磁盘空间 