# Habitat Feature Extraction Module

## 目录结构

重构后的代码组织结构如下：

```
feature_extraction/
├── __init__.py                      # 包初始化文件
├── README.md                        # 本文档
├── extractor.py                     # 原始的特征提取器实现（保留）
├── new_extractor.py                 # 重构后的特征提取器实现
├── basic_features.py                # 基本特征提取功能
├── habitat_radiomics.py             # 生境组学特征提取功能
├── msi_features.py                  # MSI特征提取功能
├── ith_features.py                  # ITH分数计算功能
├── feature_utils.py                 # 辅助工具函数
├── get_msi_features.py              # 原始MSI特征提取实现（保留）
├── traditional_radiomics_extractor.py # 原始组学特征提取实现（保留）
└── utils.py                         # 原始工具函数（保留）
```

## 重构说明

为了提高代码的可读性、可维护性和可测试性，我们将原始的`extractor.py`文件进行了模块化重构，拆分成了多个功能明确的模块：

1. **basic_features.py**: 包含基本生境特征提取功能，如不连通区域数量和体积占比。
2. **habitat_radiomics.py**: 包含生境相关的组学特征提取功能，如从原始影像中提取生境组学特征。
3. **msi_features.py**: 包含MSI特征相关的提取功能。
4. **ith_features.py**: 包含ITH（肿瘤内部异质性）分数计算功能，基于组学特征聚类分析。
5. **feature_utils.py**: 包含一些辅助功能，如字典扁平化、从CSV文件读取生境数量等。
6. **new_extractor.py**: 重构后的主类，整合了上述所有模块的功能。

注意：我们保留了原始的`extractor.py`文件，新的实现在`new_extractor.py`中。

## 主要特征说明

### ITH分数计算

ITH（Intratumoral Heterogeneity，肿瘤内部异质性）评分是量化肿瘤内部异质性的重要指标。计算步骤包括：

1. 扩展肿瘤区域3mm，包括肿瘤周边微环境
2. 设置3×3mm滑动窗口，提取局部组学特征（包括一阶特征和纹理特征）
3. 使用k-means聚类识别具有相似组学特征分布的子区域，聚类数使用肘部法确定
4. 计算ITH评分，基于公式：ITH score = 1 - (1/S_total) * Σ(S_i,max / n_i)
   - 其中n_i是每个聚类的连接区域数
   - S_i,max是每个聚类的最大连接区域面积
   - S_total是总面积

## 使用方法

### 使用新的实现

```python
from habitat_analysis.feature_extraction.new_extractor import HabitatFeatureExtractor

# 创建特征提取器实例
extractor = HabitatFeatureExtractor(
    params_file_of_non_habitat='path/to/parameter.yaml',
    params_file_of_habitat='path/to/parameter_habitat.yaml',
    raw_img_folder='path/to/raw_images',
    habitats_map_folder='path/to/habitat_maps',
    out_dir='path/to/output',
    n_processes=4,
    habitat_pattern='*_habitats.nrrd',
    voxel_cutoff=10,
    extract_ith=True  # 启用ITH分数计算
)

# 运行特征提取和解析
extractor.run(
    feature_types=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith'],
    n_habitats=4,  # 可选，如果未指定则从habitats.csv读取
    mode='both'    # 'extract', 'parse', 或 'both'
)
```

### 命令行使用

```bash
python new_extractor.py --params_file_of_non_habitat parameter.yaml --params_file_of_habitat parameter_habitat.yaml --raw_img_folder G:\raw_images --habitats_map_folder F:\habitat_maps --out_dir F:\output --n_processes 10 --habitat_pattern *_habitats.nrrd --extract_ith --feature_types traditional non_radiomics whole_habitat each_habitat msi ith --mode both
```

## 扩展

各个模块都设计为可独立使用，例如：

### 单独使用ITH分数计算

```python
from habitat_analysis.feature_extraction.ith_features import ITHFeatureExtractor

# 创建ITH特征提取器
ith_extractor = ITHFeatureExtractor(
    params_file='path/to/parameter.yaml',
    window_size=3,
    margin_size=3,
    voxel_cutoff=10
)

# 提取ITH特征
ith_features = ith_extractor.extract_ith_features(
    image_path='path/to/image.nrrd',
    mask_path='path/to/mask.nrrd',
    out_dir='path/to/visualization_output'  # 可选，用于保存可视化结果
)

# 获取ITH分数
ith_score = ith_features['ith_score']
print(f"ITH Score: {ith_score}")
```

### 单独使用MSI特征提取

```python
from habitat_analysis.feature_extraction.msi_features import MSIFeatureExtractor

msi_extractor = MSIFeatureExtractor()
features = msi_extractor.extract_MSI_features('path/to/habitat.nrrd', n_habitats=4, subj='subject_id')
``` 