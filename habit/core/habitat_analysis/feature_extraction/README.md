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
4. **feature_utils.py**: 包含一些辅助功能，如字典扁平化、从CSV文件读取生境数量等。
5. **new_extractor.py**: 重构后的主类，整合了上述所有模块的功能。

注意：我们保留了原始的`extractor.py`文件，新的实现在`new_extractor.py`中。

## 使用方法

### 使用新的实现

```python
from habitat_analysis.feature_extraction.new_extractor import NewHabitatFeatureExtractor

# 创建特征提取器实例
extractor = NewHabitatFeatureExtractor(
    params_file_of_non_habitat='path/to/parameter.yaml',
    params_file_of_habitat='path/to/parameter_habitat.yaml',
    raw_img_folder='path/to/raw_images',
    habitats_map_folder='path/to/habitat_maps',
    out_dir='path/to/output',
    n_processes=4,
    habitat_pattern='*_habitats.nrrd',
    voxel_cutoff=10
)

# 运行特征提取和解析
extractor.run(
    feature_types=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi'],
    n_habitats=4,  # 可选，如果未指定则从habitats.csv读取
    mode='both'    # 'extract', 'parse', 或 'both'
)
```

### 命令行使用

```bash
python new_extractor.py --params_file_of_non_habitat parameter.yaml --params_file_of_habitat parameter_habitat.yaml --raw_img_folder G:\raw_images --habitats_map_folder F:\habitat_maps --out_dir F:\output --n_processes 10 --habitat_pattern *_habitats.nrrd --feature_types traditional non_radiomics whole_habitat each_habitat msi --mode both
```

## 扩展

各个模块都设计为可独立使用，例如，如果只需要提取MSI特征：

```python
from habitat_analysis.feature_extraction.msi_features import MSIFeatureExtractor

msi_extractor = MSIFeatureExtractor()
features = msi_extractor.extract_MSI_features('path/to/habitat.nrrd', n_habitats=4, subj='subject_id')
``` 