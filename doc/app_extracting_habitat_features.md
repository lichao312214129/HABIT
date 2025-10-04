# app_extracting_habitat_features.py 功能文档

## 功能概述

`app_extracting_habitat_features.py` 是HABIT工具包用于从已分割的生境图中提取特征的入口程序。这个脚本支持从医学图像中提取多种类型的特征，包括传统放射组学特征、非放射组学特征、整体生境特征、单个生境特征、空间交互矩阵(multiregional spatial interaction (MSI))特征以及肿瘤内异质性指数(Intratumoral Heterogeneity Index, IHI index)。

## 用法

```bash
python scripts/app_extracting_habitat_features.py --config <config_file_path>
```

## 配置文件格式

`app_extracting_habitat_features.py` 使用YAML格式的配置文件，包含以下主要部分：

```yaml
# 特征提取设置
params_file_of_non_habitat: <原始图像放射组学参数文件路径>
params_file_of_habitat: <生境图放射组学参数文件路径>
raw_img_folder: <原始图像目录路径>
habitats_map_folder: <生境图目录路径>
out_dir: <结果输出目录路径>

# 特征类型和处理设置
feature_types: [<特征类型列表>]
n_habitats: <生境数量，如果为空，则自动检测>
mode: <操作模式>

# 程序设置
n_processes: <并行进程数量>
habitat_pattern: <生境文件匹配模式>
debug: <是否启用调试模式>
```

## 支持的特征类型

### 1. traditional（传统特征）

从原始图像中提取的传统放射组学特征，这些特征是基于整个肿瘤区域计算的。

包括以下主要类别的特征：
- 一阶统计特征（First Order Statistics）
- 形状特征（Shape Features）
- 灰度共生矩阵特征（GLCM）
- 灰度游程矩阵特征（GLRLM）
- 灰度大小区域矩阵特征（GLSZM）
- 灰度依赖矩阵特征（GLDM）
- 相邻灰度色调差异矩阵特征（NGTDM）

### 2. non_radiomics（非放射组学特征）

从生境分割图中提取的非放射组学特征（基本统计特征），包括：

- 生境数量（num_habitats）
- 各生境体积比例（label_X.volume_ratio）
- 各生境连通区域数量（label_X.num_regions）

### 3. whole_habitat（整体生境特征）

从整个生境分割图中提取的放射组学特征，将生境图像作为一个整体处理。与传统放射组学特征类别相同，但是基于生境图而非原始图像计算。

### 4. each_habitat（单个生境特征）

分别从每个生境中提取的放射组学特征，为每个生境单独计算特征。每个生境都会有完整的放射组学特征集，命名格式为"label_X_特征名"。

### 5. msi（多区域空间交互矩阵特征）

计算不同生境之间的空间关系特征，包括：
- 生境间边界数量（firstorder_i_and_j）
- 生境内邻接关系数量（firstorder_i_and_i）
- 归一化的空间交互特征
- MSI矩阵的纹理特征（对比度、同质性、相关性、能量等）

### 6. ith_score（肿瘤内异质性指数）

计算肿瘤内异质性得分（ITH Score），范围为0-1，值越高表示肿瘤内异质性越大。

## 执行流程

1. 解析命令行参数或加载配置文件
2. 验证配置参数有效性
3. 创建HabitatFeatureExtractor实例
4. 根据指定的操作模式和特征类型执行特征提取：
   - 如果mode为'extract'或'both'，从图像中提取特征
   - 如果mode为'parse'或'both'，解析特征并生成摘要
5. 保存提取的特征到输出目录

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `features_{timestamp}/` 目录，包含：
   - 针对不同特征类型的CSV文件和表格
   - 提取的特征数据及统计信息
   - 如果在mode中包含'parse'，会生成特征摘要和分析报告

## 特征输出详细说明

以下是各类特征输出CSV文件中包含的特征及其含义：

### 传统放射组学特征（traditional）
   

### 非放射组学特征（non_radiomics）
- **num_habitats**：生境的总数量。
- **label_X.num_regions**：标签为X的生境的连通区域数量。
- **label_X.volume_ratio**：标签为X的生境占总生境体积的比例。

### 生境整体特征（whole_habitat）
基于整个生境分割图提取的放射组学特征，与传统放射组学特征类别相同，但是针对生境图像而非原始图像。

### 单个生境特征（each_habitat）
对每个生境区域单独提取的放射组学特征。每个生境都会有完整的放射组学特征集，命名格式为"label_X_特征名"。

### 多区域空间交互矩阵（MSI）特征
- **firstorder_i_and_j**：生境i和生境j之间边界的数量，表示两个生境之间的空间关系强度。
- **firstorder_i_and_i**：生境i内部的邻接关系数量，表示生境的内部连接度。
- **firstorder_normalized_i_and_j**：归一化后的生境i和生境j之间的空间关系。
- **firstorder_normalized_i_and_i**：归一化后的生境i内部连接度，表示生境内部的空间一致性。该值越高，表明该生境在空间上越连续完整。
- **contrast**：MSI矩阵的对比度，表示不同生境之间的差异程度。
- **homogeneity**：MSI矩阵的同质性，表示生境空间分布的均匀程度。
- **correlation**：MSI矩阵的相关性，表示生境空间分布的相关程度。
- **energy**：MSI矩阵的能量，表示生境空间分布的规则性和简单性。

### 肿瘤内异质性指数（ITH）特征
- **ith_score**：肿瘤内异质性得分，范围为0-1，值越高表示肿瘤内异质性越大。

## 完整配置示例

```yaml
# 特征提取设置
params_file_of_non_habitat: ./config/parameter.yaml
params_file_of_habitat: ./config/parameter_habitat.yaml
raw_img_folder: /data/processed_images
habitats_map_folder: /data/habitats_output
out_dir: /data/features_output

# 特征类型和处理设置
feature_types: 
  - traditional
  - non_radiomics
  - whole_habitat
  - each_habitat
n_habitats: 5
mode: both

# 程序设置
n_processes: 4
habitat_pattern: '*_habitats.nrrd'
debug: false
```

## 特征提取工作流

1. **图像准备**：
   - 读取原始医学图像和生境分割图
   - 根据需要进行预处理和标准化

2. **特征计算**：
   - 对于traditional特征：使用PyRadiomics从原始图像中提取放射组学特征
   - 对于non_radiomics特征：计算生境分割图的统计特性
   - 对于whole_habitat特征：将整个生境图作为输入提取特征
   - 对于each_habitat特征：为每个生境单独提取特征
   - 对于msi特征：在多个空间尺度上提取特征

3. **特征整合**：
   - 组合所有提取的特征
   - 创建特征矩阵和标签

4. **特征分析**（如果mode包含'parse'）：
   - 计算特征统计信息
   - 生成特征摘要
   - 可视化重要特征

## 注意事项

1. 确保原始图像和生境图的目录结构正确，生境图文件名需要与原始图像匹配
2. PyRadiomics参数文件应根据您的图像特性进行适当配置
3. 提取复杂特征（如MSI特征、each_habitat特征）可能需要更多计算资源和时间
4. 如果n_habitats设置为空（null或0），程序将自动检测生境数量
5. 确保所有路径在配置中正确指定，尤其是在不同操作系统间切换时
6. 生境文件匹配模式（habitat_pattern）默认为'*_habitats.nrrd'，请根据实际文件命名调整
7. 特征类型可以单独选择或组合使用，建议根据研究目的选择需要的特征类型 