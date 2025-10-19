# 图像预处理模块使用文档

## 功能概述

图像预处理模块实现了一系列医学图像预处理步骤，包括重采样、配准和N4偏置场校正等，为后续的生境分析提供标准化和优化的图像数据。

## 🚀 快速开始

### 使用CLI（推荐）✨

```bash
# 使用默认配置
habit preprocess

# 使用指定配置文件
habit preprocess --config config/config_image_preprocessing.yaml

# 简写形式
habit preprocess -c config/config_image_preprocessing.yaml
```

### 使用传统脚本（兼容旧版）

```bash
# 使用指定配置文件
python scripts/app_image_preprocessing.py --config ./config/config_image_preprocessing.yaml 

# 简写形式
python scripts/app_image_preprocessing.py -c ./config/config_image_preprocessing.yaml 

# 如果不指定配置文件，脚本将使用默认配置文件
python scripts/app_image_preprocessing.py
```

## 配置文件格式

`app_image_preprocessing.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据路径
data_dir: <原始数据目录路径>
out_dir: <输出目录路径>

# 预处理设置
Preprocessing:
  # 预处理步骤配置
```

### 预处理步骤配置

```yaml
Preprocessing:
  # 重采样配置
  resample:
    images: [<图像1>, <图像2>, ...]  # 要重采样的图像列表
    target_spacing: [x, y, z]  # 目标体素间距
    img_mode: <插值模式>  # 可选，默认为linear
    padding_mode: <填充模式>  # 可选，默认为border
    align_corners: <是否对齐角点>  # 可选，默认为false

  # 配准配置
  registration:
    images: [<图像1>, <图像2>, ...]  # 要配准的图像列表
    fixed_image: <参考图像>  # 静态参考图像
    moving_images: [<图像1>, <图像2>, ...]  # 要移动的图像列表
    type_of_transform: <变换类型>  # 例如Rigid, Affine, SyNRA等
    use_mask: <是否使用掩码>  # 可选，默认为false

  # N4偏置场校正配置
  n4_correction:
    images: [<图像1>, <图像2>, ...]  # 要校正的图像列表
    num_fitting_levels: <拟合层级数>  # 可选，默认为4
    num_iterations: [<迭代次数1>, <迭代次数2>, ...]  # 可选，默认为每层50次
    convergence_threshold: <收敛阈值>  # 可选，默认为0.001
```

## 支持的预处理步骤

### 1. 重采样（Resampling）

调整图像的体素间距到指定的目标分辨率。

#### 参数

| 参数 | 类型 | 描述 | 默认值 |
|-----|-----|-----|-----|
| images | List[str] | 要重采样的图像列表 | 必需 |
| target_spacing | List[float] | 目标体素间距（毫米） | 必需 |
| img_mode | str | 图像插值模式 | "linear" |
| padding_mode | str | 边界外值填充模式 | "border" |
| align_corners | bool | 是否对齐角点 | False |

#### 支持的插值模式

- `nearest`: 最近邻插值
- `linear`: 线性插值
- `bilinear`: 双线性插值（等同于linear）
- `bspline`: B样条插值
- `bicubic`: 双三次插值（等同于bspline）
- `gaussian`: 高斯插值
- `lanczos`: Lanczos窗修正的sinc插值
- `hamming`: Hamming窗修正的sinc插值
- `cosine`: 余弦窗修正的sinc插值
- `welch`: Welch窗修正的sinc插值
- `blackman`: Blackman窗修正的sinc插值

### 2. 配准（Registration）

将图像对齐到参考图像，使用ANTs（高级标准化工具）实现。

#### 参数

| 参数 | 类型 | 描述 | 默认值 |
|-----|-----|-----|-----|
| images | List[str] | 要处理的图像列表 | 必需 |
| fixed_image | str | 参考图像的关键字 | 必需 |
| moving_images | List[str] | 要移动的图像列表 | 必需 |
| type_of_transform | str | 变换类型 | "Rigid" |
| metric | str | 相似度度量 | "MI" |
| optimizer | str | 优化方法 | "gradient_descent" |
| use_mask | bool | 是否使用掩码 | False |

#### 支持的变换类型

1. **Rigid**: 仅平移和旋转
2. **Affine**: 平移、旋转、缩放和剪切
3. **SyN**: 对称归一化（可变形配准）
4. **SyNRA**: SyN + Rigid + Affine（最常用）
5. **SyNOnly**: 不带初始刚性/仿射的SyN
6. **TRSAA**: 平移 + 旋转 + 缩放 + 仿射
7. **Elastic**: 弹性变换
8. **SyNCC**: 带交叉相关度量的SyN
9. **SyNabp**: 带互信息度量的SyN
10. **SyNBold**: 针对BOLD图像优化的SyN
11. **SyNBoldAff**: SyN + Affine for BOLD图像
12. **SyNAggro**: 使用激进优化的SyN
13. **TVMSQ**: 时变微分同胚与均方度量

### 3. N4偏置场校正

校正医学图像中的强度不均匀性，通常由MRI扫描仪中的磁场不均匀性引起。

#### 参数

| 参数 | 类型 | 描述 | 默认值 |
|-----|-----|-----|-----|
| images | List[str] | 要校正的图像列表 | 必需 |
| num_fitting_levels | int | 拟合层级数 | 4 |
| num_iterations | List[int] | 每层迭代次数 | [50] * num_fitting_levels |
| convergence_threshold | float | 收敛阈值 | 0.001 |

### 4. 直方图标准化

将图像的直方图与参考图像的直方图匹配，使得不同图像之间的强度分布更加一致。这对于跨扫描仪或跨采集序列的图像标准化非常有用。

#### 参数

| 参数 | 类型 | 描述 | 默认值 |
|-----|-----|-----|-----|
| images | List[str] | 要标准化的图像列表 | 必需 |
| reference_key | str | 参考图像的键名 | 必需 |

#### 配置示例

```yaml
histogram_standardization:
  images: [pre_contrast, LAP, delay_3min]  # 要处理的图像序列
  reference_key: PVP  # 参考图像的键名
```

## 完整配置示例

```yaml
# 数据路径
data_dir: F:\work\research\radiomics_TLSs\data\raw_data
out_dir: F:\work\research\radiomics_TLSs\data\results

# 预处理设置
Preprocessing:
  # 重采样（可选）
  resample:
    images: [T2WI, ADC]
    target_spacing: [1.0, 1.0, 1.0]

  # 配准（可选）
  registration:
    images: [T2WI, ADC]
    fixed_image: T2WI
    moving_images: [ADC]
    type_of_transform: SyNRA  # 支持所有ANTs配准方法
    use_mask: false

  # N4偏置场校正（可选）
  # n4_correction:
  #   images: [T2WI, ADC]
  #   num_fitting_levels: 2
    
  # Z-Score标准化（可选）
  # zscore_normalization:
  #   images: [T2WI, ADC]
  #   only_inmask: false
    
  # 直方图标准化（可选）
  # histogram_standardization:
  #   images: [T2WI]
  #   reference_key: ADC

# 一般设置
processes: 1  # 并行进程数
random_state: 42  # 随机种子
```

## 执行流程

1. 加载配置文件
2. 初始化BatchProcessor处理器
3. 执行批处理操作，处理所有指定的图像
4. 生成详细的处理日志

## 输出结构

```
out_dir/
├── processed_images/
│   ├── images/
│   │   └── subject_id/
│   │       ├── modality1/
│   │       │   └── modality1.nii.gz
│   │       └── modality2/
│   │           └── modality2.nii.gz
│   └── masks/
│       └── subject_id/
│           ├── modality1/
│           │   └── mask_modality1.nii.gz
│           └── modality2/
│               └── mask_modality2.nii.gz
└── logs/
    └── processing.log
```

## 日志记录

该脚本维护详细的日志文件 `preprocessing_debug.log`，包含：
- 处理进度
- 错误消息
- 参数设置
- 性能指标

## 最佳实践建议

1. **重采样**:
   - 根据分析需求选择合适的目标间距
   - 对图像使用线性插值
   - 对掩码使用最近邻插值

2. **配准**:
   - 对于可变形配准使用SyN
   - 对于多模态配准使用MI度量
   - 在有掩码的情况下始终使用掩码
   - 考虑使用SyNRA获得更好的初始对齐

3. **N4校正**:
   - 大多数情况下使用4个拟合层级
   - 根据图像复杂度调整迭代次数
   - 如有掩码可用，使用掩码获得更好的校正

## 常见问题及解决方案

1. **配准失败**:
   - 尝试不同的变换类型
   - 调整相似度度量
   - 如有可用掩码，启用use_mask
   - 检查图像方向

2. **N4校正问题**:
   - 增加迭代次数
   - 调整收敛阈值
   - 使用掩码获得更好的校正

3. **内存问题**:
   - 减少并行进程数
   - 分批处理图像
   - 对初始配准使用较低分辨率

## 性能考虑

1. **并行处理**:
   - 使用多进程加快处理速度
   - 根据可用内存调整进程数
   - 对内存密集型操作考虑使用较少进程

2. **内存使用**:
   - 监控处理过程中的内存使用情况
   - 必要时调整批量大小
   - 使用适当的图像格式

3. **磁盘空间**:
   - 处理后的图像以NIfTI格式保存
   - 考虑压缩以便长期存储
   - 定期清理临时文件 