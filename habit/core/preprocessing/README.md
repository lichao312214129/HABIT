# 图像预处理模块使用指南

本模块提供了多种医学图像预处理方法，以统一的接口进行调用。通过注册机制，可以轻松添加和使用各种预处理器。

## 基本用法

### 1. 在配置文件中指定预处理方法

预处理通常通过配置文件来指定，以下是一个示例配置文件：

```yaml
# 预处理配置
Preprocessing:
  n4_correction:
    images: [t1, t2, flair]
    num_fitting_levels: 4
    num_iterations: [50, 50, 50, 50]
    convergence_threshold: 0.001
    shrink_factor: 4
  
  resample:
    images: [t1, t2, flair]
    target_spacing: [1.0, 1.0, 1.0]  # x, y, z 目标体素间距
    
  zscore_normalization:
    images: [t1, t2, flair]
    mask_keys: [brain_mask]
    clip_values: [-3, 3]
    
  histogram_standardization:
    images: [t1, t2, flair]
    reference_path: ./reference_histograms.pkl
    mask_keys: [brain_mask]
    output_range: [0, 1]
```

### 2. 在代码中直接调用

```python
from habit.core.preprocessing import PreprocessorFactory, ZScoreNormalization

# 创建一个Z-Score标准化预处理器
zscore_processor = PreprocessorFactory.create(
    "zscore_normalization",
    keys=["t1", "t2"],
    mask_keys=["brain_mask"],
    clip_values=(-3, 3),
    allow_missing_keys=True
)

# 应用预处理
data = {
    "t1": sitk_image_t1,
    "t2": sitk_image_t2,
    "brain_mask": sitk_mask
}

processed_data = zscore_processor(data)
```

## 可用的预处理器

本模块提供以下预处理器：

### 1. 重采样预处理器 (`resample`)

将图像重采样到指定的体素间距。

**参数**:
- `keys`: 要处理的图像键
- `target_spacing`: 目标体素间距 [x, y, z]
- `mode`: 插值模式，默认为'linear'
- `allow_missing_keys`: 是否允许缺少键，默认为False

**示例**:
```python
resample_processor = PreprocessorFactory.create(
    "resample",
    keys=["t1", "t2"],
    target_spacing=[1.0, 1.0, 1.0],
    mode="linear"
)
```

### 2. N4偏置场校正预处理器 (`n4_correction`)

应用N4偏置场校正以修正医学图像的强度不均匀性。

**参数**:
- `keys`: 要处理的图像键
- `mask_keys`: 用于校正的掩码键（可选）
- `num_fitting_levels`: 偏置场校正的拟合级别数
- `num_iterations`: 每个拟合级别的迭代次数
- `convergence_threshold`: 收敛阈值
- `shrink_factor`: 加速计算的缩小因子
- `allow_missing_keys`: 是否允许缺少键，默认为False

**示例**:
```python
n4_processor = PreprocessorFactory.create(
    "n4_correction",
    keys=["t1", "t2"],
    mask_keys=["brain_mask"],
    num_fitting_levels=4,
    num_iterations=[50, 50, 50, 50]
)
```

### 3. 配准预处理器 (`registration`)

将多个图像配准到参考图像。

**参数**:
- `keys`: 要处理的图像键
- `fixed_image`: 参考图像键
- `moving_images`: 要配准的图像键列表
- `type_of_transform`: 变换类型，例如'Rigid'、'Affine'、'SyN'等
- `use_mask`: 是否使用掩码进行配准
- `allow_missing_keys`: 是否允许缺少键，默认为False

**示例**:
```python
registration_processor = PreprocessorFactory.create(
    "registration",
    keys=["t1", "t2", "flair"],
    fixed_image="t1",
    moving_images=["t2", "flair"],
    type_of_transform="SyN",
    use_mask=True
)
```

### 4. Z-Score标准化预处理器 (`zscore_normalization`)

通过减去均值并除以标准差对图像强度进行标准化，结果是均值为0、标准差为1的分布。

**参数**:
- `keys`: 要处理的图像键
- `mask_keys`: 用于计算统计信息的掩码键（可选）
- `clip_values`: 标准化后限制值的范围，例如(-3, 3)
- `allow_missing_keys`: 是否允许缺少键，默认为False

**示例**:
```python
zscore_processor = PreprocessorFactory.create(
    "zscore_normalization",
    keys=["t1", "t2"],
    mask_keys=["brain_mask"],
    clip_values=(-3, 3)
)
```

**配置文件示例**:
```yaml
Preprocessing:
  zscore_normalization:
    images: [t1, t2, flair]
    mask_keys: [brain_mask]
    clip_values: [-3, 3]
```

### 5. 直方图标准化预处理器 (`histogram_standardization`)

通过将输入图像的直方图映射到参考图像来标准化图像强度。这对于标准化来自不同扫描仪或采集协议的图像很有用。

**参数**:
- `keys`: 要处理的图像键
- `reference_key`: 数据字典中参考图像的键名
- `allow_missing_keys`: 是否允许缺少键，默认为False

**内部实现**：
- 使用SimpleITK的HistogramMatchingImageFilter
- 默认使用256个直方图级别
- 默认使用100个匹配点
- 默认在均值处进行阈值化

**示例**:
```python
# 使用数据字典中的参考图像
hist_matcher = PreprocessorFactory.create(
    "histogram_standardization",
    keys=["t1", "t2", "flair"],
    reference_key="t1_atlas"
)
```

**配置文件示例**:
```yaml
Preprocessing:
  histogram_standardization:
    images: [pre_contrast, LAP, delay_3min]
    reference_key: PVP
```

## 自定义预处理器

您可以通过以下步骤添加自定义预处理器：

1. 创建一个继承自`BasePreprocessor`的新类
2. 使用`@PreprocessorFactory.register`装饰器注册该类
3. 在`__init__.py`中导入您的模块

示例：

```python
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("my_custom_preprocessor")
class MyCustomPreprocessor(BasePreprocessor):
    def __init__(self, keys, my_param=1.0, allow_missing_keys=False, **kwargs):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.my_param = my_param
        
    def __call__(self, data):
        self._check_keys(data)
        
        # 实现您的预处理逻辑
        # ...
        
        return data
``` 