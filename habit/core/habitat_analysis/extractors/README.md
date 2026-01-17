# 特征提取器模块

本模块提供了栖息地分析中使用的特征提取器实现。

## 内置特征提取器

当前已实现以下特征提取器：

- `kinetic`: 动态特征提取器，提取基于时间序列图像的动态特征
- `simple`: 简单特征提取器，直接使用图像强度作为特征

## 如何自定义新的特征提取器

您可以通过以下步骤定义自己的特征提取器：

1. 在`features`目录下创建一个新的Python文件，命名为`your_method_feature_extractor.py`
2. 从`BaseFeatureExtractor`类继承一个新类，并使用`register_feature_extractor`装饰器注册它
3. 实现所有必需的方法：`extract_features`和属性`feature_names`

### 示例

```python
from habit.core.habitat_analysis.clustering_features.base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
import numpy as np

@register_feature_extractor('your_method')
class YourMethodFeatureExtractor(BaseFeatureExtractor):
    # 如果需要时间戳，请设置为True
    requires_timestamp = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化代码
        
    def extract_features(self, image_data, **kwargs):
        # 实现特征提取逻辑
        # 例如：提取纹理特征、形状特征等
        
        # 设置特征名称
        self.feature_names = ["feature1", "feature2", "feature3"]
        
        # 返回提取的特征
        return features
        
    # get_feature_names方法在基类中已实现，如果需要，可以重写它
```

### 自动发现与注册

您不需要修改`__init__.py`文件！只要您的文件命名符合规范（`*_feature_extractor.py`），系统会在运行时自动发现并注册您的特征提取器。

### 使用自定义特征提取器

一旦注册，您可以在配置文件中指定您的特征提取器：

```yaml
FeatureConstruction:
  method: your_method  # 这里使用您注册的特征提取器名称
  # 其他参数
```

或者在代码中直接使用：

```python
from habit.core.habitat_analysis.clustering_features.feature_extractor_factory import create_feature_extractor

# 创建您的特征提取器实例
feature_extractor = create_feature_extractor('your_method', param1=value1)
``` 