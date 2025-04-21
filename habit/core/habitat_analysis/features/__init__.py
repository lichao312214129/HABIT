"""
特征提取器包
"""

# 导入所有特征提取器，确保它们被正确注册
from habitat_analysis.features.base_feature_extractor import (
    BaseFeatureExtractor,
    register_feature_extractor,
    get_feature_extractor,
    get_available_feature_extractors
)

# 导入实现的特征提取器
from habitat_analysis.features.kinetic_feature_extractor import KineticFeatureExtractor
from habitat_analysis.features.simple_feature_extractor import SimpleFeatureExtractor

# 可以添加其他特征提取器的导入
from habitat_analysis.features.my_feature_extractor import MyFeatureExtractor




