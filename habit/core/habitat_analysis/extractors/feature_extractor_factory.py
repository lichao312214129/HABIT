"""
Feature extractor factory for creating and configuring feature extractors
"""

import importlib
from typing import Dict, Any, List, Type
from .base_extractor import (
    BaseClusteringExtractor,
    get_feature_extractor,
    get_available_feature_extractors
)

class FeatureExtractorConfig:
    """Configuration class for feature extractors"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize feature extractor configuration
        
        Args:
            name: Name of the feature extractor
            params: Parameters for the feature extractor
        """
        self.name = name
        self.params = params or {}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureExtractorConfig':
        """
        Create feature extractor configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary containing name and params fields
        
        Returns:
            FeatureExtractorConfig: Feature extractor configuration instance
        """
        if not isinstance(config_dict, dict):
            raise ValueError(f"Configuration must be a dictionary, got: {type(config_dict)}")
        
        name = config_dict.get('method')  # Use 'method' instead of 'name' to match YAML config
        if not name:
            raise ValueError("Configuration must contain 'method' field")
        
        # Copy all parameters except 'method'
        params = {k: v for k, v in config_dict.items() if k != 'method'}
        
        return cls(name=name, params=params)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'method': self.name,  # Use 'method' as the key name
            **self.params  # Expand all parameters to top level
        }
    
    def create_extractor(self) -> BaseClusteringExtractor:
        """
        Create feature extractor instance
        
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        return get_feature_extractor(self.name, **self.params)


class FeatureExtractorFactory:
    """Feature extractor factory class"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseClusteringExtractor:
        """
        Create feature extractor from configuration
        
        Args:
            config: Configuration dictionary in format:
                {
                    "method": "feature extractor name",
                    "timestamps": "timestamp file path",
                    "other_params": value,
                    ...
                }
                
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        feature_config = FeatureExtractorConfig.from_dict(config)
        return feature_config.create_extractor()
    
    @staticmethod
    def create_from_name(name: str, **kwargs) -> BaseClusteringExtractor:
        """
        Create feature extractor from name
        
        Args:
            name: Feature extractor name
            **kwargs: Feature extractor parameters
            
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        return get_feature_extractor(name, **kwargs)
    
    @staticmethod
    def get_available_extractors() -> List[str]:
        """
        Get all available feature extractor names
        
        Returns:
            List[str]: List of feature extractor names
        """
        return get_available_feature_extractors()


def create_feature_extractor(config_or_name, **kwargs) -> BaseClusteringExtractor:
    """
    Convenience function for creating feature extractors
    
    Args:
        config_or_name: Feature extractor configuration or name
            - If string, treated as feature extractor name or function expression
            - If dictionary, treated as feature extractor configuration
        **kwargs: Parameters passed to feature extractor
        
    Returns:
        BaseFeatureExtractor: Feature extractor instance
    """
    if isinstance(config_or_name, dict):
        # 处理字典配置
        config_dict = config_or_name.copy()
        method = config_dict.pop('method', None)
        if not method:
            raise ValueError("Feature extractor configuration must contain 'method' field")
        
        # 合并参数
        params = {}
        if 'params' in config_dict:
            params.update(config_dict.pop('params'))
        params.update(config_dict)
        params.update(kwargs)
        
        # 处理方法（可能是函数式语法）
        return _process_method_spec(method, **params)
    elif isinstance(config_or_name, str):
        # 直接处理方法名或函数式语法
        return _process_method_spec(config_or_name, **kwargs)
    else:
        raise ValueError(f"Invalid feature extractor configuration or name: {config_or_name}")

def _process_method_spec(method_spec, **kwargs):
    """处理方法规范（简单名称或函数式语法）"""
    # 检查是否是函数式语法
    if '(' in method_spec and ')' in method_spec:
        return _parse_functional_method(method_spec, **kwargs)
    
    # 简单方法名称
    return FeatureExtractorFactory.create_from_name(method_spec, **kwargs)

def _parse_functional_method(method_expr, **kwargs):
    """解析函数式方法表达式"""
    import re
    
    # 提取主方法名称 (例如: 从 "kinetic(raw(img1),...)" 提取 "kinetic")
    match = re.match(r'(\w+)\(', method_expr)
    if not match:
        raise ValueError(f"Invalid method expression: {method_expr}")
    
    main_method = match.group(1)
    
    # 提取参数部分
    params_str = method_expr[len(main_method)+1:].rstrip(')')
    
    # 解析参数并添加到kwargs
    parsed_kwargs = kwargs.copy()
    parsed_kwargs.update(_parse_parameters(params_str, **kwargs))
    
    # 创建特征提取器
    return FeatureExtractorFactory.create_from_name(main_method, **parsed_kwargs)

def _parse_parameters(params_str, **context_kwargs):
    """
    解析参数字符串
    
    Args:
        params_str: 参数字符串
        **context_kwargs: 上下文参数，可能包含timestamps等
        
    Returns:
        dict: 解析后的参数字典
    """
    import re
    
    params = {}
    
    # 处理raw()函数，它只是标记原始图像，不需要特殊处理
    # 我们只需要提取图像名称列表
    raw_pattern = r'raw\(([^)]+)\)'
    raw_images = []
    
    # 替换所有raw()函数为它们的参数值
    cleaned_params = re.sub(raw_pattern, r'\1', params_str)
    
    # 分割参数
    param_parts = [p.strip() for p in cleaned_params.split(',')]
    
    # 前面的参数是图像名称
    image_names = []
    for part in param_parts:
        if part and part != 'timestamps':
            image_names.append(part)
    
    if image_names:
        params['image_names'] = image_names
    
    # 如果参数中包含timestamps，使用上下文中的timestamps
    if 'timestamps' in params_str and 'timestamps' in context_kwargs:
        params['timestamps'] = context_kwargs['timestamps']
    
    return params
