"""
Feature extractors module for habitat analysis.
"""

import logging
from typing import Dict, Type, Optional
from habit.utils.log_utils import get_module_logger

# Set up logging for import errors
logger = get_module_logger(__name__)

# Base feature extractor - this should always be available
try:
    from .base_extractor import (
        BaseClusteringExtractor,
        register_feature_extractor
    )
    # Alias for backward compatibility
    BaseFeatureExtractor = BaseClusteringExtractor
except ImportError as e:
    logger.warning(f"Failed to import base_extractor: {e}")
    BaseClusteringExtractor = None
    BaseFeatureExtractor = None
    register_feature_extractor = None

# Import feature extractors with error handling
_available_extractors = {}

try:
    from .kinetic_feature_extractor import KineticFeatureExtractor
    _available_extractors['KineticFeatureExtractor'] = KineticFeatureExtractor
except ImportError as e:
    logger.warning(f"Failed to import KineticFeatureExtractor: {e}")
    KineticFeatureExtractor = None

try:
    from .raw_feature_extractor import RawFeatureExtractor
    _available_extractors['RawFeatureExtractor'] = RawFeatureExtractor
except ImportError as e:
    logger.warning(f"Failed to import RawFeatureExtractor: {e}")
    RawFeatureExtractor = None

try:
    from .voxel_radiomics_extractor import VoxelRadiomicsExtractor
    _available_extractors['VoxelRadiomicsExtractor'] = VoxelRadiomicsExtractor
except ImportError as e:
    logger.warning(f"Failed to import VoxelRadiomicsExtractor: {e}")
    VoxelRadiomicsExtractor = None

try:
    from .supervoxel_radiomics_extractor import SupervoxelRadiomicsExtractor
    _available_extractors['SupervoxelRadiomicsExtractor'] = SupervoxelRadiomicsExtractor
except ImportError as e:
    logger.warning(f"Failed to import SupervoxelRadiomicsExtractor: {e}")
    SupervoxelRadiomicsExtractor = None

try:
    from .concat_feature_extractor import ConcatImageFeatureExtractor
    _available_extractors['ConcatImageFeatureExtractor'] = ConcatImageFeatureExtractor
except ImportError as e:
    logger.warning(f"Failed to import ConcatImageFeatureExtractor: {e}")
    ConcatImageFeatureExtractor = None

try:
    from .mean_voxel_features_extractor import MeanVoxelFeaturesExtractor
    _available_extractors['MeanVoxelFeaturesExtractor'] = MeanVoxelFeaturesExtractor
except ImportError as e:
    logger.warning(f"Failed to import MeanVoxelFeaturesExtractor: {e}")
    MeanVoxelFeaturesExtractor = None

try:
    from .local_entropy_extractor import LocalEntropyExtractor
    _available_extractors['LocalEntropyExtractor'] = LocalEntropyExtractor
except ImportError as e:
    logger.warning(f"Failed to import LocalEntropyExtractor: {e}")
    LocalEntropyExtractor = None

# Custom feature extractors
try:
    from .my_feature_extractor import MyFeatureExtractor
    _available_extractors['MyFeatureExtractor'] = MyFeatureExtractor
except ImportError as e:
    logger.warning(f"Failed to import MyFeatureExtractor: {e}")
    MyFeatureExtractor = None

# Import feature preprocessing module - MOVED TO UTILS
# try:
#     from .feature_preprocessing import preprocess_features
# except ImportError as e:
#     logger.warning(f"Failed to import feature_preprocessing: {e}")
#     preprocess_features = None

def get_feature_extractors() -> Dict[str, Type]:
    """
    Get all available feature extractors.
    
    Returns:
        Dict[str, Type]: Dictionary mapping extractor names to their classes
    """
    return _available_extractors.copy()

def get_feature_extractor(name: str) -> Optional[Type]:
    """
    Get a specific feature extractor by name.
    
    Args:
        name (str): Name of the feature extractor
        
    Returns:
        Optional[Type]: The feature extractor class if available, None otherwise
    """
    return _available_extractors.get(name)

# Define __all__ with only available modules
__all__ = [
    "BaseClusteringExtractor", "BaseFeatureExtractor", "register_feature_extractor", "get_feature_extractors",
    "get_feature_extractor"
]

# Add available extractors to __all__
for extractor_name in _available_extractors.keys():
    __all__.append(extractor_name)

# Log summary of available extractors
if _available_extractors:
    logger.info(f"Successfully imported {len(_available_extractors)} feature extractors: {list(_available_extractors.keys())}")
else:
    logger.warning("No feature extractors were successfully imported")
