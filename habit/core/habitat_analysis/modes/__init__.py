"""
Modes module for habitat analysis.
"""

from .base_mode import BaseMode
from .training_mode import TrainingMode
from .testing_mode import TestingMode
from ..config_schemas import HabitatAnalysisConfig
import logging

def create_mode(
    config: HabitatAnalysisConfig,
    logger: logging.Logger,
) -> BaseMode:
    """
    Factory function to create appropriate mode based on configuration.
    
    Args:
        config: Habitat analysis configuration
        logger: Logger instance
        
    Returns:
        BaseMode: Either TrainingMode or TestingMode
        
    Raises:
        ValueError: If mode is invalid
    """
    if config.HabitatsSegmention.habitat.mode == 'training':
        return TrainingMode(config, logger)
    elif config.HabitatsSegmention.habitat.mode == 'testing':
        return TestingMode(config, logger)
    else:
        raise ValueError(f"Invalid mode: {config.HabitatsSegmention.habitat.mode}")
