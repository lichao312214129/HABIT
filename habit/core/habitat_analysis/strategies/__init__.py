"""
Strategy factory and exports for habitat analysis.
"""

from typing import Dict, Type

from .base_strategy import BaseClusteringStrategy
from .one_step_strategy import OneStepStrategy
from .two_step_strategy import TwoStepStrategy
from .direct_pooling_strategy import DirectPoolingStrategy

# Alias for backward compatibility
BaseHabitatStrategy = BaseClusteringStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseClusteringStrategy]] = {
    "one_step": OneStepStrategy,
    "two_step": TwoStepStrategy,
    "direct_pooling": DirectPoolingStrategy,
}


def get_strategy(strategy_name: str) -> Type[BaseClusteringStrategy]:
    """
    Get strategy class by name.

    Args:
        strategy_name: Strategy identifier from config

    Returns:
        Strategy class
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown clustering strategy: {strategy_name}. "
            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[strategy_name]
