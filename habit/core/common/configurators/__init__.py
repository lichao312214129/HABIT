"""
Shared configurator infrastructure.

``BaseConfigurator`` lives in ``habit.core.common.configurators.base`` as shared assembly
infrastructure. Domain-specific configurators live with their domains:

* ``habit.core.habitat_analysis.configurator.HabitatConfigurator``
* ``habit.core.machine_learning.configurator.MLConfigurator``
* ``habit.core.preprocessing.configurator.PreprocessingConfigurator``
"""

from .base import BaseConfigurator

__all__ = [
    'BaseConfigurator',
]
