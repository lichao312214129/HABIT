"""
Domain-specific configurators.

V1 splits the previous monolithic ``ServiceConfigurator`` per business
domain so that ``habit.core.common`` does not pull every business
subpackage into its import surface.

Public surface:
    - :class:`BaseConfigurator`
    - :class:`HabitatConfigurator`
    - :class:`MLConfigurator`
    - :class:`PreprocessingConfigurator`
"""

from .base import BaseConfigurator
from .habitat import HabitatConfigurator
from .ml import MLConfigurator
from .preprocessing import PreprocessingConfigurator

__all__ = [
    'BaseConfigurator',
    'HabitatConfigurator',
    'MLConfigurator',
    'PreprocessingConfigurator',
]
