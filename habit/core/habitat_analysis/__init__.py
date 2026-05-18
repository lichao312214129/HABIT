"""
Habitat Analysis module for HABIT package.

V1: imports are fail-fast. If a habitat-analysis dependency is missing the
package will raise ``ImportError`` at first import, not silently set
attributes to ``None``. For genuinely optional dependencies (e.g. optional
extractor backends), prefer ``habit.is_available(name)``.

Public surface:
    - :class:`HabitatAnalysis`         : main analysis entry point.
    - :class:`HabitatConfigurator`     : domain service assembly entry point.
    - :class:`HabitatAnalysisConfig`   : Pydantic root configuration.
    - :class:`ResultColumns`           : reserved DataFrame column names.
    - :class:`HabitatMapAnalyzer`      : post-clustering feature analyser
                                         (alias :class:`HabitatFeatureExtractor`).
    - :class:`HabitatPipeline` /
      :class:`BasePipelineStep`        : sklearn-style pipeline primitives.
    - :class:`GroupPreprocessingStep` /
      :class:`GroupClusteringStep`: re-exported pipeline steps.
"""

from .habitat_analysis import HabitatAnalysis
from .configurator import HabitatConfigurator
from .config_schemas import HabitatAnalysisConfig, ResultColumns
from .habitat_features.habitat_analyzer import HabitatMapAnalyzer
from .pipelines import BasePipelineStep, HabitatPipeline
from .pipelines.steps import (
    GroupClusteringStep,
    GroupPreprocessingStep,
)

HabitatFeatureExtractor = HabitatMapAnalyzer

__all__ = [
    "HabitatAnalysis",
    "HabitatConfigurator",
    "HabitatAnalysisConfig",
    "ResultColumns",
    "HabitatMapAnalyzer",
    "HabitatFeatureExtractor",
    "BasePipelineStep",
    "HabitatPipeline",
    "GroupClusteringStep",
    "GroupPreprocessingStep",
]
