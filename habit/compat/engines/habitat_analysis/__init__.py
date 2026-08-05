# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Habitat Analysis module for HABIT package.

V1: imports are fail-fast. If a habitat-analysis dependency is missing the
package will raise ``ImportError`` at first import, not silently set
attributes to ``None``. For genuinely optional dependencies (e.g. optional
extractor backends), prefer ``habit.is_available(name)``.

Public exports are lazy so lightweight imports such as
``habit.compat.engines.habitat_analysis.config_schemas`` do not pull PyRadiomics or
pipeline orchestration until the symbol is actually used.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HabitatAnalysis": (".habitat_analysis", "HabitatAnalysis"),
    "HabitatConfigurator": (".configurator", "HabitatConfigurator"),
    "HabitatAnalysisConfig": (".config_schemas", "HabitatAnalysisConfig"),
    "ResultColumns": (".config_schemas", "ResultColumns"),
    "HabitatMapAnalyzer": (
        "..habitat_extraction.habitat_features.habitat_analyzer",
        "HabitatMapAnalyzer",
    ),
    "BasePipelineStep": (".pipelines", "BasePipelineStep"),
    "HabitatPipeline": (".pipelines", "HabitatPipeline"),
    "GroupClusteringStep": (".pipelines.steps", "GroupClusteringStep"),
    "GroupPreprocessingStep": (".pipelines.steps", "GroupPreprocessingStep"),
}

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


def __getattr__(name: str) -> Any:
    """Resolve habitat-analysis exports on first access."""
    if name == "HabitatFeatureExtractor":
        return lazy_getattr("HabitatMapAnalyzer", globals(), _LAZY_EXPORTS)
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
