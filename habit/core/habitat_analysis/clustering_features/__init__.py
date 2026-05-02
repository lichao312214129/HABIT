"""
Feature extractors module for habitat analysis.

Import policy (V1, aligned with the package-level fail-fast rule):

* Hard dependencies (numpy / pandas / SimpleITK / scipy / six) are imported
  directly. If any of these is missing the user gets the original
  ``ImportError`` straight away — silently degrading to ``None`` placeholders
  hides real configuration bugs and shows up much later as confusing
  ``AttributeError: 'NoneType' object has no attribute ...`` traces.
* Truly optional integrations (currently the ``radiomics`` Python package)
  are guarded with ``try / except ImportError``. The module is still
  importable without ``radiomics`` installed, but the corresponding
  extractor classes will be absent from the registry — calling code can
  query :func:`get_feature_extractors` to detect this.

If you add a new extractor, decide which bucket it belongs in and follow
the same pattern; do NOT add a generic ``try / except ImportError`` "just
to be safe" — that defeats the point of fail-fast imports.
"""

import logging
from typing import Dict, Optional, Type

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# ---------------------------------------------------------------------------
# Always-available (hard dependency) extractors
# ---------------------------------------------------------------------------

from .base_extractor import (
    BaseClusteringExtractor,
    register_feature_extractor,
)

# Backward-compatible alias (downstream code referenced this name pre-V1).
BaseFeatureExtractor = BaseClusteringExtractor

from .kinetic_feature_extractor import KineticFeatureExtractor
from .raw_feature_extractor import RawFeatureExtractor
from .concat_feature_extractor import ConcatImageFeatureExtractor
from .mean_voxel_features_extractor import (
    MeanVoxelFeaturesExtractor,
    calculate_supervoxel_means,
)
from .local_entropy_extractor import LocalEntropyExtractor
from .my_feature_extractor import MyFeatureExtractor

_available_extractors: Dict[str, Type] = {
    'KineticFeatureExtractor': KineticFeatureExtractor,
    'RawFeatureExtractor': RawFeatureExtractor,
    'ConcatImageFeatureExtractor': ConcatImageFeatureExtractor,
    'MeanVoxelFeaturesExtractor': MeanVoxelFeaturesExtractor,
    'LocalEntropyExtractor': LocalEntropyExtractor,
    'MyFeatureExtractor': MyFeatureExtractor,
}

# ---------------------------------------------------------------------------
# Optional radiomics-backed extractors
# ---------------------------------------------------------------------------
# These need the third-party ``radiomics`` package; without it the package
# still imports cleanly so non-radiomics workflows keep working.

try:
    from .voxel_radiomics_extractor import VoxelRadiomicsExtractor
except ImportError as exc:  # pragma: no cover - depends on optional dep
    logger.info(
        "VoxelRadiomicsExtractor unavailable (radiomics package not installed): %s",
        exc,
    )
    VoxelRadiomicsExtractor = None  # type: ignore[assignment]
else:
    _available_extractors['VoxelRadiomicsExtractor'] = VoxelRadiomicsExtractor

try:
    from .supervoxel_radiomics_extractor import SupervoxelRadiomicsExtractor
except ImportError as exc:  # pragma: no cover - depends on optional dep
    logger.info(
        "SupervoxelRadiomicsExtractor unavailable (radiomics package not installed): %s",
        exc,
    )
    SupervoxelRadiomicsExtractor = None  # type: ignore[assignment]
else:
    _available_extractors['SupervoxelRadiomicsExtractor'] = SupervoxelRadiomicsExtractor


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def get_feature_extractors() -> Dict[str, Type]:
    """
    Get all available feature extractors.

    Returns:
        Dict[str, Type]: Dictionary mapping extractor names to their classes.
    """
    return _available_extractors.copy()


def get_feature_extractor(name: str) -> Optional[Type]:
    """
    Get a specific feature extractor by name.

    Args:
        name: Name of the feature extractor.

    Returns:
        The feature extractor class, or ``None`` when not registered (e.g.
        a radiomics extractor was requested but radiomics is not installed).
    """
    return _available_extractors.get(name)


__all__ = [
    "BaseClusteringExtractor",
    "BaseFeatureExtractor",
    "register_feature_extractor",
    "get_feature_extractors",
    "get_feature_extractor",
    "calculate_supervoxel_means",
    "KineticFeatureExtractor",
    "RawFeatureExtractor",
    "ConcatImageFeatureExtractor",
    "MeanVoxelFeaturesExtractor",
    "LocalEntropyExtractor",
    "MyFeatureExtractor",
]
if VoxelRadiomicsExtractor is not None:
    __all__.append("VoxelRadiomicsExtractor")
if SupervoxelRadiomicsExtractor is not None:
    __all__.append("SupervoxelRadiomicsExtractor")

logger.info(
    "Loaded %d feature extractors: %s",
    len(_available_extractors),
    sorted(_available_extractors.keys()),
)
