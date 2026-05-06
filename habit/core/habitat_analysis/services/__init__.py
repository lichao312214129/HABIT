"""
Service layer for Habitat Analysis.

Each service is the orchestration surface that pipeline steps call into:

- :class:`FeatureService`   : feature extraction and per-subject preprocessing.
- :class:`ClusteringService`: clustering algorithms, validation, and visualisation.
- :class:`HabitatImageWriter`     : persistence of NRRD habitat/supervoxel maps.
- :class:`HabitatResultPublisher` : CSV column order, train/predict result publishing,
  optional delegation to :class:`HabitatImageWriter` for label images.
"""
from .feature_service import FeatureService
from .clustering_service import ClusteringService
from .habitat_image_writer import HabitatImageWriter
from .result_publisher import HabitatResultPublisher, canonical_csv_column_order

__all__ = [
    "FeatureService",
    "ClusteringService",
    "HabitatImageWriter",
    "HabitatResultPublisher",
    "canonical_csv_column_order",
]
