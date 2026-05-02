"""
Service layer for Habitat Analysis.

Each service is the orchestration surface that pipeline steps call into:

- :class:`FeatureService`   : feature extraction and per-subject preprocessing.
- :class:`ClusteringService`: clustering algorithms, validation, and visualisation.
- :class:`ResultWriter`     : persistence of results (CSV, NRRD habitat maps).
"""
from .feature_service import FeatureService
from .clustering_service import ClusteringService
from .result_writer import ResultWriter

__all__ = ["FeatureService", "ClusteringService", "ResultWriter"]
