# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
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
