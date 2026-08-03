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
