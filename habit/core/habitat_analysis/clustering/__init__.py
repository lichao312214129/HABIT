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
Clustering module for habitat analysis.
"""

from .base_clustering import (
    BaseClustering,
    ClusteringAlgorithmFactory,
    get_clustering_algorithm,
    get_available_clustering_algorithms
)

# Import implemented clustering algorithms
from .kmeans_clustering import KMeansClustering
from .gmm_clustering import GMMClustering
from .slic_clustering import SLICClustering

# Import validation methods
from .cluster_validation_methods import (
    get_validation_methods,
    get_default_methods,
    is_valid_method_for_algorithm,
    get_method_description,
    get_optimization_direction
)

# Add custom clustering algorithm imports here, e.g.:
#     from .custom_clustering import CustomClustering

__all__ = [
    "BaseClustering", "ClusteringAlgorithmFactory",
    "get_clustering_algorithm",
    "KMeansClustering", "GMMClustering", "SLICClustering",
    "get_validation_methods", "get_default_methods", "is_valid_method_for_algorithm",
    "get_method_description", "get_optimization_direction"
]
