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
Clustering module for habitat analysis.
"""

from .base_clustering import (
    BaseClustering,
    ClusteringAlgorithmFactory,
    register_clustering,
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
    "BaseClustering", "ClusteringAlgorithmFactory", "register_clustering",
    "get_clustering_algorithm",
    "KMeansClustering", "GMMClustering", "SLICClustering",
    "get_validation_methods", "get_default_methods", "is_valid_method_for_algorithm",
    "get_method_description", "get_optimization_direction"
]
