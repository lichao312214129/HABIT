"""
Clustering module for habitat analysis.
"""

from .base_clustering import (
    BaseClustering,
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
    "BaseClustering", "register_clustering", "get_clustering_algorithm",
    "KMeansClustering", "GMMClustering", "SLICClustering",
    "get_validation_methods", "get_default_methods", "is_valid_method_for_algorithm",
    "get_method_description", "get_optimization_direction"
]
