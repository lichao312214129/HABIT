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
Define the mapping relationship between clustering methods and optimal cluster number determination methods
"""

import numpy as np
from sklearn.metrics import silhouette_score as _sk_silhouette_score

from habit.kernels.cluster_selection import score_direction as _kernel_score_direction

# Validation methods supported by each clustering algorithm.
#
# The 'optimization' entries below are documentation only: the authoritative
# selection rule per score lives in habit.kernels.cluster_selection so the
# v0.1 and v1.0 paths cannot disagree on a habitat count. This table's job is
# to say which scores each ALGORITHM supports and which are its defaults.
CLUSTERING_VALIDATION_METHODS = {
    # Methods supported by K-Means clustering
    'kmeans': {
        'default': ['elbow'],  # Community-used inertia elbow; k in [2, 10]
        'methods': {
            'silhouette': {
                'description': 'Silhouette Score, higher is better',
                'optimization': 'maximize'
            },
            'calinski_harabasz': {
                'description': 'Calinski-Harabasz Index, higher is better',
                'optimization': 'maximize'
            },
            'inertia': {
                'description': 'Inertia curve, selected by Kneedle elbow detection',
                'optimization': 'knee'
            },
            'kneedle': {
                'description': 'Kneedle elbow detection on the inertia curve',
                'optimization': 'knee'
            },
            'elbow': {
                # v1.0 breaking change: alias of 'kneedle' (was second-derivative).
                'description': 'Inertia curve, selected by Kneedle elbow detection',
                'optimization': 'knee'
            },
            'gap': {
                'description': 'Gap Statistic, higher is better',
                'optimization': 'maximize'
            },
            'davies_bouldin': {
                'description': 'Davies-Bouldin Index, lower is better',
                'optimization': 'minimize'
            }
        }
    },
    
    # Methods supported by Gaussian Mixture Model (GMM)
    'gmm': {
        'default': ['aic'],  # Default methods
        'methods': {
            'bic': {
                'description': 'Bayesian Information Criterion, lower is better',
                'optimization': 'minimize'
            },
            'aic': {
                'description': 'Akaike Information Criterion, lower is better',
                'optimization': 'minimize'
            },
            'silhouette': {
                'description': 'Silhouette Score, higher is better',
                'optimization': 'maximize'
            },
            'calinski_harabasz': {
                'description': 'Calinski-Harabasz Index, higher is better',
                'optimization': 'maximize'
            }
        }
    },
    
    # Methods supported by SLIC clustering
    'slic': {
        'default': ['silhouette', 'kneedle'],
        'methods': {
            'silhouette': {
                'description': 'Silhouette Score, higher is better',
                'optimization': 'maximize'
            },
            'calinski_harabasz': {
                'description': 'Calinski-Harabasz Index, higher is better',
                'optimization': 'maximize'
            },
            'davies_bouldin': {
                'description': 'Davies-Bouldin Index, lower is better',
                'optimization': 'minimize'
            },
            'inertia': {
                'description': 'Inertia curve on SLIC-regularized embedding, selected by Kneedle',
                'optimization': 'knee'
            },
            'kneedle': {
                'description': 'Kneedle elbow detection on the SLIC-regularized inertia curve',
                'optimization': 'knee'
            },
            'elbow': {
                # v1.0 breaking change: alias of 'kneedle' (was second-derivative).
                'description': 'SLIC-regularized inertia curve, selected by Kneedle elbow detection',
                'optimization': 'knee'
            }
        }
    }
}

def get_validation_methods(clustering_algorithm: str) -> dict:
    """
    Get the validation methods supported by the specified clustering algorithm
    
    Args:
        clustering_algorithm: Name of the clustering algorithm
        
    Returns:
        dict: Dictionary of validation methods supported by the clustering algorithm
    """
    if clustering_algorithm.lower() in CLUSTERING_VALIDATION_METHODS:
        return CLUSTERING_VALIDATION_METHODS[clustering_algorithm.lower()]
    else:
        # If the specified clustering algorithm is not found, return generic validation methods
        return {
            'default': ['silhouette'],
            'methods': {
                'silhouette': {
                    'description': 'Silhouette Score, higher is better',
                    'optimization': 'maximize'
                }
            }
        }

def is_valid_method_for_algorithm(clustering_algorithm: str, validation_method: str) -> bool:
    """
    Check if the given validation method is applicable to the specified clustering algorithm
    
    Args:
        clustering_algorithm: Name of the clustering algorithm
        validation_method: Name of the validation method
        
    Returns:
        bool: Whether the method is applicable
    """
    validation_info = get_validation_methods(clustering_algorithm)
    return validation_method in validation_info['methods']

def get_default_methods(clustering_algorithm: str) -> list:
    """
    Get the default validation methods for the specified clustering algorithm
    
    Args:
        clustering_algorithm: Name of the clustering algorithm
        
    Returns:
        list: List of default validation methods
    """
    validation_info = get_validation_methods(clustering_algorithm)
    return validation_info['default']

def get_all_clustering_algorithms() -> list:
    """
    Get all supported clustering algorithms
    
    Returns:
        list: List of clustering algorithms
    """
    return list(CLUSTERING_VALIDATION_METHODS.keys())

def get_optimization_direction(clustering_algorithm: str, validation_method: str) -> str:
    """
    Get the selection rule for a validation method.

    Delegates to :func:`habit.kernels.cluster_selection.score_direction`, the
    single source shared with the v1.0 domain fitters. ``clustering_algorithm``
    is accepted for backward compatibility but does not affect the rule: the
    rule is a property of the score, not of the algorithm.

    Args:
        clustering_algorithm: Name of the clustering algorithm (unused).
        validation_method: Name of the validation method.

    Returns:
        str: ``'maximize'``, ``'minimize'`` or ``'knee'``.
    """
    return _kernel_score_direction(validation_method)

def get_method_description(clustering_algorithm: str, validation_method: str) -> str:
    """
    获取验证方法的描述信息
    
    Args:
        clustering_algorithm: 聚类算法名称
        validation_method: 验证方法名称
        
    Returns:
        str: 验证方法描述
    """
    validation_info = get_validation_methods(clustering_algorithm)
    if validation_method in validation_info['methods']:
        return validation_info['methods'][validation_method]['description']
    else:
        # 默认描述
        return f"{validation_method.capitalize()} Score"


def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """
    Public wrapper around sklearn's silhouette_score for validation and tests.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        labels: Cluster label per sample, shape (n_samples,).
        **kwargs: Forwarded to sklearn.metrics.silhouette_score (e.g. metric).

    Returns:
        Scalar silhouette score in [-1, 1] when well-defined.
    """
    return float(_sk_silhouette_score(X, labels, **kwargs))