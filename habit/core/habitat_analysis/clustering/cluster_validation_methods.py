"""
Define the mapping relationship between clustering methods and optimal cluster number determination methods
"""

# Validation methods supported by each clustering algorithm
CLUSTERING_VALIDATION_METHODS = {
    # Methods supported by K-Means clustering
    'kmeans': {
        'default': ['silhouette', 'calinski_harabasz', 'inertia'],  # Default methods
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
                'description': 'Inertia, sum of squared distances to the nearest cluster center, lower is better',
                'optimization': 'minimize'
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
    
    # Methods supported by Spectral Clustering
    'spectral': {
        'default': ['silhouette', 'calinski_harabasz'],
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
            }
        }
    },
    
    # Methods supported by Hierarchical Clustering
    'hierarchical': {
        'default': ['silhouette', 'calinski_harabasz'],
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
            'cophenetic': {
                'description': 'Cophenetic Correlation Coefficient, higher is better',
                'optimization': 'maximize'
            }
        }
    },
    
    # Methods supported by DBSCAN density-based clustering
    'dbscan': {
        'default': ['silhouette', 'calinski_harabasz'],
        'methods': {
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
    
    # Methods supported by Mean Shift Clustering
    'mean_shift': {
        'default': ['silhouette', 'calinski_harabasz'],
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
            }
        }
    },
    
    # Methods supported by Affinity Propagation
    'affinity_propagation': {
        'default': ['silhouette', 'calinski_harabasz'],
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
    Get the optimization direction (maximize or minimize) for the validation method
    
    Args:
        clustering_algorithm: Name of the clustering algorithm
        validation_method: Name of the validation method
        
    Returns:
        str: 'maximize' or 'minimize'
    """
    validation_info = get_validation_methods(clustering_algorithm)
    if validation_method in validation_info['methods']:
        return validation_info['methods'][validation_method]['optimization']
    else:
        # Default to maximize
        return 'maximize'

def get_method_description(clustering_algorithm: str, validation_method: str) -> str:
    """
    Get the description of the validation method
    
    Args:
        clustering_algorithm: Name of the clustering algorithm
        validation_method: Name of the validation method
        
    Returns:
        str: Method description
    """
    validation_info = get_validation_methods(clustering_algorithm)
    if validation_method in validation_info['methods']:
        return validation_info['methods'][validation_method]['description']
    else:
        return "Unknown validation method"