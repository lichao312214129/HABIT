"""
聚类算法包
"""

# 导入所有聚类算法，确保它们被正确注册
from habitat_analysis.clustering.base_clustering import (
    BaseClustering,
    register_clustering,
    get_clustering_algorithm
)

# 导入实现的聚类算法
from habitat_analysis.clustering.kmeans_clustering import KMeansClustering
from habitat_analysis.clustering.gmm_clustering import GMMClustering

# 导入聚类验证方法映射模块
from habitat_analysis.clustering.cluster_validation_methods import (
    CLUSTERING_VALIDATION_METHODS,
    get_validation_methods,
    is_valid_method_for_algorithm,
    get_default_methods,
    get_all_clustering_algorithms,
    get_optimization_direction,
    get_method_description
)

# 用户可以在这里自定义聚类算法的导入并注册
# 比如: from habitat_analysis.clustering.custom_clustering import CustomClustering
