"""
测试聚类验证方法映射模块
"""

import os
import sys

from habit.core.habitat_analysis.clustering.cluster_validation_methods import (
    CLUSTERING_VALIDATION_METHODS,
    get_validation_methods,
    is_valid_method_for_algorithm,
    get_default_methods,
    get_all_clustering_algorithms,
    get_optimization_direction,
    get_method_description
)

def test_get_validation_methods():
    """测试获取聚类算法的验证方法"""
    print("测试获取聚类算法的验证方法...")
    
    # 测试已知的聚类算法
    kmeans_methods = get_validation_methods('kmeans')
    print(f"K-Means聚类支持的验证方法: {kmeans_methods['default']}")
    
    gmm_methods = get_validation_methods('gmm')
    print(f"GMM聚类支持的验证方法: {gmm_methods['default']}")
    
    # 测试未知的聚类算法
    unknown_methods = get_validation_methods('unknown_algorithm')
    print(f"未知聚类算法支持的验证方法: {unknown_methods['default']}")
    
    print("测试完成\n")

def test_is_valid_method():
    """测试验证方法是否适用于聚类算法"""
    print("测试验证方法是否适用于聚类算法...")
    
    # 测试有效的组合
    valid_kmeans = is_valid_method_for_algorithm('kmeans', 'inertia')
    print(f"K-Means聚类使用'inertia'方法是否有效: {valid_kmeans}")
    
    valid_gmm = is_valid_method_for_algorithm('gmm', 'bic')
    print(f"GMM聚类使用'bic'方法是否有效: {valid_gmm}")
    
    # 测试无效的组合
    invalid_kmeans = is_valid_method_for_algorithm('kmeans', 'bic')
    print(f"K-Means聚类使用'bic'方法是否有效: {invalid_kmeans}")
    
    invalid_gmm = is_valid_method_for_algorithm('gmm', 'inertia')
    print(f"GMM聚类使用'inertia'方法是否有效: {invalid_gmm}")
    
    print("测试完成\n")

def test_get_default_methods():
    """测试获取聚类算法的默认验证方法"""
    print("测试获取聚类算法的默认验证方法...")
    
    # 测试已知的聚类算法
    kmeans_default = get_default_methods('kmeans')
    print(f"K-Means聚类的默认验证方法: {kmeans_default}")
    
    gmm_default = get_default_methods('gmm')
    print(f"GMM聚类的默认验证方法: {gmm_default}")
    
    # 测试未知的聚类算法
    unknown_default = get_default_methods('unknown_algorithm')
    print(f"未知聚类算法的默认验证方法: {unknown_default}")
    
    print("测试完成\n")

def test_get_all_algorithms():
    """测试获取所有支持的聚类算法"""
    print("测试获取所有支持的聚类算法...")
    
    all_algos = get_all_clustering_algorithms()
    print(f"所有支持的聚类算法: {all_algos}")
    
    print("测试完成\n")

def test_get_optimization_direction():
    """测试获取验证方法的优化方向"""
    print("测试获取验证方法的优化方向...")
    
    # 测试已知的组合
    kmeans_inertia = get_optimization_direction('kmeans', 'inertia')
    print(f"K-Means聚类使用'inertia'方法的优化方向: {kmeans_inertia}")
    
    gmm_bic = get_optimization_direction('gmm', 'bic')
    print(f"GMM聚类使用'bic'方法的优化方向: {gmm_bic}")
    
    # 测试未知的组合
    unknown_method = get_optimization_direction('kmeans', 'unknown_method')
    print(f"K-Means聚类使用未知方法的优化方向: {unknown_method}")
    
    print("测试完成\n")

def test_get_method_description():
    """测试获取验证方法的描述"""
    print("测试获取验证方法的描述...")
    
    # 测试已知的组合
    kmeans_inertia_desc = get_method_description('kmeans', 'inertia')
    print(f"K-Means聚类使用'inertia'方法的描述: {kmeans_inertia_desc}")
    
    gmm_bic_desc = get_method_description('gmm', 'bic')
    print(f"GMM聚类使用'bic'方法的描述: {gmm_bic_desc}")
    
    # 测试未知的组合
    unknown_method_desc = get_method_description('kmeans', 'unknown_method')
    print(f"K-Means聚类使用未知方法的描述: {unknown_method_desc}")
    
    print("测试完成\n")

def main():
    """主测试函数"""
    print("开始测试聚类验证方法映射模块...\n")
    
    test_get_validation_methods()
    test_is_valid_method()
    test_get_default_methods()
    test_get_all_algorithms()
    test_get_optimization_direction()
    test_get_method_description()
    
    print("所有测试完成!")

if __name__ == "__main__":
    main() 