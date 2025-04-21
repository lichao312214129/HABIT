"""
Visualization utilities for habitat analysis
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_elbow_curve(cluster_range, scores, score_type, title=None, save_path=None):
    """
    绘制肘部曲线
    
    Args:
        cluster_range: 聚类数量范围
        scores: 对应的评分
        score_type: 评分类型，用于标题和y轴标签
        title: 图表标题，如果为None，则自动生成
        save_path: 保存路径，如果为None，则不保存
    """
    if title is None:
        title = f"The {score_type} Method showing the optimal k"
    
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(score_type)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_multiple_scores(cluster_range, scores_dict, title=None, save_path=None):
    """
    在同一张图上绘制多种评分方法的结果
    
    Args:
        cluster_range: 聚类数量范围
        scores_dict: 字典，键为评分方法名称，值为对应的评分列表
        title: 图表标题，如果为None，则自动生成
        save_path: 保存路径，如果为None，则不保存
    """
    if title is None:
        title = "Comparison of different cluster evaluation metrics"
    
    plt.figure(figsize=(12, 8))
    
    for i, (score_name, scores) in enumerate(scores_dict.items()):
        # 标准化评分，使它们在同一范围内
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        # 如果是BIC或AIC，则反转，使越小越好变为越大越好，便于比较
        if score_name.lower() in ['bic', 'aic', 'inertia']:
            normalized_scores = 1 - normalized_scores
        
        plt.plot(cluster_range, normalized_scores, 'o-', label=score_name)
    
    plt.xlabel('Number of clusters')
    plt.ylabel('Normalized score (higher is better)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_cluster_results(X, labels, centers=None, title=None, feature_names=None, save_path=None):
    """
    绘制聚类结果的散点图
    
    Args:
        X: 输入数据，形状为(n_samples, n_features)
        labels: 聚类标签，形状为(n_samples,)
        centers: 聚类中心，如果不为None，则绘制
        title: 图表标题
        feature_names: 特征名称，用于x和y轴标签
        save_path: 保存路径，如果为None，则不保存
    """
    # 如果特征数大于2，则使用PCA降维
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        if centers is not None:
            centers_pca = pca.transform(centers)
    else:
        X_pca = X
        centers_pca = centers
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    
    if centers_pca is not None:
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='Cluster centers')
    
    if title:
        plt.title(title)
    else:
        plt.title('Cluster Results')
    
    if feature_names and len(feature_names) >= 2:
        if X.shape[1] > 2:
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
        else:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
    else:
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 