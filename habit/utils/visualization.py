"""
Visualization utilities for habitat analysis
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .font_config import setup_publication_font, get_font_config

# Setup publication-quality Arial font
setup_publication_font()


def plot_cluster_scores(scores_dict: Dict[str, List[float]], 
                      cluster_range: List[int], 
                      methods: Optional[Union[List[str], str]] = None,
                      clustering_algorithm: str = 'kmeans',
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None,
                      show: bool = True,
                      dpi: int = 300):
    """
    Plot the scoring curves for cluster evaluation
    
    Args:
        scores_dict: Dictionary of scores, with method names as keys and score lists as values
        cluster_range: Range of cluster numbers to evaluate
        methods: Methods to plot, can be a string or list of strings, None means plot all methods
        clustering_algorithm: Name of the clustering algorithm
        figsize: Size of the figure
        save_path: Path to save the figure, None means do not save
        show: Whether to display the figure
        dpi: Image resolution
    """
    from habit.core.habitat_analysis.clustering.cluster_validation_methods import get_method_description, get_optimization_direction
    
    # If methods is None, use all methods in scores_dict
    if methods is None:
        methods = list(scores_dict.keys())
    elif isinstance(methods, str):
        methods = [methods]
    
    # Create figure
    n_methods = len(methods)
    if n_methods == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]  # Convert to list for consistent access
    else:
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods > 1 and not isinstance(axes, np.ndarray):
            axes = [axes]
    
    # Plot for each method
    for i, method in enumerate(methods):
        if method not in scores_dict:
            continue
        
        ax = axes[i] if i < len(axes) else axes[-1]
        scores = scores_dict[method]
        
        # Plot score curve
        ax.plot(cluster_range, scores, 'o-', linewidth=2, markersize=8)

        # Get optimization direction for the method
        optimization = get_optimization_direction(clustering_algorithm, method)
        
        # Mark the optimal number of clusters
        if optimization == 'maximize':
            # For methods where higher is better
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            criterion = "Maximum"
        elif optimization == 'minimize':
            # For methods where lower is better, use elbow method
            # Calculate first-order differences
            deltas = np.diff(scores)
            # Calculate second-order differences (inflection point typically has max second-order difference)
            deltas2 = np.diff(deltas)
            # Inflection point is the point after the max second-order difference
            best_idx = np.argmax(deltas2) + 1
            if best_idx >= len(scores) - 1:
                best_idx = len(scores) - 2
            best_score = scores[best_idx]
            criterion = "Elbow Method"
        else:
            # Default to maximum
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            criterion = "Maximum"
        
        # Get the optimal number of clusters
        best_n_clusters = cluster_range[best_idx]
        
        # Mark the optimal point on the plot
        ax.plot(best_n_clusters, best_score, 'rx', markersize=12, markeredgewidth=3)
        
        # Set title and labels
        method_desc = get_method_description(clustering_algorithm, method)
        ax.set_title(f"{method_desc}\nOptimal Clusters = {best_n_clusters} ({criterion})", fontfamily='Arial')
        ax.set_xlabel("Number of Clusters", fontfamily='Arial')
        ax.set_ylabel(f"{method.capitalize()} Score", fontfamily='Arial')
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close()


def plot_elbow_curve(cluster_range, scores, score_type, title=None, save_path=None):
    """
    Plot the elbow curve
    
    Args:
        cluster_range: Range of cluster numbers
        scores: Corresponding scores
        score_type: Type of score for title and y-axis label
        title: Figure title, automatically generated if None
        save_path: Path to save the figure, do not save if None
    """
    if title is None:
        title = f"The {score_type} Method showing the optimal k"
    
    plt.figure(figsize=(6, 4))
    plt.plot(cluster_range, scores, 'bx-')
    plt.xlabel('Number of clusters', fontfamily='Arial')
    plt.ylabel(score_type, fontfamily='Arial')
    plt.title(title, fontfamily='Arial')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_multiple_scores(cluster_range, scores_dict, title=None, save_path=None):
    """
    Plot multiple scoring methods on the same graph
    
    Args:
        cluster_range: Range of cluster numbers
        scores_dict: Dictionary with scoring method names as keys and score lists as values
        title: Figure title, automatically generated if None
        save_path: Path to save the figure, do not save if None
    """
    if title is None:
        title = "Comparison of different cluster evaluation metrics"
    
    plt.figure(figsize=(8, 6))
    
    for i, (score_name, scores) in enumerate(scores_dict.items()):
        # Normalize scores to be in the same range
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        # Invert BIC/AIC to make lower is better become higher is better
        if score_name.lower() in ['bic', 'aic', 'inertia']:
            normalized_scores = 1 - normalized_scores
        
        plt.plot(cluster_range, normalized_scores, 'o-', label=score_name)
    
    plt.xlabel('Number of clusters', fontfamily='Arial')
    plt.ylabel('Normalized score (higher is better)', fontfamily='Arial')
    plt.title(title, fontfamily='Arial')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_cluster_results(X, labels, centers=None, title=None, feature_names=None, save_path=None):
    """
    Plot scatter plot of clustering results
    
    Args:
        X: Input data, shape (n_samples, n_features)
        labels: Cluster labels, shape (n_samples,)
        centers: Cluster centers, plotted if not None
        title: Figure title
        feature_names: Feature names for x and y axis labels
        save_path: Path to save the figure, do not save if None
    """
    # Use PCA for dimensionality reduction if features > 2
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        if centers is not None:
            centers_pca = pca.transform(centers)
    else:
        X_pca = X
        centers_pca = centers
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    
    if centers_pca is not None:
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='Cluster centers')
    
    if title:
        plt.title(title, fontfamily='Arial')
    else:
        plt.title('Cluster Results', fontfamily='Arial')
    
    if feature_names and len(feature_names) >= 2:
        if X.shape[1] > 2:
            plt.xlabel('PCA Component 1', fontfamily='Arial')
            plt.ylabel('PCA Component 2', fontfamily='Arial')
        else:
            plt.xlabel(feature_names[0], fontfamily='Arial')
            plt.ylabel(feature_names[1], fontfamily='Arial')
    else:
        plt.xlabel('Feature 1', fontfamily='Arial')
        plt.ylabel('Feature 2', fontfamily='Arial')
    
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 