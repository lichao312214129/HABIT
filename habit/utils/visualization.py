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
                      figsize: Tuple[int, int] = (10, 10),
                      outdir: Optional[str] = None,
                      show: bool = True,
                      dpi: int = 600):
    """
    Plot the scoring curves for cluster evaluation
    
    Args:
        scores_dict: Dictionary of scores, with method names as keys and score lists as values
        cluster_range: Range of cluster numbers to evaluate
        methods: Methods to plot, can be a string or list of strings, None means plot all methods
        clustering_algorithm: Name of the clustering algorithm
        figsize: Size of the figure
        outdir: Path to save the figure, None means do not save
        show: Whether to display the figure
        dpi: Image resolution
    """
    from habit.core.habitat_analysis.algorithms.cluster_validation_methods import get_method_description, get_optimization_direction
    
    # If methods is None, use all methods in scores_dict
    if methods is None:
        methods = list(scores_dict.keys())
    elif isinstance(methods, str):
        methods = [methods]
    
    figdir = os.path.join(outdir, 'visualizations', 'habitat_clustering')
    os.makedirs(figdir, exist_ok=True)

    # Plot for each method
    for i, method in enumerate(methods):
        if method not in scores_dict:
            continue
        
        _, ax = plt.subplots(1, 1, figsize=figsize)
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
            # For methods where lower is better
            best_idx = np.argmin(scores)
            best_score = scores[best_idx]
            criterion = "Minimum"
        elif optimization == 'elbow':
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
        fig_path = os.path.join(figdir, f'{clustering_algorithm}_{method}_cluster_validation_scores.png')
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    
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


def plot_cluster_results(
    X, 
    labels, 
    centers=None, 
    title=None, 
    feature_names=None, 
    save_path=None, 
    show=False, 
    dpi=600, 
    plot_3d=False, 
    explained_variance=None,
    # Configurable visual parameters
    figsize: Optional[Tuple[int, int]] = None,
    alpha: float = 0.7,
    marker_size: int = 20,
    marker: str = 'o',
    center_marker: str = 'X',
    center_size: int = 50,
    center_color: str = 'red',
    cmap: str = 'tab10',
    reduction_method: str = 'pca',
    show_colorbar: bool = True,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    max_legend_items: int = 10
):
    """
    Plot scatter plot of clustering results (2D or 3D)
    
    Args:
        X: Input data, shape (n_samples, n_features)
        labels: Cluster labels, shape (n_samples,)
        centers: Cluster centers, plotted if not None
        title: Figure title
        feature_names: Feature names for x and y axis labels
        save_path: Path to save the figure, do not save if None
        show: Whether to display the figure (default False for batch processing)
        dpi: Image resolution (default 600)
        plot_3d: Whether to plot 3D scatter plot (default False)
        explained_variance: Explained variance ratio from PCA (for title)
        figsize: Figure size as (width, height), default (6, 5) for 2D, (7, 6) for 3D
        alpha: Transparency of scatter points (0-1, default 0.7)
        marker_size: Size of scatter points (default 20)
        marker: Marker style for data points (default 'o')
        center_marker: Marker style for cluster centers (default 'X')
        center_size: Size of center markers (default 50)
        center_color: Color of center markers (default 'red')
        cmap: Colormap for clusters, 'tab10' is good for discrete categories (default 'tab10')
        reduction_method: Dimensionality reduction method, 'pca' or 'tsne' (default 'pca')
        show_colorbar: Whether to show colorbar (default True)
        show_grid: Whether to show grid (default True)
        max_legend_items: Maximum number of legend items to show, hide legend if exceeded (default 10)
        grid_alpha: Transparency of grid lines (default 0.3)
    """
    # Convert to numpy array if input is DataFrame
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(labels, pd.Series):
        labels = labels.values
    if centers is not None and isinstance(centers, pd.DataFrame):
        centers = centers.values
    
    # Set default figure size based on plot type
    if figsize is None:
        figsize = (7, 6) if plot_3d else (6, 5)
    
    # Dimensionality reduction if needed
    n_components = 3 if plot_3d else 2
    explained_var = None
    centers_reduced = centers
    
    if X.shape[1] > n_components:
        if reduction_method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            if centers is not None:
                centers_reduced = reducer.transform(centers)
            explained_var = reducer.explained_variance_ratio_
        elif reduction_method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X.shape[0]-1))
            X_reduced = reducer.fit_transform(X)
            # TSNE cannot transform centers directly, set to None
            centers_reduced = None
            explained_var = None
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}. Use 'pca' or 'tsne'.")
    else:
        X_reduced = X
        explained_var = explained_variance
    
    # Get unique labels for color mapping
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Generate colors using colormap
    # For discrete colormaps like tab10, use indices directly
    if cmap in ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent']:
        cmap_obj = plt.cm.get_cmap(cmap)
        colors = [cmap_obj(i % cmap_obj.N) for i in range(n_clusters)]
    else:
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_clusters))
    
    # Create figure
    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster with different color (first, so centers appear on top)
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2], 
                c=[colors[idx]], label=f'Cluster {label}', 
                alpha=alpha, s=marker_size, marker=marker, zorder=1
            )
            # zorder=1 means the points are plotted first, so centers appear on top
        
        # Plot cluster centers (last, with alpha=1.0 and higher zorder to ensure opaque and on top)
        if centers_reduced is not None:
            ax.scatter(
                centers_reduced[:, 0], centers_reduced[:, 1], centers_reduced[:, 2], 
                c=center_color, marker=center_marker, s=center_size, 
                label='Centers', edgecolors='None', linewidths=1,
                alpha=1.0, zorder=10
            )
        
        # Set labels with labelpad for z-axis to ensure visibility
        if explained_var is not None and X.shape[1] > 3:
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontfamily='Arial', labelpad=5)
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontfamily='Arial', labelpad=5)
            ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)', fontfamily='Arial', labelpad=5)
        elif reduction_method.lower() == 'tsne':
            ax.set_xlabel('t-SNE 1', fontfamily='Arial', labelpad=5)
            ax.set_ylabel('t-SNE 2', fontfamily='Arial', labelpad=5)
            ax.set_zlabel('t-SNE 3', fontfamily='Arial', labelpad=5)
        else:
            ax.set_xlabel('Feature 1', fontfamily='Arial', labelpad=5)
            ax.set_ylabel('Feature 2', fontfamily='Arial', labelpad=5)
            ax.set_zlabel('Feature 3', fontfamily='Arial', labelpad=5)
        
        # Only show legend if number of clusters <= max_legend_items
        if n_clusters <= max_legend_items:
            ax.legend(loc='best', fontsize=8)
    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each cluster with different color (first, so centers appear on top)
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1], 
                c=[colors[idx]], label=f'Cluster {label}',
                marker=marker, alpha=alpha, s=marker_size, zorder=1
            )
        
        # Plot cluster centers (last, with alpha=1.0 and higher zorder to ensure opaque and on top)
        if centers_reduced is not None:
            ax.scatter(
                centers_reduced[:, 0], centers_reduced[:, 1], 
                c=center_color, marker=center_marker, s=center_size, 
                label='Centers', edgecolors='None', linewidths=1,
                alpha=1.0, zorder=10
            )
        
        # Set labels
        if explained_var is not None and X.shape[1] > 2:
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontfamily='Arial')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontfamily='Arial')
        elif reduction_method.lower() == 'tsne':
            ax.set_xlabel('t-SNE 1', fontfamily='Arial')
            ax.set_ylabel('t-SNE 2', fontfamily='Arial')
        elif feature_names and len(feature_names) >= 2:
            ax.set_xlabel(feature_names[0], fontfamily='Arial')
            ax.set_ylabel(feature_names[1], fontfamily='Arial')
        else:
            ax.set_xlabel('Feature 1', fontfamily='Arial')
            ax.set_ylabel('Feature 2', fontfamily='Arial')
        
        # Add legend (only if number of clusters <= max_legend_items) and grid
        if n_clusters <= max_legend_items:
            ax.legend(loc='best', fontsize=8)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=grid_alpha)
    
    # Set title
    if title:
        plt.title(title, fontfamily='Arial', fontsize=11)
    else:
        plt.title(f'Clustering Results (n_clusters={n_clusters})', fontfamily='Arial', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close() 