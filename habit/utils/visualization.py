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
Visualization utilities for habitat analysis
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .font_config import PUBLICATION_FONT, get_font_config, setup_publication_font

# Setup publication-quality font (Arial on Windows, DejaVu Sans on Linux/WSL)
setup_publication_font()


def plot_cluster_scores(scores_dict: Dict[str, List[float]],
                      cluster_range: List[int],
                      methods: Optional[Union[List[str], str]] = None,
                      clustering_algorithm: str = 'kmeans',
                      figsize: Tuple[int, int] = (10, 10),
                      outdir: Optional[str] = None,
                      save_path: Optional[str] = None,
                      show: bool = True,
                      dpi: int = 600,
                      best_n_clusters: Optional[Dict[str, int]] = None):
    """
    Plot the scoring curves for cluster evaluation
    
    Args:
        scores_dict: Dictionary of scores, with method names as keys and score lists as values
        cluster_range: Range of cluster numbers to evaluate
        methods: Methods to plot, can be a string or list of strings, None means plot all methods
        clustering_algorithm: Name of the clustering algorithm
        figsize: Size of the figure
        outdir: Directory to save figures, None means do not save
        save_path: Explicit file path to save a single figure (overrides outdir)
        show: Whether to display the figure
        dpi: Image resolution
        best_n_clusters: Precomputed best cluster number per method to mark on the plot
    """
    from habit.core.habitat_analysis.clustering.cluster_validation_methods import get_method_description, get_optimization_direction
    
    # If methods is None, use all methods in scores_dict
    if methods is None:
        methods = list(scores_dict.keys())
    elif isinstance(methods, str):
        methods = [methods]
    
    figdir = None
    if outdir:
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

        # Mark the optimal number of clusters.
        # Prefer externally provided best_n_clusters to avoid recomputing the selection logic.
        best_n_clusters_value: Optional[int] = None
        # Provide a readable criterion label even when the best index is supplied.
        if optimization in ['kneedle', 'inertia']:
            criterion = "Kneedle"
        elif optimization == 'elbow':
            criterion = "Elbow"
        elif optimization == 'maximize':
            criterion = "Maximum"
        elif optimization == 'minimize':
            criterion = "Minimum"
        else:
            criterion = "Maximum"
        if best_n_clusters is not None and method in best_n_clusters:
            best_n_clusters_value = best_n_clusters[method]
        else:
            # Fallback to internal logic to keep compatibility for other call sites.
            if optimization == 'maximize':
                best_idx = int(np.argmax(scores))
            elif optimization == 'minimize':
                best_idx = int(np.argmin(scores))
            else:
                best_idx = int(np.argmax(scores))
            best_n_clusters_value = cluster_range[best_idx]

        # Map the provided best cluster number back to score index.
        if best_n_clusters_value in cluster_range:
            best_idx = cluster_range.index(best_n_clusters_value)
            best_score = scores[best_idx]
        else:
            # If the best cluster number is outside the plotting range, skip marking.
            best_score = None
            criterion = "N/A"
        
        # Mark the optimal point on the plot
        if best_score is not None:
            ax.plot(best_n_clusters_value, best_score, 'rx', markersize=12, markeredgewidth=3)
        
        # Set title and labels
        method_desc = get_method_description(clustering_algorithm, method)
        ax.set_title(
            f"{method_desc}\nOptimal Clusters = {best_n_clusters_value} ({criterion})",
            fontfamily=PUBLICATION_FONT
        )
        ax.set_xlabel("Number of Clusters", fontfamily=PUBLICATION_FONT)
        ax.set_ylabel(f"{method.capitalize()} Score", fontfamily=PUBLICATION_FONT)
        ax.grid(True)
    
        # Adjust layout
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        elif figdir:
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
    plt.xlabel('Number of clusters', fontfamily=PUBLICATION_FONT)
    plt.ylabel(score_type, fontfamily=PUBLICATION_FONT)
    plt.title(title, fontfamily=PUBLICATION_FONT)
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
    
    plt.xlabel('Number of clusters', fontfamily=PUBLICATION_FONT)
    plt.ylabel('Normalized score (higher is better)', fontfamily=PUBLICATION_FONT)
    plt.title(title, fontfamily=PUBLICATION_FONT)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def _get_cluster_3d_axis_labels(
    X: np.ndarray,
    explained_var: Optional[np.ndarray],
    reduction_method: str,
) -> Tuple[str, str, str]:
    """
    Build axis labels for 3D cluster scatter plots.

    Args:
        X: Original feature matrix, shape (n_samples, n_features)
        explained_var: PCA explained variance ratio or None
        reduction_method: 'pca' or 'tsne'

    Returns:
        Tuple of (x_label, y_label, z_label)
    """
    if explained_var is not None and X.shape[1] > 3:
        return (
            f'PC1 ({explained_var[0] * 100:.1f}%)',
            f'PC2 ({explained_var[1] * 100:.1f}%)',
            f'PC3 ({explained_var[2] * 100:.1f}%)',
        )
    if reduction_method.lower() == 'tsne':
        return ('t-SNE 1', 't-SNE 2', 't-SNE 3')
    return ('Feature 1', 'Feature 2', 'Feature 3')


def _save_interactive_cluster_3d_html(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    centers_reduced: Optional[np.ndarray],
    colors: List,
    unique_labels: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    title: Optional[str],
    n_clusters: int,
    save_path: str,
    alpha_3d: float,
    marker_size: int,
    marker: str,
    center_marker: str,
    center_size: int,
    center_color: str,
    max_legend_items: int,
) -> None:
    """
    Save an interactive 3D cluster scatter plot as a self-contained HTML file.

    Scatter points use partial transparency; cluster centers are fully opaque
    so they remain visible through dense point clouds. Users can rotate the view
    in a browser to find the best viewing angle.

    Args:
        X_reduced: Reduced coordinates, shape (n_samples, 3)
        labels: Cluster labels, shape (n_samples,)
        centers_reduced: Reduced cluster centers, shape (n_centers, 3) or None
        colors: Matplotlib colors, one per unique label
        unique_labels: Sorted unique cluster labels
        x_label: X-axis title
        y_label: Y-axis title
        z_label: Z-axis title
        title: Plot title
        n_clusters: Number of clusters
        save_path: Output HTML file path
        alpha_3d: Opacity for scatter points in [0, 1]
        marker_size: Matplotlib-style scatter marker area scale
        marker: Matplotlib marker style for data points
        center_marker: Matplotlib marker style for centers
        center_size: Matplotlib-style center marker area scale
        center_color: Color for cluster centers
        max_legend_items: Hide legend when cluster count exceeds this value
    """
    try:
        import matplotlib.colors as mcolors
        import plotly.graph_objects as go
    except ImportError:
        return

    plotly_marker_map = {
        'o': 'circle',
        's': 'square',
        '^': 'triangle-up',
        'v': 'triangle-down',
        'X': 'x',
        'x': 'x',
        'D': 'diamond',
    }
    scatter_symbol = plotly_marker_map.get(marker, 'circle')
    center_symbol = plotly_marker_map.get(center_marker, 'x')
    scatter_marker_size = max(2, marker_size // 5)
    center_marker_size = max(6, center_size // 4)

    fig = go.Figure()
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        hex_color = mcolors.to_hex(mcolors.to_rgba(colors[idx])[:3])
        fig.add_trace(
            go.Scatter3d(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                z=X_reduced[mask, 2],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(
                    size=scatter_marker_size,
                    color=hex_color,
                    opacity=alpha_3d,
                    symbol=scatter_symbol,
                ),
            )
        )

    if centers_reduced is not None:
        fig.add_trace(
            go.Scatter3d(
                x=centers_reduced[:, 0],
                y=centers_reduced[:, 1],
                z=centers_reduced[:, 2],
                mode='markers',
                name='Centers',
                marker=dict(
                    size=center_marker_size,
                    color=center_color,
                    opacity=1.0,
                    symbol=center_symbol,
                    line=dict(width=2, color=center_color),
                ),
            )
        )

    display_title = title if title else f'Clustering Results (n_clusters={n_clusters})'
    display_title = display_title.replace('\n', '<br>')

    fig.update_layout(
        title=dict(text=display_title, font=dict(family=PUBLICATION_FONT, size=14)),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        showlegend=n_clusters <= max_legend_items,
        width=900,
        height=700,
        font=dict(family=PUBLICATION_FONT),
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.write_html(save_path)


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
    max_legend_items: int = 10,
    random_state: int = 42,
    alpha_3d: float = 0.35,
    save_interactive_3d: bool = True,
    interactive_save_path: Optional[str] = None,
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
        alpha: Transparency of 2D scatter points (0-1, default 0.7)
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
        random_state: Random seed for t-SNE dimensionality reduction (default 42)
        alpha_3d: Transparency of 3D scatter points (0-1, default 0.35); centers stay opaque
        save_interactive_3d: When plot_3d is True, also save a rotatable HTML plot (default True)
        interactive_save_path: HTML output path; defaults to save_path stem + '_interactive.html'
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
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, X.shape[0]-1),
            )
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
        x_label_3d, y_label_3d, z_label_3d = _get_cluster_3d_axis_labels(
            X, explained_var, reduction_method
        )
        
        # Plot each cluster with different color (first, so centers appear on top)
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2], 
                c=[colors[idx]], label=f'Cluster {label}', 
                alpha=alpha_3d, s=marker_size, marker=marker, zorder=1
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
        ax.set_xlabel(x_label_3d, fontfamily=PUBLICATION_FONT, labelpad=5)
        ax.set_ylabel(y_label_3d, fontfamily=PUBLICATION_FONT, labelpad=5)
        ax.set_zlabel(z_label_3d, fontfamily=PUBLICATION_FONT, labelpad=5)
        
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
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontfamily=PUBLICATION_FONT)
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontfamily=PUBLICATION_FONT)
        elif reduction_method.lower() == 'tsne':
            ax.set_xlabel('t-SNE 1', fontfamily=PUBLICATION_FONT)
            ax.set_ylabel('t-SNE 2', fontfamily=PUBLICATION_FONT)
        elif feature_names and len(feature_names) >= 2:
            ax.set_xlabel(feature_names[0], fontfamily=PUBLICATION_FONT)
            ax.set_ylabel(feature_names[1], fontfamily=PUBLICATION_FONT)
        else:
            ax.set_xlabel('Feature 1', fontfamily=PUBLICATION_FONT)
            ax.set_ylabel('Feature 2', fontfamily=PUBLICATION_FONT)
        
        # Add legend (only if number of clusters <= max_legend_items) and grid
        if n_clusters <= max_legend_items:
            ax.legend(loc='best', fontsize=8)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=grid_alpha)
    
    # Set title
    if title:
        plt.title(title, fontfamily=PUBLICATION_FONT, fontsize=11)
    else:
        plt.title(f'Clustering Results (n_clusters={n_clusters})', fontfamily=PUBLICATION_FONT, fontsize=11)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if plot_3d and save_interactive_3d and save_path:
        html_path = interactive_save_path
        if html_path is None:
            base_path, _ = os.path.splitext(save_path)
            html_path = f'{base_path}_interactive.html'
        _save_interactive_cluster_3d_html(
            X_reduced=X_reduced,
            labels=labels,
            centers_reduced=centers_reduced,
            colors=colors,
            unique_labels=unique_labels,
            x_label=x_label_3d,
            y_label=y_label_3d,
            z_label=z_label_3d,
            title=title,
            n_clusters=n_clusters,
            save_path=html_path,
            alpha_3d=alpha_3d,
            marker_size=marker_size,
            marker=marker,
            center_marker=center_marker,
            center_size=center_size,
            center_color=center_color,
            max_legend_items=max_legend_items,
        )
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close() 