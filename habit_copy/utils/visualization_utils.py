"""
Visualization utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_habitats(habitat_labels: np.ndarray, feature_data: np.ndarray, coords: np.ndarray = None, 
                  figsize: tuple = (10, 10), cmap: str = 'viridis', title: str = 'Habitat Distribution', 
                  alpha: float = 0.8, s: int = 50, legend: bool = True, **kwargs) -> tuple:
    """
    Plot habitat distribution
    
    Args:
        habitat_labels: Habitat labels, shape (n_samples,)
        feature_data: Feature data, shape (n_samples, n_features)
        coords: Coordinate information, shape (n_samples, 2), if None, PCA dimensionality reduction is used
        figsize: Figure size
        cmap: Color map
        title: Figure title
        alpha: Point transparency
        s: Point size
        legend: Whether to display legend
        **kwargs: Other parameters passed to plt.scatter
    
    Returns:
        tuple: (fig, ax) Figure and axis objects
    """
    if coords is None:
        # Use PCA dimensionality reduction to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(feature_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(habitat_labels)
    
    # If cmap is a string type, convert to ListedColormap
    if isinstance(cmap, str):
        # Use built-in color map
        cmap = plt.get_cmap(cmap, len(unique_labels))
    elif isinstance(cmap, list):
        # Use custom color list
        cmap = ListedColormap(cmap)
    
    # Plot points for each habitat
    for i, label in enumerate(unique_labels):
        mask = habitat_labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  color=cmap(i) if hasattr(cmap, '__call__') else cmap, 
                  label=f'Habitat {label}',
                  alpha=alpha, s=s, **kwargs)
    
    if legend:
        ax.legend(loc='best')
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax

def plot_feature_scores(feature_names: list, feature_scores: list, figsize: tuple = (12, 8), top_n: int = None, 
                        title: str = 'Feature Importance Scores', color: str = 'skyblue', 
                        sort: bool = True, **kwargs) -> tuple:
    """
    Plot feature importance scores
    
    Args:
        feature_names: List of feature names
        feature_scores: List of feature scores
        figsize: Figure size
        top_n: Only display top N features
        title: Figure title
        color: Bar chart color
        sort: Whether to sort by score
        **kwargs: Other parameters passed to plt.bar
    
    Returns:
        tuple: (fig, ax) Figure and axis objects
    """
    # Create dataframe
    import pandas as pd
    df = pd.DataFrame({'feature': feature_names, 'score': feature_scores})
    
    # Sort
    if sort:
        df = df.sort_values('score', ascending=False)
    
    # Get top_n features
    if top_n is not None:
        df = df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw bar chart
    ax.bar(df['feature'], df['score'], color=color, **kwargs)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Importance Score')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax 