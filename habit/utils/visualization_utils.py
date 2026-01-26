"""
Visualization utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from .font_config import setup_publication_font, get_font_config

# Setup publication-quality Arial font
setup_publication_font()

def plot_habitats(habitat_labels: np.ndarray, feature_data: np.ndarray, coords: np.ndarray = None, 
                  figsize: tuple = (6, 6), cmap: str = 'viridis', title: str = 'Habitat Distribution', 
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
    
    ax.set_title(title, fontfamily='Arial')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax

def plot_feature_scores(feature_names: list, feature_scores: list, figsize: tuple = (8, 6), top_n: int = None, 
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
    ax.set_title(title, fontfamily='Arial')
    ax.set_xlabel('Feature Name', fontfamily='Arial')
    ax.set_ylabel('Importance Score', fontfamily='Arial')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax 

def format_shap_data(X: np.ndarray, decimal_places: int = 2) -> np.ndarray:
    """
    Format SHAP feature data to reduce decimal places
    
    Args:
        X (np.ndarray): Feature data, either a single instance or multiple instances
        decimal_places (int): Number of decimal places to round to
    
    Returns:
        np.ndarray: Formatted data with reduced decimal places
    """
    # Handle different input types
    if isinstance(X, np.ndarray):
        # Round numpy array
        return np.round(X, decimal_places)
    elif hasattr(X, 'values') and isinstance(X.values, np.ndarray):
        # Handle pandas DataFrame
        X.values = np.round(X.values, decimal_places)
        return X
    elif isinstance(X, list):
        # Handle list of arrays
        return [np.round(arr, decimal_places) if isinstance(arr, np.ndarray) else arr for arr in X]
    
    # Return original data if we don't know how to process it
    return X

def process_shap_explanation(shap_values, decimal_places: int = 2):
    """
    Process SHAP Explanation object to reduce decimal places in the display
    
    Args:
        shap_values: SHAP values object, either from old or new API
        decimal_places (int): Number of decimal places to round to
    
    Returns:
        Processed SHAP values with reduced decimal places
    """
    # Handle different SHAP object types
    try:
        # Check if it's a newer Explanation object
        if hasattr(shap_values, 'data') and isinstance(shap_values.data, np.ndarray):
            # Use a manual deep copy approach instead of copy method
            # Since Explanation object in newer versions might not have copy method
            import copy
            processed_values = copy.deepcopy(shap_values)
            processed_values.data = np.round(processed_values.data, decimal_places)
            if hasattr(processed_values, 'values') and isinstance(processed_values.values, np.ndarray):
                processed_values.values = np.round(processed_values.values, decimal_places)
            return processed_values
        elif isinstance(shap_values, list):
            # Handle list of arrays (common in older SHAP versions)
            return [np.round(arr, decimal_places) if isinstance(arr, np.ndarray) else arr 
                   for arr in shap_values]
        elif isinstance(shap_values, np.ndarray):
            # Handle numpy array directly
            return np.round(shap_values, decimal_places)
        else:
            # If we can't process it, return the original
            return shap_values
    except Exception as e:
        print(f"Warning: Failed to process SHAP values: {str(e)}")
        return shap_values 