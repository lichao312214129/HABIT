"""
Visualization utility functions for HABIT package.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import seaborn as sns


def plot_feature_importance(
    feature_importances: Dict[str, np.ndarray],
    feature_names: List[str],
    output_dir: Union[str, Path],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importances: Dictionary mapping feature selector names to importance values
        feature_names: List of feature names
        output_dir: Directory to save plots
        top_n: Number of top features to show
        figsize: Figure size
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure there are no Chinese characters in feature names
    feature_names = [name.encode('ascii', 'ignore').decode() for name in feature_names]
    
    for name, importance in feature_importances.items():
        plt.figure(figsize=figsize)
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        top_indices = indices[:top_n]
        
        # Plot bar chart
        plt.barh(range(len(top_indices)), importance[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {name}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'feature_importance_{name}.png'), dpi=300)
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
        cmap: Colormap
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize if required
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(
    fpr: Dict[str, np.ndarray],
    tpr: Dict[str, np.ndarray],
    auc: Dict[str, float],
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: Dictionary mapping model names to false positive rates
        tpr: Dictionary mapping model names to true positive rates
        auc: Dictionary mapping model names to AUC values
        output_path: Path to save the plot
        figsize: Figure size
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curve for each model
    for name in fpr.keys():
        plt.plot(
            fpr[name], 
            tpr[name], 
            lw=2,
            label=f'{name} (AUC = {auc[name]:.3f})'
        )
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_habitat_map(
    habitat_map: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    output_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = 'viridis',
    alpha: float = 0.7
) -> None:
    """
    Plot habitat map overlaid on original image.
    
    Args:
        habitat_map: 3D habitat map array
        original_image: 3D original image array (optional)
        output_path: Path to save the plot
        figsize: Figure size
        cmap: Colormap for habitat map
        alpha: Alpha value for overlay
    """
    # Calculate middle slice for 2D visualization
    if len(habitat_map.shape) == 3:
        slice_index = habitat_map.shape[0] // 2
        habitat_slice = habitat_map[slice_index, :, :]
        
        if original_image is not None and len(original_image.shape) == 3:
            image_slice = original_image[slice_index, :, :]
        else:
            image_slice = None
    else:
        habitat_slice = habitat_map
        image_slice = original_image
    
    plt.figure(figsize=figsize)
    
    # If original image is provided, show it first
    if image_slice is not None:
        plt.imshow(image_slice, cmap='gray')
        plt.imshow(habitat_slice, cmap=cmap, alpha=alpha)
    else:
        plt.imshow(habitat_slice, cmap=cmap)
    
    plt.colorbar(label='Habitat ID')
    plt.axis('off')
    plt.title('Habitat Map')
    
    # Save or show
    if output_path:
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_distribution(
    features: pd.DataFrame,
    group_column: str,
    output_dir: Union[str, Path],
    max_features: int = 10,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot feature distribution by group.
    
    Args:
        features: DataFrame containing features
        group_column: Column name for grouping
        output_dir: Directory to save plots
        max_features: Maximum number of features to plot
        figsize: Figure size
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns
    feature_columns = [col for col in features.columns if col != group_column]
    
    # Limit number of features if needed
    if len(feature_columns) > max_features:
        feature_columns = feature_columns[:max_features]
    
    # Plot each feature
    for feature in feature_columns:
        plt.figure(figsize=figsize)
        
        # Box plot with swarm plot overlay
        ax = sns.boxplot(x=group_column, y=feature, data=features)
        sns.swarmplot(x=group_column, y=feature, data=features, color='black', alpha=0.5)
        
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'distribution_{feature}.png'), dpi=300)
        plt.close() 