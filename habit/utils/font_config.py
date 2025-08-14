"""
Global font configuration for publication-quality plots
Sets Arial font across all matplotlib plots in the project
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_publication_font():
    """
    Setup Arial font for publication-quality plots
    This function should be called at the beginning of any visualization module
    Optimized for smaller figure sizes suitable for SCI journals
    """
    # Configure font settings for publication quality - optimized for smaller figures
    font_config = {
        'font.family': 'Arial',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 10,      # Reduced from 12 for smaller figures
        'axes.labelsize': 10, # Reduced from 12
        'axes.titlesize': 11, # Reduced from 14
        'xtick.labelsize': 9, # Reduced from 11
        'ytick.labelsize': 9, # Reduced from 11
        'legend.fontsize': 9, # Reduced from 11
        'figure.titlesize': 12, # Reduced from 16
        'pdf.fonttype': 42,  # embed TrueType fonts for better compatibility
        'ps.fonttype': 42,   # embed TrueType fonts for better compatibility
        'axes.linewidth': 1.0,  # Slightly thinner axes
        'lines.linewidth': 1.2, # Slightly thinner lines
    }
    
    # Apply the configuration
    mpl.rcParams.update(font_config)
    
    # Also set it directly on pyplot for backwards compatibility
    plt.rcParams.update(font_config)
    
    return font_config

def get_font_config():
    """
    Returns the standard font configuration dictionary
    Optimized for smaller figures suitable for SCI journals
    """
    return {
        'fontfamily': 'Arial',
        'fontsize': 10  # Reduced from 12 for smaller figures
    }

def apply_font_to_text_elements(ax, fontfamily='Arial'):
    """
    Apply Arial font to all text elements in a matplotlib axis
    
    Args:
        ax: matplotlib axis object
        fontfamily: font family to apply (default: Arial)
    """
    # Set font for title
    if ax.get_title():
        ax.set_title(ax.get_title(), fontfamily=fontfamily)
    
    # Set font for axis labels
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontfamily=fontfamily)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontfamily=fontfamily)
    
    # Set font for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily(fontfamily)
    for label in ax.get_yticklabels():
        label.set_fontfamily(fontfamily)
    
    # Set font for legend if present
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontfamily(fontfamily)

# Initialize font configuration when module is imported
setup_publication_font()
