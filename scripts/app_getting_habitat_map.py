"""
Command-line interface for running habitat analysis
This module provides functionality for analyzing medical image habitats through feature extraction,
supervoxel clustering, and habitat clustering. It supports various feature extraction methods
and clustering algorithms for comprehensive habitat analysis.
"""

import json
import os
import sys  
import yaml
import tkinter as tk
import argparse
from tkinter import filedialog, messagebox, BooleanVar, Checkbutton
from habit.core.habitat_analysis import HabitatAnalysis
from habit.utils.io_utils import load_config

# 导入特征提取和聚类算法工具函数
# TODO: 实现相应的功能函数，当前使用占位函数
def get_available_feature_extractors():
    """Return available feature extractors"""
    return ["radiomics", "statistical", "texture", "wavelet"]

def get_available_clustering_algorithms():
    """Return available clustering algorithms"""
    return ["kmeans", "gmm", "spectral", "hierarchical"]


def select_config_file():
    """
    Open a file dialog for user to select a configuration file
    
    Returns:
        str: Path to the selected configuration file, or None if user cancels
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        filetypes=[
            ("Configuration Files", "*.yaml;*.yml;*.json"), 
            ("YAML Files", "*.yaml;*.yml"),
            ("JSON Files", "*.json"),
            ("All Files", "*.*")
        ]
    )
    
    if not file_path:
        return None
        
    return file_path


def parse_args():
    """
    Parse command line arguments for Habitat analysis.

    This function sets up and parses the command line arguments required for running
    the Habitat analysis pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Path to the configuration file
            - debug: Boolean flag indicating whether to run in debug mode
    """
    parser = argparse.ArgumentParser(description='Run Habitat analysis')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def show_config_dialog(config_file):
    """
    Display a configuration dialog for user to confirm parameters and choose debug mode
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        tuple: (config_dict, debug_mode) where:
            - config_dict: Configuration dictionary, or None if user cancels
            - debug_mode: Boolean indicating debug mode status, or None if user cancels
    """
    try:
        config = load_config(config_file)
    except Exception as e:
        messagebox.showerror("Configuration Error", f"Failed to load configuration file: {str(e)}")
        return None, None
    
    # Create dialog window
    dialog = tk.Toplevel()
    dialog.title("Habitat Analysis Configuration")
    dialog.geometry("500x400")
    
    # Display loaded configuration information
    tk.Label(dialog, text=f"Configuration File: {config_file}", anchor="w").pack(fill="x", padx=10, pady=5)
    tk.Label(dialog, text=f"Data Directory: {config.get('data_dir')}", anchor="w").pack(fill="x", padx=10, pady=2)
    tk.Label(dialog, text=f"Output Directory: {config.get('out_dir')}", anchor="w").pack(fill="x", padx=10, pady=2)
    
    feature_config = config.get('FeatureConstruction', {})
    tk.Label(dialog, text=f"Feature Extraction Method: {feature_config.get('method', 'Not Specified')}", anchor="w").pack(fill="x", padx=10, pady=2)
    
    # Add Debug mode option
    debug_var = BooleanVar(value=False)
    debug_check = Checkbutton(dialog, text="Enable Debug Mode", variable=debug_var)
    debug_check.pack(padx=10, pady=10)
    
    # Result and control variables
    result = {"config": None, "debug": False}
    
    # Confirm and cancel buttons
    def on_confirm():
        result["config"] = config
        result["debug"] = debug_var.get()
        dialog.destroy()
    
    def on_cancel():
        result["config"] = None
        dialog.destroy()
    
    button_frame = tk.Frame(dialog)
    button_frame.pack(fill="x", pady=20)
    
    tk.Button(button_frame, text="Confirm", command=on_confirm).pack(side="right", padx=10)
    tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=10)
    
    # Wait for dialog to close
    dialog.transient()
    dialog.grab_set()
    dialog.wait_window()
    
    return result["config"], result["debug"]


def main() -> None:
    """
    Main function to run habitat analysis pipeline
    
    This function:
    1. Parses command line arguments
    2. Loads configuration (either from command line or GUI)
    3. Sets up analysis parameters
    4. Creates and runs the HabitatAnalysis instance
    """
    # Parse command line arguments
    args = parse_args()
    
    # Prioritize configuration file specified in command line arguments
    config_file = args.config
    debug_mode = args.debug
    
    # If no configuration file specified in command line, use GUI selection
    if not config_file:
        config_file = select_config_file()
        if not config_file:
            print("No configuration file selected, program exiting")
            return
        
        # If configuration file selected via GUI, confirm configuration through GUI
        config, gui_debug_mode = show_config_dialog(config_file)
        if not config:
            print("Configuration cancelled, program exiting")
            return
        # If debug mode not specified in command line, use GUI setting
        if not debug_mode:
            debug_mode = gui_debug_mode
    else:
        # Configuration file specified in command line, load directly
        try:
            config = load_config(config_file)
        except Exception as e:
            print(f"Configuration load error: {str(e)}")
            return
    
    # Display registered feature extractors and clustering algorithms
    print("Registered feature extractors:", get_available_feature_extractors())
    print("Registered clustering algorithms:", get_available_clustering_algorithms())
    
    # Set basic parameters
    data_dir = config.get('data_dir')
    out_dir = config.get('out_dir')
    n_processes = config.get('processes', 4)
    plot_curves = config.get('plot_curves', True)
    random_state = config.get('random_state', 42)
    
    # Extract feature construction configuration
    feature_config = config.get('FeatureConstruction', {})
    
    # Extract habitat segmentation configuration
    habitats_config = config.get('HabitatsSegmention', {})
    
    # Extract supervoxel method configuration
    supervoxel_config = habitats_config.get('supervoxel', {})
    supervoxel_method = supervoxel_config.get('algorithm', 'kmeans')
    n_clusters_supervoxel = supervoxel_config.get('n_clusters', 50)

    
    # Extract habitat method configuration
    habitat_config = habitats_config.get('habitat', {})
    habitat_method = habitat_config.get('algorithm', 'kmeas')
    n_clusters_habitats_max = habitat_config.get('max_clusters', 10)
    habitat_cluster_selection_method = habitat_config.get('habitat_cluster_selection_method', None)
    best_n_clusters = habitat_config.get('best_n_clusters', None)
    # Convert best_n_clusters to integer if it's not None and can be converted to an integer
    if best_n_clusters is not None and str(best_n_clusters).isdigit():
        best_n_clusters = int(best_n_clusters)
    
    
    # Print parameters
    print("==== Habitat Clustering Parameters ====")
    print(f"Config file: {config_file}")
    print(f"Data directory: {data_dir}")
    print(f"Output folder: {out_dir}")
    print(f"Feature configuration: {feature_config}")
    print(f"Supervoxel method: {supervoxel_method}")
    print(f"Supervoxel clusters: {n_clusters_supervoxel}")
    print(f"Habitat method: {habitat_method}")
    print(f"Maximum habitat clusters: {n_clusters_habitats_max}")
    print(f"Habitat cluster selection method: {habitat_cluster_selection_method}")
    print(f"Best number of clusters (if specified): {best_n_clusters}")
    print(f"Number of processes: {n_processes}")
    print(f"Generate plots: {plot_curves}")
    print(f"Random seed: {random_state}")
    print(f"Debug mode: {debug_mode}")
    print("=========================")
    
    # Create and run HabitatAnalysis
    habitat_analysis = HabitatAnalysis(
        root_folder=data_dir,
        out_folder=out_dir,
        feature_config=feature_config,
        supervoxel_clustering_method=supervoxel_method,
        n_clusters_supervoxel=n_clusters_supervoxel,
        habitat_clustering_method=habitat_method,
        n_clusters_habitats_max=n_clusters_habitats_max,
        habitat_cluster_selection_method=habitat_cluster_selection_method,
        best_n_clusters=best_n_clusters,
        n_processes=n_processes,
        plot_curves=plot_curves,
        random_state=random_state
    )
    
    habitat_analysis.run()


if __name__ == "__main__":
    # 如果没有命令行参数，添加默认的调试用参数
    if len(sys.argv) == 1:
        # 添加默认的配置文件路径，方便调试
        sys.argv.extend(["--config", "./config/config_kmeans.yaml", "--debug"])
        print(f"调试模式：使用默认配置文件 'config_kmeans.yaml'")
    
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 