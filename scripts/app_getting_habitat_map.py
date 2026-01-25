"""
Command-line interface for running habitat analysis
This module provides functionality for analyzing medical image habitats through feature extraction,
supervoxel clustering, and habitat clustering. It supports various feature extraction methods
and clustering algorithms for comprehensive habitat analysis.
"""

import sys  
import tkinter as tk
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, BooleanVar, Checkbutton
from habit.core.habitat_analysis import HabitatAnalysis
from habit.core.common.service_configurator import ServiceConfigurator

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
        configurator = ServiceConfigurator(config_path=config_file)
        config = configurator.config
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


def setup_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name (str): Name of the logger
        log_dir (str, optional): Directory to store log files. If None, logs will only be printed to console
        level (int, optional): Logging level. Defaults to logging.INFO
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is provided)
    if log_dir:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

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
    
    # Initialize logger
    logger = setup_logger(
        name="habitat_analysis",
        log_dir="./logs" if not debug_mode else None,
        level=logging.DEBUG if debug_mode else logging.INFO
    )
    
    # If no configuration file specified in command line, use GUI selection
    if not config_file:
        config_file = select_config_file()
        if not config_file:
            logger.error("No configuration file selected, program exiting")
            return
        
        # If configuration file selected via GUI, confirm configuration through GUI
        config, gui_debug_mode = show_config_dialog(config_file)
        if not config:
            logger.error("Configuration cancelled, program exiting")
            return
        # If debug mode not specified in command line, use GUI setting
        if not debug_mode:
            debug_mode = gui_debug_mode
    else:
        # Configuration file specified in command line, load using ServiceConfigurator
        try:
            configurator = ServiceConfigurator(config_path=config_file)
            config = configurator.config
        except Exception as e:
            logger.error(f"Configuration load error: {str(e)}")
            return
    
    # Display registered feature extractors and clustering algorithms
    logger.info("Registered feature extractors: %s", get_available_feature_extractors())
    logger.info("Registered clustering algorithms: %s", get_available_clustering_algorithms())
    
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
    
    # Get clustering mode (one_step or two_step)
    clustering_mode = habitats_config.get('clustering_mode', 'two_step')
    
    # Extract supervoxel method configuration
    supervoxel_config = habitats_config.get('supervoxel', {})
    supervoxel_method = supervoxel_config.get('algorithm', 'kmeans')
    n_clusters_supervoxel = supervoxel_config.get('n_clusters', 50)
    
    # Extract one-step settings (for one_step mode)
    one_step_settings = supervoxel_config.get('one_step_settings', None)

    # Extract habitat method configuration
    habitat_config = habitats_config.get('habitat', {})
    habitat_method = habitat_config.get('algorithm', 'kmeans')
    n_clusters_habitats_max = habitat_config.get('max_clusters', 10)
    n_clusters_habitats_min = habitat_config.get('min_clusters', 2)
    habitat_cluster_selection_method = habitat_config.get('habitat_cluster_selection_method', None)
    best_n_clusters = habitat_config.get('best_n_clusters', None)
    # Convert best_n_clusters to integer if it's not None and can be converted to an integer
    if best_n_clusters is not None and str(best_n_clusters).isdigit():
        best_n_clusters = int(best_n_clusters)
    
    # Get mode parameter - either 'training' or 'testing'
    mode = habitat_config.get('mode', 'training')
    
    # Log parameters
    logger.info("==== Habitat Clustering Parameters ====")
    logger.info("Config file: %s", config_file)
    logger.info("Data directory: %s", data_dir)
    logger.info("Output folder: %s", out_dir)
    logger.info("Feature configuration: %s", feature_config)
    logger.info("Clustering mode: %s", clustering_mode)
    logger.info("Supervoxel method: %s", supervoxel_method)
    logger.info("Supervoxel clusters: %d", n_clusters_supervoxel)
    if clustering_mode == 'one_step' and one_step_settings:
        logger.info("One-step settings: %s", one_step_settings)
    logger.info("Habitat method: %s", habitat_method)
    logger.info("Maximum habitat clusters: %d", n_clusters_habitats_max)
    logger.info("Habitat cluster selection method: %s", habitat_cluster_selection_method)
    logger.info("Best number of clusters (if specified): %s", best_n_clusters)
    logger.info("Mode: %s", mode)
    logger.info("Number of processes: %d", n_processes)
    logger.info("Generate plots: %s", plot_curves)
    logger.info("Random seed: %d", random_state)
    logger.info("Debug mode: %s", debug_mode)
    logger.info("=========================")
    
    # Create and run HabitatAnalysis
    habitat_analysis = HabitatAnalysis(
        root_folder=data_dir,
        out_folder=out_dir,
        feature_config=feature_config,
        clustering_mode=clustering_mode,
        supervoxel_clustering_method=supervoxel_method,
        n_clusters_supervoxel=n_clusters_supervoxel,
        habitat_clustering_method=habitat_method,
        n_clusters_habitats_max=n_clusters_habitats_max,
        n_clusters_habitats_min=n_clusters_habitats_min,
        habitat_cluster_selection_method=habitat_cluster_selection_method,
        best_n_clusters=best_n_clusters,
        one_step_settings=one_step_settings,
        mode=mode,
        n_processes=n_processes,
        plot_curves=plot_curves,
        random_state=random_state
    )
    
    try:
        habitat_analysis.run()
        logger.info("Habitat analysis completed successfully")
    except Exception as e:
        logger.error("Error during habitat analysis: %s", str(e))
        raise


if __name__ == "__main__":
    # 如果没有命令行参数，添加默认的调试用参数
    if len(sys.argv) == 1:
        # 添加默认的配置文件路径，方便调试
        sys.argv.extend(["--config", "./config/config_getting_habitat.yaml", "--debug"])
        print(f"调试模式：使用默认配置文件 'config_getting_habitat.yaml'")
    
    try:
        main()
    except Exception as e:
        logger = setup_logger("habitat_analysis", level=logging.ERROR)
        logger.error("Fatal error: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 

    # python scripts/app_getting_habitat_map.py --config F:\work\workstation_b\dingHuYingXiang\the_4_training_202504\users\config_getting_habitat.yaml