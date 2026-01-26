import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from habit.utils.log_utils import get_module_logger


class MetricsStore:
    """
    Unified metrics storage manager.
    Handles nested dictionary operations for metrics data structure.
    """
    def __init__(self):
        self.store: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def ensure_group(self, group_name: str) -> Dict[str, Any]:
        if group_name not in self.store:
            self.store[group_name] = {}
        return self.store[group_name]

    def ensure_model(self, group_name: str, model_name: str) -> Dict[str, Any]:
        group = self.ensure_group(group_name)
        if model_name not in group:
            group[model_name] = {}
        return group[model_name]

    def add_metrics(self, group_name: str, model_name: str, metrics_type: str, metrics_data: Dict[str, Any]):
        model_store = self.ensure_model(group_name, model_name)
        model_store[metrics_type] = metrics_data

    def add_threshold(self, group_name: str, model_name: str, threshold_type: str, threshold_value: float):
        model_store = self.ensure_model(group_name, model_name)
        if 'thresholds' not in model_store:
            model_store['thresholds'] = {}
        model_store['thresholds'][threshold_type] = threshold_value

    def get(self) -> Dict[str, Any]:
        return self.store

    def clear(self):
        self.store.clear()


class ReportExporter:
    """
    Handles data export and reporting for machine learning workflows.
    Responsible for saving predictions, metrics, and other artifacts.
    """
    
    def __init__(self, output_dir: str, logger=None):
        """
        Initialize the ReportExporter.
        
        Args:
            output_dir (str): Directory where reports will be saved.
            logger: Logger instance. If None, a new one will be created.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logger or get_module_logger('reporting')

    def save_predictions(self, data: pd.DataFrame, filename: str = 'predictions.csv') -> str:
        """
        Save prediction DataFrame to CSV.
        
        Args:
            data (pd.DataFrame): The data to save.
            filename (str): The filename (relative to output_dir).
            
        Returns:
            str: The full path to the saved file.
        """
        save_path = os.path.join(self.output_dir, filename)
        try:
            data.to_csv(save_path, index=True) # Assuming index contains meaningful ID if set
            self.logger.info(f"Predictions saved to {save_path}")
            return save_path
        except Exception as e:
            self.logger.error(f"Failed to save predictions to {save_path}: {e}")
            raise

    def save_metrics(self, metrics: Dict[str, Any], filename: str = 'metrics.json') -> str:
        """
        Save metrics dictionary to JSON.
        
        Args:
            metrics (Dict): The metrics dictionary.
            filename (str): The filename.
            
        Returns:
            str: The full path to the saved file.
        """
        # Create subdirectory if filename contains path separators
        if os.path.dirname(filename):
            os.makedirs(os.path.join(self.output_dir, os.path.dirname(filename)), exist_ok=True)
            
        save_path = os.path.join(self.output_dir, filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Metrics saved to {save_path}")
            return save_path
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {save_path}: {e}")
            raise

    def update_metrics_file(self, new_metrics: Dict[str, Any], filename: str = 'metrics.json') -> str:
        """
        Update an existing metrics JSON file with new metrics.
        Merges strictly nested dictionaries.
        
        Args:
            new_metrics (Dict): New metrics to merge.
            filename (str): The filename.
            
        Returns:
            str: The full path to the saved file.
        """
        save_path = os.path.join(self.output_dir, filename)
        existing_metrics = {}
        
        # Load existing if available
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read existing metrics file {save_path}: {e}. Overwriting.")
        
        # Merge logic (simplified version of deep merge)
        # 1. Iterate over top-level keys (e.g., dataset groups)
        for group_key, group_val in new_metrics.items():
            if group_key not in existing_metrics:
                existing_metrics[group_key] = group_val
            else:
                # 2. Iterate over second-level keys (e.g., models)
                if isinstance(group_val, dict):
                    for model_key, model_val in group_val.items():
                        if model_key not in existing_metrics[group_key]:
                            existing_metrics[group_key][model_key] = model_val
                        else:
                            # 3. Update model metrics
                            if isinstance(model_val, dict):
                                existing_metrics[group_key][model_key].update(model_val)
                            else:
                                existing_metrics[group_key][model_key] = model_val
                else:
                    existing_metrics[group_key] = group_val
                    
        return self.save_metrics(existing_metrics, filename)

    def merge_and_save_metrics(self, all_metrics: Dict[str, Any], subdir: str = 'metrics',
                               filename: str = 'metrics.json') -> str:
        """
        Merge and save all metrics to a single JSON file.

        Args:
            all_metrics (Dict): Complete metrics dictionary to save.
            subdir (str): Subdirectory for metrics file.
            filename (str): The filename.

        Returns:
            str: The full path to the saved file.
        """
        metrics_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(metrics_dir, exist_ok=True)

        save_path = os.path.join(metrics_dir, filename)

        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
                for group_key in all_metrics:
                    if group_key in existing_metrics:
                        for model_key in all_metrics[group_key]:
                            if model_key in existing_metrics[group_key]:
                                existing_metrics[group_key][model_key].update(all_metrics[group_key][model_key])
                            else:
                                existing_metrics[group_key][model_key] = all_metrics[group_key][model_key]
                    else:
                        existing_metrics[group_key] = all_metrics[group_key]
                all_metrics = existing_metrics
            except Exception as e:
                self.logger.warning(f"Failed to read existing metrics file: {e}. Overwriting.")

        return self.save_metrics(all_metrics, os.path.join(subdir, filename))
