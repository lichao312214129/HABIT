"""
Model persistence helpers for machine-learning workflows.
"""

from __future__ import annotations

import os
from typing import Dict

import joblib

from ..core.results import RunResult


class ModelStore:
    """
    Save trained estimators from a run result.
    """

    def __init__(self, output_dir: str, is_save_model: bool = True) -> None:
        self.output_dir = output_dir
        self.is_save_model = is_save_model

    def save(self, run_result: RunResult) -> Dict[str, str]:
        """
        Save holdout final pipelines.

        Parameters
        ----------
        run_result:
            Structured holdout run output.

        Returns
        -------
        Dict[str, str]
            Mapping ``model_name -> saved_path``.
        """
        if not self.is_save_model:
            return {}

        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        saved_paths: Dict[str, str] = {}
        for model_name, model_result in run_result.models.items():
            save_path = os.path.join(model_dir, f"{model_name}_final_pipeline.pkl")
            joblib.dump(model_result.fitted_estimator, save_path)
            saved_paths[model_name] = save_path
        return saved_paths
