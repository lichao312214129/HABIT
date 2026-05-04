"""
Advanced K-Fold cross-validation workflow.

This workflow remains the public entry point while delegating fold/model loops
to ``KFoldRunner`` so the class is easier to maintain.
"""

import os
import joblib
from typing import Any, Dict, List

import pandas as pd
from .base import BaseWorkflow
from ..runners.kfold import KFoldRunner
from ..models.ensemble import HabitEnsembleModel
from ..config_schemas import MLConfig
from habit.utils.io_utils import save_csv, save_json


class MachineLearningKFoldWorkflow(BaseWorkflow):
    """
    Backward-compatible K-Fold workflow facade.

    The public API and output structure stay unchanged. Internal fold execution
    is delegated to :class:`~habit.core.machine_learning.runners.kfold.KFoldRunner`.
    """

    def __init__(self, config: MLConfig):
        super().__init__(config, module_name="ml_kfold")
        self.results = {'folds': [], 'aggregated': {}}
        self.fold_pipelines: Dict[str, List[Any]] = {}
        self.runner = KFoldRunner(workflow=self)

    def run(self) -> None:
        self.logger.info("Starting K-Fold Pipeline...")

        # 1. Load data.
        X, y = self._load_and_prepare_data()

        # 2. Delegate fold execution to runner and keep public attributes stable.
        self.results, self.fold_pipelines, summary_results = self.runner.run(X=X, y=y)

        # 3. Save ensemble models.
        self._save_ensemble_models()

        # 4. Persist reports explicitly.
        save_json(
            self.results,
            os.path.join(self.output_dir, f"{self.module_name}_results.json"),
        )
        save_csv(
            pd.DataFrame(summary_results),
            os.path.join(self.output_dir, f"{self.module_name}_summary.csv"),
        )

        # 5. Render plots explicitly when visualization is enabled.
        if bool(getattr(self.config_obj, "is_visualize", True)):
            self.plot_manager.run_workflow_plots(
                self.results.get("aggregated", {}),
                prefix="kfold_",
            )

    def _save_ensemble_models(self):
        """Creates and saves HabitEnsembleModel for each model type."""
        self.logger.info("Creating and saving ensemble models...")
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for m_name, pipelines in self.fold_pipelines.items():
            if not pipelines:
                continue
            
            ensemble = HabitEnsembleModel(estimators=pipelines, voting='soft')
            
            # Save the ensemble
            save_path = os.path.join(models_dir, f'{m_name}_ensemble_final.pkl')
            try:
                joblib.dump(ensemble, save_path)
                self.logger.info(f"Saved ensemble model for {m_name} to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save ensemble model for {m_name}: {e}")
