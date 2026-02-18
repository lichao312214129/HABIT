"""
Standard Machine Learning Workflow (Train/Test Split).
Inherits from BaseWorkflow for consistent infrastructure.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from ..base_workflow import BaseWorkflow
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer
from ..config_schemas import MLConfig

class MachineLearningWorkflow(BaseWorkflow):
    def __init__(self, config: MLConfig):
        super().__init__(config, module_name="ml_standard")
        self.X_train = None
        self.X_test = None

    def run_pipeline(self):
        self.logger.info("Starting Standard ML Pipeline...")
        self.callbacks.on_pipeline_start()
        
        # 1. Load Data
        X, y = self._load_and_prepare_data()
        
        # 2. Split Data
        self.X_train, self.X_test, y_train, y_test = self.data_manager.split_data()
        
        # Access models from Pydantic config object
        models_config = self.config_obj.models
        summary_results = []

        # 3. Process Models
        for m_name, m_params in models_config.items():
            # Extract params dict from ModelConfig object
            model_params_dict = m_params.params if hasattr(m_params, 'params') else m_params
            
            self.logger.info(f"Training Model: {m_name}")
            self.callbacks.on_model_start(m_name)
            
            # Use centralized pipeline builder
            pipeline = self.pipeline_builder.build(m_name, model_params_dict, feature_names=list(self.X_train.columns))
            trained_estimator = self._train_with_optional_sampling(
                pipeline, self.X_train, y_train
            )
            
            # Prediction Containers (Pass explicit predictions from the model)
            train_container = PredictionContainer(
                y_true=y_train.values, 
                y_prob=trained_estimator.predict_proba(self.X_train),
                y_pred=trained_estimator.predict(self.X_train)
            )
            test_container = PredictionContainer(
                y_true=y_test.values, 
                y_prob=trained_estimator.predict_proba(self.X_test),
                y_pred=trained_estimator.predict(self.X_test)
            )
            
            train_metrics = calculate_metrics(train_container)
            test_metrics = calculate_metrics(test_container)
            
            # Store results
            self.results[m_name] = {
                'train': train_container.to_dict(),
                'test': test_container.to_dict(),
                'pipeline': trained_estimator,
                'features': list(self.X_train.columns)
            }
            # Manually inject metrics for downstream callbacks
            self.results[m_name]['train']['metrics'] = train_metrics
            self.results[m_name]['test']['metrics'] = test_metrics
            
            # Trigger model end callback (handles saving)
            self.callbacks.on_model_end(m_name, logs={'pipeline': trained_estimator})
            
            # Aggregate summary
            row = {'Model': m_name}
            row.update({f'Train_{k}': v for k, v in train_metrics.items()})
            row.update({f'Test_{k}': v for k, v in test_metrics.items()})
            summary_results.append(row)

        # 4. End Pipeline (handles final reports and plotting)
        self.callbacks.on_pipeline_end(logs={'summary_results': summary_results})
        self.logger.info(f"Standard ML workflow completed. Results saved to {self.output_dir}")