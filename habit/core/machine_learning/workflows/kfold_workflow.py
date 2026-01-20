"""
Advanced K-Fold Cross-Validation Workflow.
Inherits from BaseWorkflow for consistent infrastructure.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import StratifiedKFold, KFold
from ..base_workflow import BaseWorkflow
from ..evaluation.metrics import calculate_metrics
from ..evaluation.prediction_container import PredictionContainer

class MachineLearningKFoldWorkflow(BaseWorkflow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, module_name="ml_kfold")
        self.results = {'folds': [], 'aggregated': {}}

    def run_pipeline(self):
        self.logger.info("Starting K-Fold Pipeline...")
        self.callbacks.on_pipeline_start()
        
        # 1. Load Data
        X, y = self._load_and_prepare_data()
        
        # 2. Setup Split
        kf_conf = self.config.get('KFold', self.config) # Handle both nesting styles
        kf = (StratifiedKFold if kf_conf.get('stratified', True) else KFold)(
            n_splits=kf_conf.get('n_splits', 5), 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # 3. Iterate Folds
        for fold_idx, (t_idx, v_idx) in enumerate(kf.split(X, y)):
            fold_id = fold_idx + 1
            self.logger.info(f"--- Processing Fold {fold_id} ---")
            self.callbacks.on_fold_start(fold_id)
            
            X_train, X_val = X.iloc[t_idx], X.iloc[v_idx]
            y_train, y_val = y.iloc[t_idx], y.iloc[v_idx]
            
            fold_res = self._process_fold(X_train, y_train, X_val, y_val, fold_id)
            self.results['folds'].append(fold_res)
            self.callbacks.on_fold_end(fold_id)
            
        # 4. Aggregate & Finalize
        summary_results = self._aggregate_results()
        
        # 5. End Pipeline
        self.callbacks.on_pipeline_end(logs={'summary_results': summary_results})

    def _process_fold(self, X_train, y_train, X_val, y_val, fold_id):
        fold_data = {'fold': fold_id, 'models': {}}
        models_config = self.config.get('models', {})
        
        for m_name, m_params in models_config.items():
            self.callbacks.on_model_start(m_name, logs={'fold_id': fold_id})
            
            pipeline = self.pipeline_builder.build(m_name, m_params, feature_names=list(X_train.columns))
            pipeline.fit(X_train, y_train)
            
            container = PredictionContainer(
                y_true=y_val.values, 
                y_prob=pipeline.predict_proba(X_val),
                y_pred=pipeline.predict(X_val)
            )
            metrics_dict = calculate_metrics(container)
            
            fold_data['models'][m_name] = container.to_dict()
            fold_data['models'][m_name]['metrics'] = metrics_dict
            
            self.callbacks.on_model_end(m_name, logs={'pipeline': pipeline, 'fold_id': fold_id})
            
        return fold_data

    def _aggregate_results(self):
        """Standard aggregation logic."""
        model_names = self.results['folds'][0]['models'].keys()
        summary = []
        
        for m_name in model_names:
            y_true, y_prob, y_pred, fold_aucs = [], [], [], []
            for f in self.results['folds']:
                m_data = f['models'][m_name]
                y_true.extend(m_data['y_true'])
                y_prob.extend(m_data['y_prob'])
                y_pred.extend(m_data['y_pred'])
                fold_aucs.append(m_data['metrics']['auc'])
            
            # Use container for aggregated metrics
            agg_container = PredictionContainer(np.array(y_true), np.array(y_prob), np.array(y_pred))
            overall_metrics = calculate_metrics(agg_container)
            
            self.results['aggregated'][m_name] = agg_container.to_dict()
            self.results['aggregated'][m_name].update({
                'auc_mean': np.mean(fold_aucs), 
                'auc_std': np.std(fold_aucs),
                'metrics': overall_metrics
            })
            
            row = {'Model': m_name, 'AUC_Mean': np.mean(fold_aucs), 'AUC_Std': np.std(fold_aucs)}
            row.update({f"Overall_{k}": v for k, v in overall_metrics.items()})
            summary.append(row)
            
        return summary
