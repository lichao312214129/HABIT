"""
Advanced K-Fold Cross-Validation Workflow with sklearn Pipeline support.
Inherits from BaseWorkflow for consistent infrastructure.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

from .base_workflow import BaseWorkflow
from .pipeline_utils import FeatureSelectTransformer
from .models.factory import ModelFactory
from .evaluation.model_evaluation import calculate_metrics

class MachineLearningKFoldWorkflow(BaseWorkflow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, module_name="ml_kfold")
        self.results = {'folds': [], 'aggregated': {}}

    def run_pipeline(self):
        self.logger.info("Starting K-Fold Pipeline with Pipeline support...")
        
        # 1. Load Data
        X, y = self._load_and_prepare_data()
        
        # 2. Setup Split
        kf_conf = self.config.get('KFold', {})
        kf = (StratifiedKFold if kf_conf.get('stratified', True) else KFold)(
            n_splits=kf_conf.get('n_splits', 5), 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # 3. Iterate Folds
        for fold_idx, (t_idx, v_idx) in enumerate(kf.split(X, y)):
            self.logger.info(f"--- Processing Fold {fold_idx+1} ---")
            X_train, X_val = X.iloc[t_idx], X.iloc[v_idx]
            y_train, y_val = y.iloc[t_idx], y.iloc[v_idx]
            
            fold_res = self._process_fold(X_train, y_train, X_val, y_val, fold_idx + 1)
            self.results['folds'].append(fold_res)
            
        # 4. Aggregate & Finalize
        self._aggregate_results()
        self._finalize()

    def _process_fold(self, X_train, y_train, X_val, y_val, fold_id):
        fold_data = {'fold': fold_id, 'models': {}}
        models_config = self.config.get('models', {})
        
        for m_name, m_params in models_config.items():
            self.logger.info(f"Building Pipeline for {m_name}")
            
            # Create Pipeline: Selection Before -> Normalization -> Selection After -> Model
            pipeline = Pipeline([
                ('selector_before', FeatureSelectTransformer(
                    self.config.get('feature_selection_methods', []),
                    feature_names=list(X_train.columns),
                    before_z_score_only=True,
                    outdir=self.output_dir
                )),
                ('scaler', self._get_scaler()),
                ('selector_after', FeatureSelectTransformer(
                    self.config.get('feature_selection_methods', []),
                    feature_names=None,  # Will be auto-detected from input during fit
                    after_z_score_only=True,
                    outdir=self.output_dir
                )),
                ('model', ModelFactory.create_model(m_name, m_params))
            ])
            
            # Fit entire pipeline (Prevents data leakage automatically!)
            pipeline.fit(X_train, y_train)
            
            # Predict using pipeline
            probs = pipeline.predict_proba(X_val)
            # Improvement: Support both binary and multi-class probabilities
            num_classes = len(np.unique(y_train))
            if num_classes == 2 and probs.ndim > 1:
                probs_for_metrics = probs[:, 1]
            else:
                probs_for_metrics = probs
                
            preds = pipeline.predict(X_val)
            
            # Save fold metrics
            fold_data['models'][m_name] = {
                'metrics': calculate_metrics(y_val.values, preds, probs_for_metrics),
                'y_true': y_val.tolist(), 'y_prob': probs_for_metrics.tolist(), 'y_pred': preds.tolist()
            }
            
            # Save Pipeline Object for future external testing
            model_save_path = os.path.join(self.output_dir, 'models', f'fold_{fold_id}')
            os.makedirs(model_save_path, exist_ok=True)
            joblib.dump(pipeline, os.path.join(model_save_path, f'{m_name}_pipeline.pkl'))
            
        return fold_data

    def _get_scaler(self):
        """
        Get scaler and configure it to output pandas DataFrame.
        This ensures feature names are preserved through the pipeline.
        """
        method = self.config.get('normalization', {}).get('method', 'z_score')
        scaler = {'z_score': StandardScaler(), 'min_max': MinMaxScaler(), 'robust': RobustScaler()}.get(method, StandardScaler())
        
        # Set output to pandas to preserve column names (sklearn 1.2+)
        try:
            scaler.set_output(transform="pandas")
        except AttributeError:
            # Fallback for older sklearn versions - scaler will output numpy array
            self.logger.warning("sklearn version does not support set_output. Consider upgrading to sklearn>=1.2 for better DataFrame support.")
        
        return scaler

    def _aggregate_results(self):
        """Standard aggregation logic."""
        model_names = self.results['folds'][0]['models'].keys()
        for m_name in model_names:
            y_true, y_prob, y_pred, fold_aucs = [], [], [], []
            for f in self.results['folds']:
                m_data = f['models'][m_name]
                y_true.extend(m_data['y_true']); y_prob.extend(m_data['y_prob'])
                y_pred.extend(m_data['y_pred']); fold_aucs.append(m_data['metrics']['auc'])
            
            # Convert list of lists back to numpy for potential multi-class processing
            y_true_arr = np.array(y_true)
            y_prob_arr = np.array(y_prob)
            y_pred_arr = np.array(y_pred)

            self.results['aggregated'][m_name] = {
                'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
                'overall_metrics': calculate_metrics(y_true_arr, y_pred_arr, y_prob_arr),
                'raw': {'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}
            }

    def _finalize(self):
        summary = []
        for m_name, res in self.results['aggregated'].items():
            row = {'Model': m_name, 'AUC_Mean': res['auc_mean'], 'AUC_Std': res['auc_std']}
            row.update({f"Overall_{k}": v for k, v in res['overall_metrics'].items()})
            summary.append(row)
        
        self._save_common_results(pd.DataFrame(summary), prefix="kfold_")
        
        # Plotting via PlotManager (aggregated results across all folds)
        self.plot_manager.run_workflow_plots(
            self.results['aggregated'], 
            prefix="kfold_",
            dataset_type='raw'
        )
        
        # Save comparison CSV
        results_df = pd.DataFrame({'subject_id': self.data_manager.data.index, 'label': self.data_manager.data[self.data_manager.label_col].values})
        for m_name, res in self.results['aggregated'].items():
            results_df[f"{m_name}_prob"] = res['raw']['y_prob']
            results_df[f"{m_name}_pred"] = res['raw']['y_pred']
        results_df.to_csv(os.path.join(self.output_dir, 'all_prediction_results.csv'), index=False)