"""
Standard Machine Learning Workflow (Train/Test Split).
Inherits from BaseWorkflow for consistent infrastructure.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .base_workflow import BaseWorkflow
from .pipeline_utils import FeatureSelectTransformer
from .models.factory import ModelFactory
from .evaluation.model_evaluation import calculate_metrics

class MachineLearningWorkflow(BaseWorkflow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, module_name="ml_standard")
        self.results = {}

    def run_pipeline(self):
        self.logger.info("Starting Standard ML Pipeline with Pipeline support...")
        
        # 1. Load Data
        X, y = self._load_and_prepare_data()
        
        # 2. Split Data (using DataManager's robust splitting)
        X_train, X_test, y_train, y_test = self.data_manager.split_data()
        
        models_config = self.config.get('models', {})
        summary_results = []

        # 3. Process Models
        for m_name, m_params in models_config.items():
            self.logger.info(f"Training Model: {m_name}")
            
            # Create and Fit Pipeline
            # 1. Selection before scaling (e.g. variance filtering)
            # 2. Scaling
            # 3. Selection after scaling
            # 4. Model
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
            
            pipeline.fit(X_train, y_train)
            
            # Determine number of classes for probability handling
            num_classes = len(np.unique(y_train))
            
            # Evaluate on Training Set
            train_probs = pipeline.predict_proba(X_train)
            if num_classes == 2 and train_probs.ndim > 1:
                train_probs_for_metrics = train_probs[:, 1]
            else:
                train_probs_for_metrics = train_probs
            train_preds = pipeline.predict(X_train)
            train_metrics = calculate_metrics(y_train.values, train_preds, train_probs_for_metrics)
            
            # Evaluate on Test Set
            test_probs = pipeline.predict_proba(X_test)
            if num_classes == 2 and test_probs.ndim > 1:
                test_probs_for_metrics = test_probs[:, 1]
            else:
                test_probs_for_metrics = test_probs
            test_preds = pipeline.predict(X_test)
            test_metrics = calculate_metrics(y_test.values, test_preds, test_probs_for_metrics)
            
            # Store results for both train and test sets
            self.results[m_name] = {
                'train': {
                    'metrics': train_metrics,
                    'y_true': y_train.tolist(),
                    'y_prob': train_probs_for_metrics.tolist(),
                    'y_pred': train_preds.tolist(),
                },
                'test': {
                    'metrics': test_metrics,
                    'y_true': y_test.tolist(),
                    'y_prob': test_probs_for_metrics.tolist(),
                    'y_pred': test_preds.tolist(),
                },
                'pipeline': pipeline,
                'features': list(X_train.columns)
            }
            
            # Save artifacts
            model_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(pipeline, os.path.join(model_dir, f'{m_name}_final_pipeline.pkl'))
            
            # Summary results include both train and test metrics
            row = {'Model': m_name}
            row.update({f'Train_{k}': v for k, v in train_metrics.items()})
            row.update({f'Test_{k}': v for k, v in test_metrics.items()})
            summary_results.append(row)

        # 4. Finalize
        # Remove non-serializable objects before saving common results
        results_for_json = {}
        for m_name, res in self.results.items():
            serializable_res = res.copy()
            if 'pipeline' in serializable_res:
                del serializable_res['pipeline']
            results_for_json[m_name] = serializable_res
        
        # Backup original results, swap with serializable ones for saving, then restore
        original_results = self.results
        self.results = results_for_json
        self._save_common_results(pd.DataFrame(summary_results), prefix="standard_")
        self.results = original_results
        
        # Save detailed prediction results to CSV
        self._save_prediction_results(X_train, y_train, X_test, y_test)
        
        # 5. Plotting via PlotManager
        # Plot for training set
        self.logger.info("Generating plots for training set...")
        self.plot_manager.run_workflow_plots(
            self.results, 
            prefix="standard_train_", 
            X_test=X_train,
            dataset_type='train'
        )
        
        # Plot for test set
        self.logger.info("Generating plots for test set...")
        self.plot_manager.run_workflow_plots(
            self.results, 
            prefix="standard_test_", 
            X_test=X_test,
            dataset_type='test'
        )
        
        self.logger.info(f"Standard ML workflow completed. Models saved to {self.output_dir}")

    def _save_prediction_results(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series):
        """
        Save detailed prediction results for both train and test sets to CSV files.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        # Create train results DataFrame
        train_df = pd.DataFrame({
            'subject_id': X_train.index,
            'label': y_train.values,
            'dataset': 'train'
        })
        
        # Create test results DataFrame
        test_df = pd.DataFrame({
            'subject_id': X_test.index,
            'label': y_test.values,
            'dataset': 'test'
        })
        
        # Add predictions for each model
        for m_name, res in self.results.items():
            train_df[f'{m_name}_prob'] = res['train']['y_prob']
            train_df[f'{m_name}_pred'] = res['train']['y_pred']
            
            test_df[f'{m_name}_prob'] = res['test']['y_prob']
            test_df[f'{m_name}_pred'] = res['test']['y_pred']
        
        # Combine and save
        all_results_df = pd.concat([train_df, test_df], ignore_index=True)
        all_results_df.to_csv(os.path.join(self.output_dir, 'all_prediction_results.csv'), index=False)
        
        # Also save separate files for train and test
        train_df.to_csv(os.path.join(self.output_dir, 'train_prediction_results.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_dir, 'test_prediction_results.csv'), index=False)
        
        self.logger.info("Prediction results saved to CSV files.")

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
