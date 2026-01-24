import os
import pandas as pd
from .base import Callback
from habit.utils.io_utils import save_json, save_csv

class ReportCallback(Callback):
    """Handles saving summary metrics and prediction CSVs."""
    
    def on_pipeline_end(self, logs=None):
        self.workflow.logger.info("Generating final reports...")
        
        # 1. Save summary.csv and results.json
        summary_results = logs.get('summary_results', [])
        if summary_results:
            # We create a safe copy of results without complex objects for JSON
            results_for_json = {}
            for m_name, res in self.workflow.results.items():
                res_copy = res.copy()
                if 'pipeline' in res_copy: del res_copy['pipeline']
                results_for_json[m_name] = res_copy
            
            save_json(results_for_json, os.path.join(self.workflow.output_dir, f'{self.workflow.module_name}_results.json'))
            save_csv(pd.DataFrame(summary_results), os.path.join(self.workflow.output_dir, f'{self.workflow.module_name}_summary.csv'))

        # 2. Save prediction CSVs (if standard workflow)
        if hasattr(self.workflow, 'X_train'):
            self._save_standard_predictions()

    def _save_standard_predictions(self):
        wf = self.workflow
        y_train = wf.data_manager.data.loc[wf.X_train.index, wf.data_manager.label_col]
        y_test = wf.data_manager.data.loc[wf.X_test.index, wf.data_manager.label_col]
        
        train_df = pd.DataFrame({'subject_id': wf.X_train.index, 'label': y_train.values, 'dataset': 'train'})
        test_df = pd.DataFrame({'subject_id': wf.X_test.index, 'label': y_test.values, 'dataset': 'test'})
        
        for m_name, res in wf.results.items():
            train_df[f'{m_name}_prob'] = res['train']['y_prob']
            train_df[f'{m_name}_pred'] = res['train']['y_pred']
            test_df[f'{m_name}_prob'] = res['test']['y_prob']
            test_df[f'{m_name}_pred'] = res['test']['y_pred']
            
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        all_df.to_csv(os.path.join(wf.output_dir, 'all_prediction_results.csv'), index=False)
        wf.logger.info(f"Full prediction results saved to {wf.output_dir}")
