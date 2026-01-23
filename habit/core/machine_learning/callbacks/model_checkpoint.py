import os
import joblib
from .base import Callback

class ModelCheckpoint(Callback):
    """Saves model pipelines to disk."""
    def on_model_end(self, model_name, logs=None):
        # Use config_accessor for unified access
        is_save_model = self.workflow.config_accessor.get('is_save_model', True)
        if not is_save_model:
            return
            
        model_dir = os.path.join(self.workflow.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if it's a fold or a final model
        fold_id = logs.get('fold_id') if logs else None
        if fold_id:
            save_path = os.path.join(model_dir, f'fold_{fold_id}', f'{model_name}_pipeline.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = os.path.join(model_dir, f'{model_name}_final_pipeline.pkl')
            
        joblib.dump(logs['pipeline'], save_path)
        self.workflow.logger.info(f"Model {model_name} saved to {save_path}")
