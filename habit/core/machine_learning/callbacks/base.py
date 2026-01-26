"""
Base Callback System for Machine Learning Workflows
Inspired by Keras and PyTorch Lightning.
"""

from typing import Dict, Any, Optional

class Callback:
    """
    Abstract base class for all callbacks.
    """
    def __init__(self):
        self.workflow = None

    def set_workflow(self, workflow):
        self.workflow = workflow

    def on_pipeline_start(self, logs: Optional[Dict] = None):
        """Called at the very beginning of run_pipeline."""
        pass

    def on_model_start(self, model_name: str, logs: Optional[Dict] = None):
        """Called before training a specific model."""
        pass

    def on_model_end(self, model_name: str, logs: Optional[Dict] = None):
        """Called after a model is trained and evaluated."""
        pass

    def on_fold_start(self, fold_id: int, logs: Optional[Dict] = None):
        """Called before starting a new cross-validation fold."""
        pass

    def on_fold_end(self, fold_id: int, logs: Optional[Dict] = None):
        """Called after completing a cross-validation fold."""
        pass

    def on_pipeline_end(self, logs: Optional[Dict] = None):
        """Called at the very end of run_pipeline."""
        pass

class CallbackList:
    """
    Container for managing and triggering multiple callbacks.
    """
    def __init__(self, callbacks=None, workflow=None):
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            cb.set_workflow(workflow)

    def on_pipeline_start(self, logs=None):
        for cb in self.callbacks: cb.on_pipeline_start(logs)

    def on_model_start(self, model_name, logs=None):
        for cb in self.callbacks: cb.on_model_start(model_name, logs)

    def on_model_end(self, model_name, logs=None):
        for cb in self.callbacks: cb.on_model_end(model_name, logs)

    def on_fold_start(self, fold_id, logs=None):
        for cb in self.callbacks: cb.on_fold_start(fold_id, logs)

    def on_fold_end(self, fold_id, logs=None):
        for cb in self.callbacks: cb.on_fold_end(fold_id, logs)

    def on_pipeline_end(self, logs=None):
        for cb in self.callbacks: cb.on_pipeline_end(logs)
