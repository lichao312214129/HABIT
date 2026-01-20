from .base import Callback

class VisualizationCallback(Callback):
    """Triggers plotting routines."""
    def on_pipeline_end(self, logs=None):
        if not self.workflow.config.get('is_visualize', True):
            return
            
        # Holdout workflow plotting
        if hasattr(self.workflow, 'X_train'):
            self.workflow.logger.info("Generating standard workflow plots...")
            self.workflow.plot_manager.run_workflow_plots(
                self.workflow.results, 
                prefix="standard_train_", 
                X_test=self.workflow.X_train
            )
            self.workflow.plot_manager.run_workflow_plots(
                self.workflow.results, 
                prefix="standard_test_", 
                X_test=self.workflow.X_test
            )
        # K-Fold workflow plotting
        elif hasattr(self.workflow, 'results') and 'aggregated' in self.workflow.results:
            self.workflow.logger.info("Generating K-Fold workflow plots...")
            self.workflow.plot_manager.run_workflow_plots(
                self.workflow.results['aggregated'], 
                prefix="kfold_"
            )
