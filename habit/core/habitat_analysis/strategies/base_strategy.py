"""
Base strategy interface for habitat analysis.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Dict, Any

import pandas as pd

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis
    from habit.core.habitat_analysis.pipelines.base_pipeline import HabitatPipeline


class BaseClusteringStrategy(ABC):
    """
    Abstract base class for habitat analysis strategies.
    
    Each strategy should implement run() and return a results DataFrame.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize the strategy with a HabitatAnalysis instance.

        Args:
            analysis: HabitatAnalysis instance with shared utilities and configuration
        """
        self.analysis = analysis
        self.config = analysis.config
        self.logger = analysis.logger

    def _get_attributes_to_update(self) -> Dict[str, Any]:
        """
        Get mapping of attribute names to their values that should be updated in pipeline steps.
        
        This method can be overridden by subclasses to customize which attributes are updated.
        By default, it includes:
        - config: Updated to self.config
        - All manager attributes from self.analysis (automatically discovered)
        
        Returns:
            Dictionary mapping attribute names to their values
        """
        attributes_to_update: Dict[str, Any] = {
            'config': self.config,
        }
        
        # Automatically discover and update all manager attributes from analysis
        # This handles current managers (feature_manager, clustering_manager, result_manager)
        # and any future managers that follow the naming convention
        for attr_name in dir(self.analysis):
            # Match attributes ending with '_manager' (e.g., feature_manager, data_manager)
            # Exclude private attributes (starting with '_')
            if attr_name.endswith('_manager') and not attr_name.startswith('_'):
                manager = getattr(self.analysis, attr_name, None)
                if manager is not None:
                    attributes_to_update[attr_name] = manager
        
        return attributes_to_update

    def _select_pipeline_config(self, pipeline: "HabitatPipeline") -> Any:
        """
        Choose which config should be applied to a loaded pipeline.

        In predict mode, users may provide a minimal YAML without FeatureConstruction.
        In that case, we keep the pipeline's trained config (which includes the
        FeatureConstruction details) and only override runtime-safe fields like
        output paths and logging/plot flags.

        Args:
            pipeline: Loaded HabitatPipeline instance

        Returns:
            Configuration object to apply to the pipeline and its steps
        """
        if (
            self.config.run_mode == 'predict'
            and self.config.FeatureConstruction is None
            and pipeline.config is not None
        ):
            pipeline_config = pipeline.config
            # Override only runtime fields that should follow the predict config.
            pipeline_config.out_dir = self.config.out_dir
            pipeline_config.plot_curves = self.config.plot_curves
            pipeline_config.save_results_csv = self.config.save_results_csv
            pipeline_config.save_images = self.config.save_images
            pipeline_config.processes = self.config.processes
            pipeline_config.random_state = self.config.random_state
            pipeline_config.verbose = self.config.verbose
            pipeline_config.debug = self.config.debug
            return pipeline_config

        return self.config

    def _sync_feature_manager(
        self,
        pipeline_feature_manager: Any,
        runtime_feature_manager: Any
    ) -> None:
        """
        Synchronize a loaded pipeline's FeatureManager with current runtime paths.

        We keep the trained FeatureManager instance (it contains the fitted feature
        extraction configuration) and only update data paths and logging targets so
        it can operate on the current dataset and write logs consistently.

        Args:
            pipeline_feature_manager: FeatureManager from the loaded pipeline
            runtime_feature_manager: FeatureManager created for the current run
        """
        if (
            getattr(runtime_feature_manager, "images_paths", None) is not None
            and getattr(runtime_feature_manager, "mask_paths", None) is not None
        ):
            pipeline_feature_manager.set_data_paths(
                runtime_feature_manager.images_paths,
                runtime_feature_manager.mask_paths
            )

        if hasattr(runtime_feature_manager, "_log_file_path"):
            pipeline_feature_manager.set_logging_info(
                runtime_feature_manager._log_file_path,
                runtime_feature_manager._log_level
            )
    
    def _update_pipeline_references(self, pipeline: "HabitatPipeline") -> None:
        """
        Update references in loaded pipeline to use current analysis instances.
        
        This method automatically updates config and manager references in the pipeline
        and all its steps to use the current analysis instances. This ensures that
        config changes (like out_dir, plot_curves) are reflected in all steps.
        
        The method is highly extensible:
        1. Automatically discovers all manager attributes (ending with '_manager')
        2. Can be extended by overriding _get_attributes_to_update() in subclasses
        3. Supports any future managers without code changes (as long as they follow naming convention)
        
        Examples of automatically discovered attributes:
        - feature_manager: Updated to self.analysis.feature_manager
        - clustering_manager: Updated to self.analysis.clustering_manager
        - result_manager: Updated to self.analysis.result_manager
        - data_manager: Updated to self.analysis.data_manager (if added in future)
        - cache_manager: Updated to self.analysis.cache_manager (if added in future)
        
        Args:
            pipeline: Loaded HabitatPipeline instance to update
        """
        # Update pipeline-level config with predict-safe selection
        config_to_apply = self._select_pipeline_config(pipeline)
        pipeline.config = config_to_apply
        
        # Get attributes to update (can be customized by subclasses)
        attributes_to_update = self._get_attributes_to_update()
        attributes_to_update['config'] = config_to_apply
        
        # Update all steps in the pipeline
        for _, step in pipeline.steps:
            for attr_name, attr_value in attributes_to_update.items():
                if hasattr(step, attr_name):
                    if attr_name == 'feature_manager':
                        # Keep trained FeatureManager; only sync runtime paths/logging.
                        pipeline_feature_manager = getattr(step, attr_name, None)
                        if pipeline_feature_manager is not None:
                            self._sync_feature_manager(pipeline_feature_manager, attr_value)
                            continue
                    setattr(step, attr_name, attr_value)

    @abstractmethod
    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Execute the strategy and return results.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV (defaults to config.save_results_csv)

        Returns:
            Results DataFrame
        """
        raise NotImplementedError
