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
        # Update pipeline-level config
        pipeline.config = self.config
        
        # Get attributes to update (can be customized by subclasses)
        attributes_to_update = self._get_attributes_to_update()
        
        # Update all steps in the pipeline
        for _, step in pipeline.steps:
            for attr_name, attr_value in attributes_to_update.items():
                if hasattr(step, attr_name):
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
