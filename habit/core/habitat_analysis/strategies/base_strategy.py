"""
Base strategy interface for habitat analysis.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from pathlib import Path

import pandas as pd

from habit.core.habitat_analysis.config_schemas import ResultColumns

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis
    from habit.core.habitat_analysis.pipelines.base_pipeline import HabitatPipeline


def _canonical_csv_column_order(df: pd.DataFrame) -> List[str]:
    """
    Return column order for habitats.csv: metadata columns first (fixed order),
    then all other columns in their current order.
    This ensures habitats.csv has a consistent, predictable column order across runs.
    """
    # Fixed order for standard metadata columns (only include if present)
    meta_order = [
        ResultColumns.SUBJECT,
        ResultColumns.SUPERVOXEL,
        ResultColumns.HABITATS,
        ResultColumns.COUNT,
    ]
    meta_cols = [c for c in meta_order if c in df.columns]
    # Rest of columns (features) in their current order
    other_cols = [c for c in df.columns if c not in meta_cols]
    return meta_cols + other_cols


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

    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None,
        load_from: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Template method for executing the strategy.
        
        This method defines the algorithm skeleton. Subclasses can override specific steps
        if needed, but most will only need to implement strategy-specific logic in hooks.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV (defaults to config.save_results_csv)
            load_from: Optional path to a saved pipeline. If provided, the pipeline
                is loaded and only transform() is executed.

        Returns:
            Results DataFrame
        """
        # Use config value if parameter not provided, allowing runtime override
        if save_results_csv is None:
            save_results_csv = self.config.save_results_csv
        
        subjects = self._prepare_subjects(subjects)
        X = self._build_input(subjects)
        pipeline_path = self._resolve_pipeline_path(load_from)
        
        # Ensure output directory exists
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)
        
        if load_from:
            self._run_predict_mode(pipeline_path, X)
        else:
            self._run_train_mode(pipeline_path, X)
        
        # Post-process results (hook for strategy-specific logic)
        self._post_process_results()
        
        # Update ResultManager with new results
        self.analysis.result_manager.results_df = self.analysis.results_df
        
        # Save results
        if save_results_csv:
            self._save_results()
        
        return self.analysis.results_df
    
    def _run_predict_mode(self, pipeline_path: Path, X: Dict[str, Dict]) -> None:
        """
        Run pipeline in predict mode (load from file).
        
        Args:
            pipeline_path: Path to saved pipeline
            X: Input data dict
        """
        from ..pipelines.base_pipeline import HabitatPipeline
        
        if self.config.verbose:
            strategy_name = self._get_strategy_name()
            self.logger.info(f"Loading and running {strategy_name} pipeline...")
        
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Saved pipeline not found at {pipeline_path}. "
                "Provide a valid load_from path or run without load_from to train."
            )
        
        # Load pipeline
        self.pipeline = HabitatPipeline.load(str(pipeline_path))
        
        # Update references in loaded pipeline to use current analysis instances
        self._update_pipeline_references(self.pipeline)
        
        # Disable image outputs and plots for prediction runs to avoid unnecessary I/O
        self.pipeline.config.plot_curves = False
        
        # Transform
        self.analysis.results_df = self.pipeline.transform(X)
    
    def _run_train_mode(self, pipeline_path: Path, X: Dict[str, Dict]) -> None:
        """
        Run pipeline in train mode (build and fit).
        
        Args:
            pipeline_path: Path to save trained pipeline
            X: Input data dict
        """
        from ..pipelines.pipeline_builder import build_habitat_pipeline
        
        if self.config.verbose:
            strategy_name = self._get_strategy_name()
            self.logger.info(f"Building and fitting {strategy_name} pipeline...")
        
        # Build new pipeline
        self.pipeline = build_habitat_pipeline(
            config=self.config,
            feature_manager=self.analysis.feature_manager,
            clustering_manager=self.analysis.clustering_manager,
            result_manager=self.analysis.result_manager
        )
        
        # Fit and transform
        self.analysis.results_df = self.pipeline.fit_transform(X)
        
        # Save pipeline
        if self.config.verbose:
            self.logger.info(f"Saving fitted pipeline to {pipeline_path}")
        self.pipeline.save(str(pipeline_path))
    
    def _get_strategy_name(self) -> str:
        """
        Get human-readable strategy name for logging.
        
        Returns:
            Strategy name (e.g., "One-Step", "Two-Step", "Direct Pooling")
        """
        # Default implementation: extract from class name
        class_name = self.__class__.__name__
        if 'OneStep' in class_name:
            return "One-Step"
        elif 'TwoStep' in class_name:
            return "Two-Step"
        elif 'DirectPooling' in class_name or 'Pooling' in class_name:
            return "Direct Pooling"
        return class_name.replace('Strategy', '')
    
    def _prepare_subjects(self, subjects: Optional[List[str]]) -> List[str]:
        """
        Normalize subject list and validate it is not empty.

        Args:
            subjects: Optional list of subject IDs

        Returns:
            List of subject IDs
        """
        if subjects is None:
            subjects = list(self.analysis.images_paths.keys())

        if not subjects:
            strategy_name = self._get_strategy_name()
            raise ValueError(f"No subjects provided for {strategy_name} strategy.")

        return list(subjects)

    def _build_input(self, subjects: List[str]) -> Dict[str, Dict]:
        """
        Build input dict for the pipeline.

        Args:
            subjects: List of subject IDs

        Returns:
            Dict of subject_id -> empty dict (pipeline will populate data)
        """
        return {subject: {} for subject in subjects}

    def _resolve_pipeline_path(self, load_from: Optional[str]) -> Path:
        """
        Resolve pipeline path for saving or loading.

        Args:
            load_from: Optional path to a saved pipeline

        Returns:
            Path to pipeline file
        """
        if load_from:
            return Path(load_from)
        return Path(self.config.out_dir) / "habitat_pipeline.pkl"
    
    def _post_process_results(self) -> None:
        """
        Post-process results after pipeline execution.
        
        Hook for strategy-specific result processing. Subclasses can override
        this method to add custom logic (e.g., column renaming, validation).
        
        Default implementation does nothing.
        """
        pass
    
    def _save_results(self) -> None:
        """
        Save results for the strategy.
        
        Hook for strategy-specific result saving. Subclasses can override
        this method to customize saving behavior.
        
        Default implementation saves CSV and habitat images.
        """
        if self.config.verbose:
            self.logger.info("Saving results...")
        
        # Save results CSV with consistent column order (metadata first, then features)
        csv_path = Path(self.config.out_dir) / "habitats.csv"
        df = self.analysis.results_df
        canonical_order = _canonical_csv_column_order(df)
        df[canonical_order].to_csv(str(csv_path), index=False)
        if self.config.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        # Save habitat images for each subject
        if self.config.save_images:
            self.analysis.result_manager.save_all_habitat_images(failed_subjects=[])
