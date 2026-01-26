"""
Service Configurator

Provides a unified way to create and configure services with their dependencies.
This replaces the fallback pattern in class __init__ methods.
"""

import os
from typing import Any, Dict, Optional

from habit.utils.log_utils import get_module_logger, setup_logger, LoggerManager
from habit.core.machine_learning.workflows.comparison_workflow import ModelComparison
from habit.core.machine_learning.evaluation.model_evaluation import MultifileEvaluator
from habit.core.machine_learning.reporting.report_exporter import ReportExporter, MetricsStore
from habit.core.machine_learning.evaluation.threshold_manager import ThresholdManager
from habit.core.machine_learning.visualization.plot_manager import PlotManager
from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis
from habit.core.habitat_analysis.managers import FeatureManager, ClusteringManager, ResultManager
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig


class ServiceConfigurator:
    """
    Unified service configurator.
    
    Responsible for creating and configuring all services with their dependencies.
    Use this class instead of direct instantiation of complex workflow classes.
    
    Example:
        ```python
        configurator = ServiceConfigurator(config)
        
        # Create fully configured services
        comparison = configurator.create_model_comparison()
        habitat_analysis = configurator.create_habitat_analysis()
        ```
    """
    
    def __init__(
        self,
        config: Any,
        logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize ServiceConfigurator.
        
        Args:
            config: Configuration object or dictionary.
            logger: Logger instance (will be created if not provided).
            output_dir: Output directory (will be derived from config if not provided).
        """
        self.config = config
        self.output_dir = output_dir or getattr(config, 'output_dir', None) or getattr(config, 'out_dir', './output')
        self.logger = logger or self._create_logger()
        
        self._services: Dict[str, Any] = {}
    
    def _create_logger(self) -> Any:
        """Create and return a logger instance."""
        manager = LoggerManager()
        
        if manager.get_log_file() is not None:
            logger = get_module_logger('service_configurator')
            logger.info("Using existing logging configuration from CLI entry point")
        else:
            logger = setup_logger(
                name='service_configurator',
                output_dir=self.output_dir,
                log_filename='service_configurator.log'
            )
        
        return logger
    
    def _ensure_output_dir(self) -> str:
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
    
    def _get_habitat_config(self, config: Optional[Any] = None) -> HabitatAnalysisConfig:
        """
        Get HabitatAnalysisConfig from dict or existing config.
        
        Args:
            config: Configuration dict or object.
            
        Returns:
            HabitatAnalysisConfig Pydantic model.
        """
        config = config or self.config
        if isinstance(config, HabitatAnalysisConfig):
            return config
        if isinstance(config, dict):
            return HabitatAnalysisConfig.model_validate(config)
        return config
    
    # === ModelComparison Services ===
    
    def create_evaluator(self, output_dir: Optional[str] = None) -> MultifileEvaluator:
        """Create MultifileEvaluator instance."""
        output_dir = output_dir or self._ensure_output_dir()
        return MultifileEvaluator(output_dir=output_dir)
    
    def create_reporter(self, output_dir: Optional[str] = None) -> ReportExporter:
        """Create ReportExporter instance."""
        output_dir = output_dir or self._ensure_output_dir()
        return ReportExporter(output_dir=output_dir, logger=self.logger)
    
    def create_threshold_manager(self) -> ThresholdManager:
        """Create ThresholdManager instance."""
        return ThresholdManager()
    
    def create_plot_manager(self, config: Optional[Any] = None) -> PlotManager:
        """Create PlotManager instance."""
        output_dir = self._ensure_output_dir()
        # PlotManager expects a Pydantic config object, not a dict
        # Pass the config object directly if it's a Pydantic model, otherwise use self.config
        plot_config = config or self.config
        # If it's a dict, we need to convert it to the appropriate config type
        # But for ModelComparison, PlotManager should receive the full ModelComparisonConfig
        return PlotManager(config=plot_config, output_dir=output_dir)
    
    def create_metrics_store(self) -> MetricsStore:
        """Create MetricsStore instance."""
        return MetricsStore()
    
    def create_model_comparison(
        self,
        config: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> ModelComparison:
        """
        Create fully configured ModelComparison instance.
        
        Args:
            config: Configuration (uses configurator's config if not provided).
            output_dir: Output directory (uses configurator's output_dir if not provided).
            
        Returns:
            Configured ModelComparison instance.
        """
        config = config or self.config
        output_dir = output_dir or self._ensure_output_dir()
        
        return ModelComparison(
            config=config,
            evaluator=self.create_evaluator(output_dir),
            reporter=self.create_reporter(output_dir),
            threshold_manager=self.create_threshold_manager(),
            plot_manager=self.create_plot_manager(config),
            metrics_store=self.create_metrics_store(),
            logger=self.logger,
        )
    
    # === HabitatAnalysis Services ===
    
    def create_feature_manager(self, config: Optional[Any] = None) -> FeatureManager:
        """Create FeatureManager instance."""
        habitat_config = self._get_habitat_config(config)
        return FeatureManager(habitat_config, self.logger)
    
    def create_clustering_manager(self, config: Optional[Any] = None) -> ClusteringManager:
        """Create ClusteringManager instance."""
        habitat_config = self._get_habitat_config(config)
        return ClusteringManager(habitat_config, self.logger)
    
    def create_result_manager(self, config: Optional[Any] = None) -> ResultManager:
        """Create ResultManager instance."""
        habitat_config = self._get_habitat_config(config)
        return ResultManager(habitat_config, self.logger)
    
    def create_habitat_analysis(
        self,
        config: Optional[Any] = None,
    ) -> HabitatAnalysis:
        """
        Create fully configured HabitatAnalysis instance.
        
        Args:
            config: Configuration (uses configurator's config if not provided).
            
        Returns:
            Configured HabitatAnalysis instance.
        """
        config = config or self.config
        
        return HabitatAnalysis(
            config=config,
            feature_manager=self.create_feature_manager(config),
            clustering_manager=self.create_clustering_manager(config),
            result_manager=self.create_result_manager(config),
            logger=self.logger,
        )

    # === Preprocessing Services ===

    def create_batch_processor(self) -> Any:
        """
        Create BatchProcessor instance for image preprocessing.
        
        Returns:
            Configured BatchProcessor instance.
        """
        from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
        
        # BatchProcessor currently requires config_path.
        # We try to get it from the config object.
        config_file = getattr(self.config, 'config_file', None)
        
        if not config_file:
            # Fallback: if config was passed as dict or config_file is missing
            raise ValueError("BatchProcessor requires 'config_file' to be set in the configuration object.")

        return BatchProcessor(
            config_path=config_file,
            verbose=True
            # logger is handled internally by BatchProcessor for now, 
            # ideally we should inject self.logger into it in the future.
        )
    
    # === Machine Learning Workflow Services ===
    
    def create_ml_workflow(self, config: Optional[Any] = None) -> Any:
        """
        Create MachineLearningWorkflow instance.
        
        Args:
            config: Configuration (uses configurator's config if not provided).
            
        Returns:
            Configured MachineLearningWorkflow instance.
        """
        from habit.core.machine_learning.workflows.holdout_workflow import MachineLearningWorkflow
        from habit.core.machine_learning.config_schemas import MLConfig
        
        cfg = config or self.config
        
        # Ensure config is MLConfig object
        if not isinstance(cfg, MLConfig):
            if isinstance(cfg, dict):
                cfg = MLConfig.model_validate(cfg)
            elif hasattr(cfg, 'to_dict'):
                cfg = MLConfig.model_validate(cfg.to_dict())
            else:
                raise ValueError(f"Invalid configuration type for ML workflow: {type(cfg)}")
        
        return MachineLearningWorkflow(cfg)
    
    def create_kfold_workflow(self, config: Optional[Any] = None) -> Any:
        """
        Create MachineLearningKFoldWorkflow instance.
        
        Args:
            config: Configuration (uses configurator's config if not provided).
            
        Returns:
            Configured MachineLearningKFoldWorkflow instance.
        """
        from habit.core.machine_learning.workflows.kfold_workflow import MachineLearningKFoldWorkflow
        from habit.core.machine_learning.config_schemas import MLConfig
        
        cfg = config or self.config
        
        # Ensure config is MLConfig object
        if not isinstance(cfg, MLConfig):
            if isinstance(cfg, dict):
                cfg = MLConfig.model_validate(cfg)
            elif hasattr(cfg, 'to_dict'):
                cfg = MLConfig.model_validate(cfg.to_dict())
            else:
                raise ValueError(f"Invalid configuration type for K-Fold workflow: {type(cfg)}")
        
        return MachineLearningKFoldWorkflow(cfg)
    
    def create_prediction_workflow(self) -> Any:
        """Create PredictionWorkflow for model inference."""
        from habit.core.machine_learning.workflows.prediction_workflow import PredictionWorkflow
        from habit.core.machine_learning.config_schemas import PredictionConfig
        
        cfg = self.config
        
        # Ensure config is PredictionConfig
        if not isinstance(cfg, PredictionConfig):
             if isinstance(cfg, dict):
                 cfg = PredictionConfig.model_validate(cfg)
             elif hasattr(cfg, 'to_dict'):
                 cfg = PredictionConfig.model_validate(cfg.to_dict())
             else:
                 # Last resort: try to cast assuming fields match
                 cfg = PredictionConfig.model_validate(cfg.dict())

        return PredictionWorkflow(cfg, self.logger)
    
    # === Feature Extraction Services ===
    
    def create_feature_extractor(self, config: Optional[Any] = None) -> Any:
        """
        Create HabitatMapAnalyzer for feature extraction.
        
        Args:
            config: Optional configuration override.
            
        Returns:
            Configured HabitatMapAnalyzer instance.
        """
        from habit.core.habitat_analysis.analyzers.habitat_analyzer import HabitatMapAnalyzer
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
        
        cfg = config or self.config
        
        # Ensure we have a typed config
        if isinstance(cfg, dict):
            cfg = FeatureExtractionConfig.model_validate(cfg)
        elif not isinstance(cfg, FeatureExtractionConfig):
            # Try to validate if it's compatible, or raise error if unrelated config
            try:
                # If it's a BaseConfig but not FeatureExtractionConfig, try conversion via dict
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                cfg = FeatureExtractionConfig.model_validate(cfg_dict)
            except Exception as e:
                raise ValueError(f"Invalid configuration for Feature Extraction: {e}")
        
        return HabitatMapAnalyzer(
            params_file_of_non_habitat=str(cfg.params_file_of_non_habitat),
            params_file_of_habitat=str(cfg.params_file_of_habitat),
            raw_img_folder=str(cfg.raw_img_folder),
            habitats_map_folder=str(cfg.habitats_map_folder),
            out_dir=str(cfg.out_dir),
            n_processes=cfg.n_processes,
            habitat_pattern=cfg.habitat_pattern
        )


    
    # === Utility Methods ===
    
    def get_service(self, name: str) -> Any:
        """
        Get a cached service instance.
        
        Args:
            name: Service name.
            
        Returns:
            Service instance.
        """
        if name not in self._services:
            raise ValueError(f"Service '{name}' not found. Available services: {list(self._services.keys())}")
        return self._services[name]
    
    def register_service(self, name: str, service: Any) -> None:
        """
        Register a custom service instance.
        
        Args:
            name: Service name.
            service: Service instance.
        """
        self._services[name] = service
    
    def clear_cache(self) -> None:
        """Clear cached services."""
        self._services.clear()
