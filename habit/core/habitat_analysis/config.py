"""
Configuration dataclasses for Habitat Analysis.

This module provides structured configuration management using dataclasses,
replacing the 24+ parameters in HabitatAnalysis.__init__() with organized,
type-safe configuration objects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import logging


@dataclass
class ClusteringConfig:
    """
    Configuration for clustering algorithms and parameters.
    
    Attributes:
        strategy: Clustering strategy - 'one_step' or 'two_step'
            - one_step: Individual-level clustering only (each tumor gets its own habitats)
            - two_step: Individual + Population clustering (supervoxels then habitats)
        supervoxel_method: Method for voxel-to-supervoxel clustering (default: 'kmeans')
        habitat_method: Method for supervoxel-to-habitat clustering (default: 'kmeans')
        n_clusters_supervoxel: Number of supervoxel clusters for two_step mode
        n_clusters_habitats_min: Minimum number of habitat clusters to test
        n_clusters_habitats_max: Maximum number of habitat clusters to test
        best_n_clusters: Explicitly specify the best number of clusters (skip search)
        selection_methods: Methods for selecting optimal cluster number
        random_state: Random seed for reproducibility
    """
    strategy: str = "two_step"
    supervoxel_method: str = "kmeans"
    habitat_method: str = "kmeans"
    n_clusters_supervoxel: int = 50
    n_clusters_habitats_min: int = 2
    n_clusters_habitats_max: int = 10
    best_n_clusters: Optional[int] = None
    selection_methods: Optional[Union[str, List[str]]] = None
    random_state: int = 42
    
    def __post_init__(self):
        """Validate clustering configuration after initialization."""
        valid_strategies = ['one_step', 'two_step', 'direct_pooling']
        if self.strategy.lower() not in valid_strategies:
            raise ValueError(
                f"clustering strategy must be one of {valid_strategies}, "
                f"got '{self.strategy}'"
            )
        self.strategy = self.strategy.lower()
        
        if self.n_clusters_habitats_min < 2:
            raise ValueError("n_clusters_habitats_min must be >= 2")
        
        if self.n_clusters_habitats_max < self.n_clusters_habitats_min:
            raise ValueError(
                "n_clusters_habitats_max must be >= n_clusters_habitats_min"
            )


@dataclass
class OneStepConfig:
    """
    Configuration specific to one-step clustering mode.
    
    Attributes:
        best_n_clusters: Fixed number of clusters (if set, skip optimal search)
        min_clusters: Minimum number of clusters to test
        max_clusters: Maximum number of clusters to test  
        selection_method: Method to determine optimal clusters
        plot_validation_curves: Whether to plot validation curves for each tumor
    """
    best_n_clusters: Optional[int] = None
    min_clusters: int = 2
    max_clusters: int = 10
    selection_method: str = 'silhouette'
    plot_validation_curves: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> 'OneStepConfig':
        """Create OneStepConfig from dictionary, using defaults for missing keys."""
        if config_dict is None:
            return cls()
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class IOConfig:
    """
    Configuration for input/output paths and directories.
    
    Attributes:
        root_folder: Root directory containing the data
        out_folder: Output directory for results (default: {root_folder}/habitats_output)
        images_dir: Name of the images subdirectory
        masks_dir: Name of the masks subdirectory
        config_file: Path to the original configuration file (for copying)
    """
    root_folder: str = ""
    out_folder: Optional[str] = None
    images_dir: str = "images"
    masks_dir: str = "masks"
    config_file: Optional[str] = None
    
    def __post_init__(self):
        """Resolve paths to absolute paths."""
        if self.root_folder:
            self.root_folder = str(Path(self.root_folder).resolve())
        
        if self.out_folder is None and self.root_folder:
            self.out_folder = str(Path(self.root_folder) / "habitats_output")
        elif self.out_folder:
            self.out_folder = str(Path(self.out_folder).resolve())


@dataclass
class RuntimeConfig:
    """
    Configuration for runtime behavior and performance.
    
    Attributes:
        mode: Analysis mode - 'training' or 'testing'
        n_processes: Number of parallel processes for multiprocessing
        verbose: Whether to output detailed information
        plot_curves: Whether to generate evaluation curve plots
        save_intermediate_results: Whether to save intermediate results
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR)
        progress_callback: Optional callback function for progress updates
    """
    mode: str = "training"
    n_processes: int = 1
    verbose: bool = True
    plot_curves: bool = True
    save_intermediate_results: bool = False
    log_level: str = "INFO"
    progress_callback: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate runtime configuration."""
        valid_modes = ['training', 'testing']
        if self.mode.lower() not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        self.mode = self.mode.lower()
        
        if self.n_processes < 1:
            raise ValueError("n_processes must be >= 1")
        
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        self.log_level = self.log_level.upper()
    
    def get_log_level_int(self) -> int:
        """Convert log level string to logging module constant."""
        return getattr(logging, self.log_level)


@dataclass  
class HabitatConfig:
    """
    Master configuration container for HabitatAnalysis.
    
    This dataclass aggregates all configuration aspects into a single,
    organized structure with validation and sensible defaults.
    
    Attributes:
        clustering: Clustering algorithm configuration
        io: Input/output path configuration
        runtime: Runtime behavior configuration
        one_step: One-step mode specific configuration (optional)
        feature_config: Feature extraction configuration dictionary
    """
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    io: IOConfig = field(default_factory=IOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    one_step: Optional[OneStepConfig] = None
    feature_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize one_step config if strategy is one_step."""
        if self.clustering.strategy == 'one_step' and self.one_step is None:
            self.one_step = OneStepConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HabitatConfig':
        """
        Create HabitatConfig from a flat dictionary (backward compatible).
        
        This method maps the old flat parameter structure to the new
        nested configuration structure.
        
        Args:
            config_dict: Dictionary with flat configuration parameters
            
        Returns:
            HabitatConfig: Configured instance
        """
        # Map old parameter names to new structure
        clustering = ClusteringConfig(
            strategy=config_dict.get('clustering_strategy', 'two_step'),
            supervoxel_method=config_dict.get('supervoxel_clustering_method', 'kmeans'),
            habitat_method=config_dict.get('habitat_clustering_method', 'kmeans'),
            n_clusters_supervoxel=config_dict.get('n_clusters_supervoxel', 50),
            n_clusters_habitats_min=config_dict.get('n_clusters_habitats_min', 2),
            n_clusters_habitats_max=config_dict.get('n_clusters_habitats_max', 10),
            best_n_clusters=config_dict.get('best_n_clusters'),
            selection_methods=config_dict.get('habitat_cluster_selection_method'),
            random_state=config_dict.get('random_state', 42),
        )
        
        io = IOConfig(
            root_folder=config_dict.get('root_folder', ''),
            out_folder=config_dict.get('out_folder'),
            images_dir=config_dict.get('images_dir', 'images'),
            masks_dir=config_dict.get('masks_dir', 'masks'),
            config_file=config_dict.get('config_file'),
        )
        
        runtime = RuntimeConfig(
            mode=config_dict.get('mode', 'training'),
            n_processes=config_dict.get('n_processes', 1),
            verbose=config_dict.get('verbose', True),
            plot_curves=config_dict.get('plot_curves', True),
            save_intermediate_results=config_dict.get('save_intermediate_results', False),
            log_level=config_dict.get('log_level', 'INFO'),
            progress_callback=config_dict.get('progress_callback'),
        )
        
        one_step = None
        if config_dict.get('one_step_settings'):
            one_step = OneStepConfig.from_dict(config_dict['one_step_settings'])
        
        return cls(
            clustering=clustering,
            io=io,
            runtime=runtime,
            one_step=one_step,
            feature_config=config_dict.get('feature_config', {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a flat dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Flat dictionary representation
        """
        result = {
            # Clustering config
            'clustering_strategy': self.clustering.strategy,
            'supervoxel_clustering_method': self.clustering.supervoxel_method,
            'habitat_clustering_method': self.clustering.habitat_method,
            'n_clusters_supervoxel': self.clustering.n_clusters_supervoxel,
            'n_clusters_habitats_min': self.clustering.n_clusters_habitats_min,
            'n_clusters_habitats_max': self.clustering.n_clusters_habitats_max,
            'best_n_clusters': self.clustering.best_n_clusters,
            'habitat_cluster_selection_method': self.clustering.selection_methods,
            'random_state': self.clustering.random_state,
            # IO config
            'data_dir': self.io.root_folder,
            'out_folder': self.io.out_folder,
            'images_dir': self.io.images_dir,
            'masks_dir': self.io.masks_dir,
            'config_file': self.io.config_file,
            # Runtime config
            'mode': self.runtime.mode,
            'n_processes': self.runtime.n_processes,
            'verbose': self.runtime.verbose,
            'plot_curves': self.runtime.plot_curves,
            'save_intermediate_results': self.runtime.save_intermediate_results,
            # Feature config
            'feature_config': self.feature_config,
        }
        
        if self.one_step:
            result['one_step_settings'] = {
                'best_n_clusters': self.one_step.best_n_clusters,
                'min_clusters': self.one_step.min_clusters,
                'max_clusters': self.one_step.max_clusters,
                'selection_method': self.one_step.selection_method,
                'plot_validation_curves': self.one_step.plot_validation_curves,
            }
        
        return result


# Column name constants to avoid magic strings
class ResultColumns:
    """
    Constants for result DataFrame column names.
    
    Using these constants instead of hardcoded strings improves
    maintainability and reduces errors from typos.
    """
    SUBJECT = "Subject"
    SUPERVOXEL = "Supervoxel"
    COUNT = "Count"
    HABITATS = "Habitats"
    
    # Suffix for original (non-processed) feature columns
    ORIGINAL_SUFFIX = "-original"
    
    @classmethod
    def metadata_columns(cls) -> List[str]:
        """Return list of metadata column names (non-feature columns)."""
        return [cls.SUBJECT, cls.SUPERVOXEL, cls.COUNT]
    
    @classmethod
    def is_feature_column(cls, col_name: str) -> bool:
        """Check if a column name represents a feature (not metadata)."""
        return (
            col_name not in cls.metadata_columns() and 
            not col_name.endswith(cls.ORIGINAL_SUFFIX)
        )
