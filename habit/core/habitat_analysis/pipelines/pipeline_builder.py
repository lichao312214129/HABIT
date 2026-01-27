"""
Pipeline builder for habitat analysis.

This module provides factory functions to build pipelines for different clustering strategies.
"""

from typing import Optional, List, Tuple
from ..config_schemas import HabitatAnalysisConfig
from ..managers.feature_manager import FeatureManager
from ..managers.clustering_manager import ClusteringManager
from ..managers.result_manager import ResultManager
from .base_pipeline import HabitatPipeline, BasePipelineStep
from .steps.voxel_feature_extractor import VoxelFeatureExtractor
from .steps.subject_preprocessing import SubjectPreprocessingStep
from .steps.individual_clustering import IndividualClusteringStep
from .steps.supervoxel_feature_extraction import SupervoxelFeatureExtractionStep
from .steps.supervoxel_aggregation import SupervoxelAggregationStep
from .steps.combine_supervoxels import CombineSupervoxelsStep
from .steps.concatenate_voxels import ConcatenateVoxelsStep
from .steps.group_preprocessing import GroupPreprocessingStep
from .steps.population_clustering import PopulationClusteringStep


def build_habitat_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager] = None,
    load_from: Optional[str] = None
) -> HabitatPipeline:
    """
    Build habitat analysis pipeline based on strategy.
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance (optional, needed for saving supervoxel maps)
        load_from: Optional path to load saved pipeline
        
    Returns:
        Configured HabitatPipeline for the specified strategy
        
    Note: No mode_handler needed - Pipeline Steps manage their own state internally.
    """
    if load_from:
        return HabitatPipeline(load_from=load_from)
    
    strategy = config.HabitatsSegmention.clustering_mode
    
    if strategy == 'two_step':
        return _build_two_step_pipeline(
            config, feature_manager, clustering_manager, result_manager
        )
    elif strategy == 'one_step':
        return _build_one_step_pipeline(
            config, feature_manager, clustering_manager, result_manager
        )
    elif strategy == 'direct_pooling':
        return _build_pooling_pipeline(
            config, feature_manager, clustering_manager, result_manager
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_two_step_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager] = None
) -> HabitatPipeline:
    """
    Build two-step strategy pipeline.
    
    This pipeline includes:
    - Step 1: Voxel feature extraction
    - Step 2: Subject-level preprocessing
    - Step 3: Individual clustering (voxel → supervoxel)
    - Step 4: Supervoxel feature extraction (conditionally)
    - Step 5: Supervoxel aggregation
    - Step 6: Group-level preprocessing
    - Step 7: Population clustering (supervoxel → habitat)
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance (for saving supervoxel maps)
        
    Returns:
        Configured HabitatPipeline for two-step strategy
    """
    if result_manager is None:
        raise ValueError("result_manager is required for two-step pipeline")
    
    steps: List[Tuple[str, BasePipelineStep]] = []
    
    # Step 1: Voxel feature extraction
    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager)
    ))
    
    # Step 2: Subject-level preprocessing
    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager)
    ))
    
    # Step 3: Individual clustering (voxel → supervoxel)
    steps.append((
        'individual_clustering',
        IndividualClusteringStep(
            feature_manager=feature_manager,
            clustering_manager=clustering_manager,
            result_manager=result_manager,
            config=config,
            target='supervoxel'  # Cluster to supervoxel
        )
    ))
    
    # Step 4: Conditionally add Supervoxel Feature Extraction
    # Check if advanced supervoxel features are needed
    supervoxel_config = config.FeatureConstruction.supervoxel_level
    method = supervoxel_config.method if supervoxel_config else None
    should_extract_advanced = (
        method is not None and 
        'mean_voxel_features' not in method
    )
    
    if should_extract_advanced:
        # Add Step 4: Extract advanced features from supervoxel maps
        steps.append((
            'supervoxel_feature_extraction', 
            SupervoxelFeatureExtractionStep(feature_manager, config)
        ))
    
    # Step 5: Always add Supervoxel Aggregation (Individual-level)
    # This step calculates supervoxel features for each subject independently
    steps.append((
        'supervoxel_aggregation',
        SupervoxelAggregationStep(feature_manager, config)
    ))
    
    # Step 5.5: Combine all subjects' supervoxels (Group-level)
    # This step merges individual supervoxel DataFrames into one
    steps.append((
        'combine_supervoxels',
        CombineSupervoxelsStep()
    ))
    
    # Step 6: Group-level preprocessing
    if config.FeatureConstruction.preprocessing_for_group_level:
        methods = config.FeatureConstruction.preprocessing_for_group_level.methods
        if methods:
            steps.append((
                'group_preprocessing',
                GroupPreprocessingStep(
                    methods=methods,
                    out_dir=config.out_dir
                )
            ))
    
    # Step 7: Population clustering
    steps.append((
        'population_clustering',
        PopulationClusteringStep(
            clustering_manager=clustering_manager,
            config=config,
            out_dir=config.out_dir
        )
    ))
    
    # Return unified pipeline
    # Memory usage is controlled by the `processes` parameter in individual-level steps
    return HabitatPipeline(steps=steps, config=config)


def _build_one_step_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager] = None
) -> HabitatPipeline:
    """
    Build one-step strategy pipeline.
    
    This pipeline includes:
    - Step 1: Voxel feature extraction
    - Step 2: Subject-level preprocessing
    - Step 3: Individual clustering (voxel → habitat, per subject)
    - No group-level steps
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance (optional, for saving habitat maps)
        
    Returns:
        Configured HabitatPipeline for one-step strategy
    """
    steps: List[Tuple[str, BasePipelineStep]] = []
    
    # Step 1: Voxel feature extraction
    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager)
    ))
    
    # Step 2: Subject-level preprocessing
    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager)
    ))
    
    # Step 3: Individual clustering (voxel → habitat, per subject)
    # For one-step, we cluster directly to habitats
    if result_manager is None:
        # Create a minimal ResultManager if not provided
        import logging
        logger = logging.getLogger(__name__)
        result_manager = ResultManager(config, logger)
    
    steps.append((
        'individual_clustering',
        IndividualClusteringStep(
            feature_manager=feature_manager,
            clustering_manager=clustering_manager,
            result_manager=result_manager,
            config=config,
            target='habitat',  # Cluster directly to habitat
            find_optimal=True  # Find optimal cluster number per subject
        )
    ))
    
    # Step 4: Aggregate features (calculate means) - Individual-level
    # Even for one-step, we need to return a DataFrame with mean features
    steps.append((
        'supervoxel_aggregation',
        SupervoxelAggregationStep(feature_manager, config)
    ))
    
    # Step 5: Combine all subjects' habitats - Group-level
    steps.append((
        'combine_supervoxels',
        CombineSupervoxelsStep()
    ))
    
    return HabitatPipeline(steps=steps, config=config)


def _build_pooling_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager] = None
) -> HabitatPipeline:
    """
    Build direct pooling strategy pipeline.
    
    This pipeline includes:
    - Step 1: Voxel feature extraction
    - Step 2: Subject-level preprocessing
    - Step 3: Concatenate all voxels
    - Step 4: Group-level preprocessing
    - Step 5: Population clustering (all voxels → habitat)
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance (optional, not used in pooling)
        
    Returns:
        Configured HabitatPipeline for direct pooling strategy
    """
    steps: List[Tuple[str, BasePipelineStep]] = []
    
    # Step 1: Voxel feature extraction
    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager)
    ))
    
    # Step 2: Subject-level preprocessing
    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager)
    ))
    
    # Step 3: Concatenate all voxels
    steps.append((
        'concatenate_voxels',
        ConcatenateVoxelsStep()
    ))
    
    # Step 4: Group-level preprocessing
    if config.FeatureConstruction.preprocessing_for_group_level:
        methods = config.FeatureConstruction.preprocessing_for_group_level.methods
        if methods:
            steps.append((
                'group_preprocessing',
                GroupPreprocessingStep(
                    methods=methods,
                    out_dir=config.out_dir
                )
            ))
    
    # Step 5: Population clustering
    steps.append((
        'population_clustering',
        PopulationClusteringStep(
            clustering_manager=clustering_manager,
            config=config,
            out_dir=config.out_dir
        )
    ))
    
    return HabitatPipeline(steps=steps, config=config)
