"""
Clustering Manager for Habitat Analysis.
Handles clustering algorithms, model selection, and validation.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

from ..config import ResultColumns
from ..config_schemas import HabitatAnalysisConfig
from ..algorithms.base_clustering import get_clustering_algorithm
from ..algorithms.cluster_validation_methods import (
    get_validation_methods,
    is_valid_method_for_algorithm,
    get_default_methods
)

# Visualization imports
try:
    from habit.utils.visualization import plot_cluster_scores, plot_cluster_results
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

class ClusteringManager:
    """
    Manages clustering operations for habitat analysis.
    """
    
    def __init__(self, config: HabitatAnalysisConfig, logger: logging.Logger):
        """
        Initialize ClusteringManager.
        
        Args:
            config: Habitat analysis configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        self.voxel2supervoxel_clustering = None
        self.supervoxel2habitat_clustering = None
        self.selection_methods = None
        
        self._init_clustering_algorithms()
        self._init_selection_methods()

    def _init_clustering_algorithms(self) -> None:
        """Initialize clustering algorithm instances."""
        supervoxel_cfg = self.config.HabitatsSegmention.supervoxel
        habitat_cfg = self.config.HabitatsSegmention.habitat
        self.voxel2supervoxel_clustering = get_clustering_algorithm(
            supervoxel_cfg.algorithm,
            n_clusters=supervoxel_cfg.n_clusters,
            random_state=supervoxel_cfg.random_state
        )
        
        self.supervoxel2habitat_clustering = get_clustering_algorithm(
            habitat_cfg.algorithm,
            n_clusters=habitat_cfg.max_clusters,
            random_state=habitat_cfg.random_state
        )

    def _init_selection_methods(self) -> None:
        """Initialize and validate cluster selection methods."""
        habitat_cfg = self.config.HabitatsSegmention.habitat
        validation_info = get_validation_methods(habitat_cfg.algorithm)
        valid_methods = list(validation_info['methods'].keys())
        default_methods = get_default_methods(habitat_cfg.algorithm)
        
        if self.config.verbose:
            self.logger.info(
                f"Validation methods supported by '{habitat_cfg.algorithm}': "
                f"{', '.join(valid_methods)}"
            )
            self.logger.info(f"Default validation methods: {', '.join(default_methods)}")
        
        # Validate and set selection methods
        selection_methods = habitat_cfg.habitat_cluster_selection_method
        
        if selection_methods is None:
            self.selection_methods = default_methods
            if self.config.verbose:
                self.logger.info(
                    f"No clustering evaluation method specified, "
                    f"using defaults: {', '.join(default_methods)}"
                )
        elif isinstance(selection_methods, str):
            if is_valid_method_for_algorithm(
                habitat_cfg.algorithm,
                selection_methods.lower()
            ):
                self.selection_methods = selection_methods.lower()
            else:
                self.selection_methods = default_methods
                if self.config.verbose:
                    self.logger.warning(
                        f"Validation method '{selection_methods}' is invalid for "
                        f"'{habitat_cfg.algorithm}'"
                    )
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        elif isinstance(selection_methods, list):
            valid = [
                m.lower() for m in selection_methods 
                if is_valid_method_for_algorithm(
                    habitat_cfg.algorithm, m.lower()
                )
            ]
            invalid = [
                m for m in selection_methods 
                if not is_valid_method_for_algorithm(
                    habitat_cfg.algorithm, m.lower()
                )
            ]
            
            if valid:
                self.selection_methods = valid
                if invalid and self.config.verbose:
                    self.logger.warning(
                        f"Invalid methods for '{habitat_cfg.algorithm}': "
                        f"{', '.join(invalid)}"
                    )
            else:
                self.selection_methods = default_methods
                if self.config.verbose:
                    self.logger.warning("All specified methods are invalid")
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        else:
            self.selection_methods = default_methods

    def cluster_subject_voxels(
        self, 
        subject: str, 
        feature_df: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Cluster voxels to supervoxels for a single subject.
        
        Args:
            subject: Subject ID
            feature_df: Feature DataFrame
            n_clusters: Number of clusters (if None, uses config default)
            
        Returns:
            Array of supervoxel labels (1-indexed)
        """
        try:
            if n_clusters is not None:
                # Custom number of clusters (e.g., for one-step optimal clusters)
                supervoxel_cfg = self.config.HabitatsSegmention.supervoxel
                clusterer = get_clustering_algorithm(
                    supervoxel_cfg.algorithm,
                    n_clusters=n_clusters,
                    random_state=supervoxel_cfg.random_state
                )
                clusterer.fit(feature_df.values)
                supervoxel_labels = clusterer.predict(feature_df.values)
            else:
                # Use default clustering instance
                self.voxel2supervoxel_clustering.fit(feature_df.values)
                supervoxel_labels = self.voxel2supervoxel_clustering.predict(feature_df.values)
            
            supervoxel_labels += 1  # 1-indexed
            return supervoxel_labels
            
        except Exception as e:
            self.logger.error(
                f"Error performing supervoxel clustering for subject {subject}: {e}"
            )
            raise

    def find_optimal_clusters_for_subject(
        self, 
        subject: str, 
        feature_df: pd.DataFrame,
        min_clusters: int,
        max_clusters: int,
        selection_method: str,
        plot_validation: bool = False
    ) -> int:
        """
        Find optimal number of clusters for a single subject.
        Atomic capability for OneStep strategy.
        
        Args:
            subject: Subject ID
            feature_df: Feature DataFrame
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            selection_method: Validation method to use
            plot_validation: Whether to plot validation curves
            
        Returns:
            Optimal number of clusters
        """
        self.logger.info(
            f"Determining optimal clusters for {subject} using {selection_method}"
        )
        
        supervoxel_cfg = self.config.HabitatsSegmention.supervoxel
        clusterer = get_clustering_algorithm(
            supervoxel_cfg.algorithm,
            n_clusters=max_clusters,
            random_state=supervoxel_cfg.random_state
        )
        
        optimal_n_clusters, scores_dict = clusterer.find_optimal_clusters(
            X=feature_df.values,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            methods=[selection_method],
            show_progress=False
        )
        
        # Plot validation curves if requested
        if plot_validation and self.config.plot_curves:
            self.plot_one_step_validation(subject, scores_dict, clusterer, selection_method)
        
        self.logger.info(f"Subject {subject}: optimal clusters = {optimal_n_clusters}")
        
        return optimal_n_clusters

    def plot_one_step_validation(
        self, 
        subject: str, 
        scores_dict: Dict, 
        clusterer: Any,
        selection_method: str
    ) -> None:
        """Plot validation curves for one-step clustering."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.out_dir, 'visualizations', 'optimal_clusters'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            plot_file = os.path.join(viz_dir, f'{subject}_cluster_validation.png')
            plot_cluster_scores(
                scores_dict=scores_dict,
                cluster_range=clusterer.cluster_range,
                methods=[selection_method],
                clustering_algorithm=self.config.HabitatsSegmention.supervoxel.algorithm,
                figsize=(8, 6),
                save_path=plot_file,
                show=False,
                dpi=600
            )
            self.logger.info(f"Validation plot saved to: {plot_file}")
        except Exception as e:
            self.logger.warning(f"Failed to plot validation curves for {subject}: {e}")

    def plot_habitat_scores(self, scores: Dict, optimal_n_clusters: int) -> None:
        """Plot habitat clustering validation scores."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            os.makedirs(self.config.out_dir, exist_ok=True)
            
            # Get cluster range from configuration
            min_clusters = self.config.HabitatsSegmention.habitat.min_clusters or 2
            max_clusters = self.config.HabitatsSegmention.habitat.max_clusters
            cluster_range = list(range(min_clusters, max_clusters + 1))
            
            plot_cluster_scores(
                scores_dict=scores,
                cluster_range=cluster_range,
                methods=self.selection_methods,
                clustering_algorithm=self.config.HabitatsSegmention.habitat.algorithm,
                figsize=(6, 6),
                outdir=self.config.out_dir,
                show=False
            )
            
            if self.config.verbose:
                self.logger.info(f"Clustering scores plot saved to {self.config.out_dir}")
                
        except Exception as e:
            if self.config.verbose:
                self.logger.error(f"Error plotting clustering scores: {e}")
                self.logger.info("Continuing with other processes...")

    def visualize_supervoxel_clustering(
        self,
        subject: str,
        feature_df: pd.DataFrame,
        supervoxel_labels: np.ndarray
    ) -> None:
        """Create visualizations for supervoxel clustering results."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.out_dir, 'visualizations', 'supervoxel_clustering'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            centers = None
            if hasattr(self.voxel2supervoxel_clustering, 'cluster_centers_'):
                centers = self.voxel2supervoxel_clustering.cluster_centers_
            
            title = (
                f'Supervoxel Clustering: {subject}\n'
                f'(n_clusters={self.config.HabitatsSegmention.supervoxel.n_clusters})'
            )
            
            # 2D scatter
            plot_cluster_results(
                X=feature_df.values,
                labels=supervoxel_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, f'{subject}_supervoxel_clustering_2D.png'),
                show=False,
                dpi=600,
                plot_3d=False
            )
            
            # 3D scatter
            plot_cluster_results(
                X=feature_df.values,
                labels=supervoxel_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, f'{subject}_supervoxel_clustering_3D.png'),
                show=False,
                dpi=600,
                plot_3d=True
            )
            
            if self.config.verbose:
                self.logger.info(f"Saved supervoxel clustering visualizations to {viz_dir}")
                
        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to create visualization for {subject}: {e}")

    def visualize_habitat_clustering(
        self,
        features: pd.DataFrame,
        habitat_labels: np.ndarray,
        optimal_n_clusters: int
    ) -> None:
        """Create visualizations for habitat clustering results."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.out_dir, 'visualizations', 'habitat_clustering'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            centers = None
            if hasattr(self.supervoxel2habitat_clustering, 'cluster_centers_'):
                centers = self.supervoxel2habitat_clustering.cluster_centers_
            
            title = (
                f'Habitat Clustering (Population Level)\n'
                f'(n_clusters={optimal_n_clusters})'
            )
            
            # 2D scatter
            plot_cluster_results(
                X=features,
                labels=habitat_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, 'habitat_clustering_2D.png'),
                show=False,
                dpi=600,
                plot_3d=False
            )
            
            # 3D scatter
            plot_cluster_results(
                X=features,
                labels=habitat_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, 'habitat_clustering_3D.png'),
                show=False,
                dpi=600,
                plot_3d=True
            )
            
            if self.config.verbose:
                self.logger.info(f"Saved habitat clustering visualizations to {viz_dir}")
                
        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to create habitat clustering visualization: {e}")
