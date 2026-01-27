"""
Population-level clustering step for habitat analysis pipeline.

This step manages clustering model internally for population-level habitat identification.

Note: This module should not be run directly. Import it as part of the package:
    from habit.core.habitat_analysis.pipelines.steps import PopulationClusteringStep
"""

from typing import Any, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import logging

try:
    from ..base_pipeline import GroupLevelStep
    from ...algorithms.base_clustering import get_clustering_algorithm
    from ...config_schemas import HabitatAnalysisConfig, ResultColumns
except ImportError as e:
    # Provide helpful error message if imported incorrectly
    import sys
    if __name__ == "__main__":
        print("Error: This module cannot be run directly.")
        print("Please import it as part of the package:")
        print("  from habit.core.habitat_analysis.pipelines.steps import PopulationClusteringStep")
        sys.exit(1)
    raise


class PopulationClusteringStep(GroupLevelStep):
    """
    Population-level clustering (supervoxel → habitat).
    
    Stateful: fit() trains clustering model, transform() applies to new data.
    
    Note: This step manages clustering model internally, no need for external Mode classes.
    
    Attributes:
        clustering_manager: ClusteringManager instance (for accessing algorithm instances)
        config: Configuration object
        out_dir: Output directory for saving model (if needed)
        clustering_model: Fitted clustering model
        optimal_n_clusters_: Optimal number of clusters found during fit
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(
        self, 
        clustering_manager: Any, 
        config: HabitatAnalysisConfig,
        out_dir: str
    ):
        """
        Initialize population clustering step.
        
        Args:
            clustering_manager: ClusteringManager instance
            config: Configuration object
            out_dir: Output directory for saving model (if needed)
        """
        super().__init__()
        self.clustering_manager = clustering_manager
        self.config = config
        self.out_dir = out_dir
        self.clustering_model = None  # Will be created in fit()
        self.optimal_n_clusters_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None, **fit_params) -> 'PopulationClusteringStep':
        """
        Fit clustering model on data (train model and find optimal clusters).
        
        Args:
            X: Preprocessed supervoxel features DataFrame
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        feature_matrix = self._extract_feature_matrix(X)

        # Find optimal number of clusters
        optimal_n, scores = self._find_optimal_clusters(feature_matrix)
        self.optimal_n_clusters_ = optimal_n
        self.habitat_scores_ = scores
        
        # Get clustering algorithm instance from manager
        # Use the supervoxel2habitat_clustering instance
        self.clustering_model = self.clustering_manager.supervoxel2habitat_clustering
        
        # Set optimal number of clusters and fit
        self.clustering_model.n_clusters = optimal_n
        self.clustering_model.fit(feature_matrix)
        
        # Predict on training data (for consistency)
        # Note: This is optional, but useful for debugging
        if self.config.verbose:
            self.clustering_manager.logger.info(
                f"Fitted population clustering model with {optimal_n} clusters"
            )
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted clustering model to new data.
        
        Args:
            X: Preprocessed supervoxel features DataFrame
            
        Returns:
            DataFrame with habitat labels added
            
        Raises:
            ValueError: If step has not been fitted
        """
        if not self.fitted_:
            raise ValueError(
                "Must fit before transform. "
                "Either call fit() first, or load a saved pipeline."
            )
        
        if self.clustering_model is None:
            raise ValueError(
                "Clustering model is None. "
                "This should not happen if the step was properly fitted."
            )
        
        # Apply fitted model
        feature_matrix = self._extract_feature_matrix(X)
        habitat_labels = self.clustering_model.predict(feature_matrix) + 1  # Start from 1
        result_df = X.copy()
        result_df[ResultColumns.HABITATS] = habitat_labels

        if self.config.plot_curves and self.habitat_scores_ is not None:
            self.clustering_manager.plot_habitat_scores(
                scores=self.habitat_scores_,
                optimal_n_clusters=self.optimal_n_clusters_
            )

        if self.config.plot_curves:
            self.clustering_manager.visualize_habitat_clustering(
                feature_matrix.values,
                habitat_labels,
                self.optimal_n_clusters_
            )
        
        return result_df
    
    def _find_optimal_clusters(
        self, 
        features: pd.DataFrame
    ) -> Tuple[int, Optional[Dict]]:
        """
        Find optimal number of clusters using validation methods.
        
        This method is adapted from legacy optimal-cluster selection logic.
        
        Args:
            features: DataFrame of supervoxel features for clustering
            
        Returns:
            Tuple of (optimal_n_clusters, scores_dict)
        """
        habitat_cfg = self.config.HabitatsSegmention.habitat
        
        # Check if fixed_n_clusters is specified (disables automatic selection)
        if habitat_cfg.fixed_n_clusters is not None:
            optimal_n_clusters = habitat_cfg.fixed_n_clusters
            if self.config.verbose:
                self.clustering_manager.logger.info(
                    f"Using fixed number of clusters: {optimal_n_clusters}"
                )
            return optimal_n_clusters, None
        
        # Find optimal number of clusters
        if self.config.verbose:
            self.clustering_manager.logger.info("Finding optimal number of clusters...")
        
        try:
            min_clusters = max(2, habitat_cfg.min_clusters or 2)
            max_clusters = min(
                habitat_cfg.max_clusters,
                len(features) - 1
            )
            
            if max_clusters <= min_clusters:
                if self.config.verbose:
                    self.clustering_manager.logger.warning(
                        f"Invalid cluster range [{min_clusters}, {max_clusters}], "
                        "using default value"
                    )
                return min_clusters, None
            
            # Create new clustering algorithm for optimization
            cluster_for_best_n = get_clustering_algorithm(
                habitat_cfg.algorithm
            )
            
            optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                features,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                methods=habitat_cfg.habitat_cluster_selection_method,
                show_progress=True
            )
            
            if self.config.verbose:
                self.clustering_manager.logger.info(
                    f"Optimal number of clusters: {optimal_n_clusters}"
                )
            
            return optimal_n_clusters, scores
            
        except Exception as e:
            if self.config.verbose:
                self.clustering_manager.logger.error(
                    f"Exception when determining optimal clusters: {e}"
                )
                self.clustering_manager.logger.info("Using default number of clusters")
            return 3, None

    def _extract_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numeric feature columns and exclude metadata columns.

        Args:
            df: Input DataFrame that may include metadata columns

        Returns:
            DataFrame containing only numeric feature columns
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if ResultColumns.is_feature_column(col)]
        if not feature_cols:
            raise ValueError("No numeric feature columns found for clustering.")
        return df[feature_cols]
