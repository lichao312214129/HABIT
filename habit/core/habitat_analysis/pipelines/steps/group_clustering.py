"""
Group-level clustering step for habitat analysis pipeline.

Clusters pooled cohort-level features (typically supervoxel rows) into habitat labels.
Stateful: ``fit()`` trains the clustering model; ``transform()`` applies it.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from habit.utils.log_utils import get_module_logger

from ..base_pipeline import GroupLevelStep
from ...clustering.base_clustering import ClusteringAlgorithmFactory
from ...config_schemas import HabitatAnalysisConfig, ResultColumns


class GroupClusteringStep(GroupLevelStep):
    """
    Group-level clustering (supervoxel rows to habitat labels).

    Stateful: ``fit()`` trains the clustering model; ``transform()`` applies it.

    This step manages the clustering model internally (no separate orchestration Mode).

    Attributes:
        clustering_service: ClusteringService instance (algorithm instances).
        config: Habitat analysis configuration.
        out_dir: Output directory for saving artefacts when applicable.
        clustering_model: Fitted clustering model (trained in ``fit()``).
        optimal_n_clusters_: Selected cluster count.
        fitted_: Whether ``fit()`` has completed successfully.
    """

    def __init__(
        self,
        clustering_service: Any,
        config: HabitatAnalysisConfig,
        out_dir: str,
    ) -> None:
        """
        Args:
            clustering_service: ClusteringService instance.
            config: Validated habitat-analysis configuration.
            out_dir: Output directory for saved models or diagnostics.
        """
        super().__init__()
        self.clustering_service = clustering_service
        self.config = config
        self.out_dir = out_dir
        self.clustering_model = None  # Assigned in fit()
        self.optimal_n_clusters_: Optional[int] = None
        self.logger = get_module_logger(__name__)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
        **fit_params: Any,
    ) -> "GroupClusteringStep":
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
        self.logger.info(
            f"GroupClusteringStep.fit: input rows={len(feature_matrix)}, "
            f"feature_columns={feature_matrix.shape[1]}"
        )

        # Find optimal number of clusters
        optimal_n, scores = self._find_optimal_clusters(feature_matrix)
        self.optimal_n_clusters_ = optimal_n
        self.habitat_scores_ = scores
        
        # Get clustering algorithm instance from manager
        # Use the supervoxel2habitat_clustering instance
        self.clustering_model = self.clustering_service.supervoxel2habitat_clustering
        
        # Set optimal number of clusters and fit
        self.clustering_model.n_clusters = optimal_n
        self.clustering_model.fit(feature_matrix)

        self.logger.info(
            f"GroupClusteringStep.fit: fitted with {optimal_n} habitat cluster(s)"
        )
        
        # Predict on training data (for consistency)
        # Note: This is optional, but useful for debugging
        if self.config.verbose:
            self.clustering_service.logger.info(
                f"Fitted group-level clustering model with {optimal_n} clusters"
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
            self.clustering_service.plot_habitat_scores(
                scores=self.habitat_scores_,
                optimal_n_clusters=self.optimal_n_clusters_
            )

        if self.config.plot_curves:
            self.clustering_service.visualize_habitat_clustering(
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
        habitat_cfg = self.config.HabitatSegmentation.habitat
        
        # Check if fixed_n_clusters is specified (disables automatic selection)
        if habitat_cfg.fixed_n_clusters is not None:
            optimal_n_clusters = habitat_cfg.fixed_n_clusters
            if self.config.verbose:
                self.clustering_service.logger.info(
                    f"Using fixed number of clusters: {optimal_n_clusters}"
                )
            return optimal_n_clusters, None
        
        # Find optimal number of clusters
        if self.config.verbose:
            self.clustering_service.logger.info("Finding optimal number of clusters...")
        
        try:
            min_clusters = max(2, habitat_cfg.min_clusters or 2)
            max_clusters = min(
                habitat_cfg.max_clusters,
                len(features) - 1
            )
            
            if max_clusters <= min_clusters:
                if self.config.verbose:
                    self.clustering_service.logger.warning(
                        f"Invalid cluster range [{min_clusters}, {max_clusters}], "
                        "using default value"
                    )
                return min_clusters, None
            
            selection_methods = (
                self.clustering_service.selection_methods
                or habitat_cfg.habitat_cluster_selection_method
            )

            # Create a validation model with the same core parameters used later
            # for final fitting, so model selection reflects the configured model.
            cluster_for_best_n = ClusteringAlgorithmFactory.create_algorithm(
                habitat_cfg.algorithm,
                random_state=self.config.effective_habitat_random_state(),
                max_iter=habitat_cfg.max_iter,
                n_init=habitat_cfg.n_init
            )
            
            optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                features,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                methods=selection_methods,
                show_progress=True
            )
            
            if self.config.verbose:
                self.clustering_service.logger.info(
                    f"Optimal number of clusters: {optimal_n_clusters}"
                )
            
            return optimal_n_clusters, scores
            
        except Exception as e:
            if self.config.verbose:
                self.clustering_service.logger.error(
                    f"Exception when determining optimal clusters: {e}"
                )
                self.clustering_service.logger.info("Using default number of clusters")
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
