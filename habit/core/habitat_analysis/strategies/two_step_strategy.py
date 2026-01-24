"""
Two-step strategy: voxel -> supervoxel -> habitat clustering.
Refactored to use HabitatPipeline.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
import pandas as pd

from .base_strategy import BaseClusteringStrategy
from ..pipelines.pipeline_builder import build_habitat_pipeline
from ..pipelines.base_pipeline import HabitatPipeline

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class TwoStepStrategy(BaseClusteringStrategy):
    """
    Two-step clustering strategy using HabitatPipeline.

    Flow:
    1) Voxel feature extraction (Pipeline Step 1)
    2) Subject-level preprocessing (Pipeline Step 2)
    3) Individual clustering (voxel -> supervoxel) (Pipeline Step 3)
    4) Supervoxel feature extraction (conditional) (Pipeline Step 4)
    5) Supervoxel feature aggregation (Pipeline Step 5)
    6) Group-level preprocessing (Pipeline Step 6)
    7) Population clustering (supervoxel -> habitat) (Pipeline Step 7)
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize two-step strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)
        self.pipeline: Optional[HabitatPipeline] = None

    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None,
        load_from: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute two-step clustering using Pipeline.

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

        # Prepare input data for pipeline (Dict of subject_id -> empty dict).
        # The pipeline steps (VoxelFeatureExtractor) will use feature_manager to get paths.
        X = self._build_input(subjects)
        if not X:
            raise ValueError(
                "Prediction/training input is empty. "
                "Verify data_dir contains valid images and masks, "
                "or pass an explicit non-empty subjects list."
            )

        pipeline_path = self._resolve_pipeline_path(load_from)
        
        # Ensure output directory exists
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

        if load_from:
            if self.config.verbose:
                self.logger.info("Loading and running Two-Step pipeline...")
            
            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Saved pipeline not found at {pipeline_path}. "
                    "Provide a valid load_from path or run without load_from to train."
                )
            
            # Load pipeline
            # Note: We load the pipeline structure and state from file.
            # If managers include environment-specific paths, ensure they match the current environment.
            self.pipeline = HabitatPipeline.load(str(pipeline_path))
            
            # Update references in loaded pipeline to use current analysis instances.
            # This ensures that config changes (like out_dir, plot_curves) are reflected in all steps.
            self._update_pipeline_references(self.pipeline)
            
            # Disable image outputs and plots for prediction runs to avoid unnecessary I/O.
            self.pipeline.config.plot_curves = False

            # Transform
            self.analysis.results_df = self.pipeline.transform(X)

            if save_results_csv:
                self._save_results()
        else:
            if self.config.verbose:
                self.logger.info("Building and fitting Two-Step pipeline...")
            
            # Build new pipeline
            self.pipeline = build_habitat_pipeline(
                config=self.config,
                feature_manager=self.analysis.feature_manager,
                clustering_manager=self.analysis.clustering_manager,
                result_manager=self.analysis.result_manager
            )
            
            # Fit and transform
            # fit_transform will execute all steps, training stateful steps (6 & 7)
            self.analysis.results_df = self.pipeline.fit_transform(X)
            
            # Save pipeline (including trained models)
            if self.config.verbose:
                self.logger.info(f"Saving fitted pipeline to {pipeline_path}")
            self.pipeline.save(str(pipeline_path))

        # Update ResultManager with new results
        self.analysis.result_manager.results_df = self.analysis.results_df

        # Save results (CSV files and habitat images)
        if save_results_csv:
            self._save_results()

        return self.analysis.results_df

    def _save_results(self) -> None:
        """
        Save all results including config, CSV, and habitat images.
        """
        if self.config.verbose:
            self.logger.info("Saving results...")
        
        # Save results CSV
        csv_path = Path(self.config.out_dir) / "habitats.csv"
        self.analysis.results_df.to_csv(str(csv_path), index=False)
        if self.config.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        if self.config.save_images:
            # Save habitat images for each subject.
            # Note: Pipeline generated the 'Habitats' column in results_df.
            # We need to map these back to images.
            # This assumes supervoxel maps were saved during IndividualClusteringStep (Step 3).
            self.analysis.result_manager.save_all_habitat_images(failed_subjects=[])

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
            raise ValueError("No subjects provided for two-step strategy.")

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