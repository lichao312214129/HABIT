# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
#!/usr/bin/env python
"""
HabitatMapAnalyzer — orchestrator for post-clustering feature extraction.

All feature types (built-in and optional) are implemented as plugins that
inherit from HabitatFeaturePluginBase and are registered with
@HabitatFeatureRegistry.register.  This class is a pure dispatcher: it never
contains feature-type-specific logic.  Adding a new feature type requires
only creating a new plugin class — no changes here.

Built-in types:  non_radiomics, traditional, whole_habitat,
                 each_habitat, msi, ith_score
Optional types:  graph (HABIT-v2), and any future additions.
"""

import logging
import multiprocessing
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from habit.core.habitat_analysis.feature_registry import (
    BatchExportContext,
    HabitatFeaturePluginBase,
    SubjectExtractionContext,
    bootstrap_optional_plugins,
    build_plugin,
    get_default_feature_types,
    validate_feature_types,
)
from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.job_cancel import iter_until_cancelled
from habit.utils.progress_utils import CustomTqdm

from .feature_utils import FeatureUtils

warnings.filterwarnings("ignore")


class HabitatMapAnalyzer:
    """Orchestrator for extracting features from pre-computed habitat maps.

    Dispatches work to registered HabitatFeaturePluginBase instances.
    Neither process_subject() nor run() contains any feature-type-specific
    branches — all logic lives in the individual plugin classes.

    Args:
        params_file_of_non_habitat: PyRadiomics param file for the raw image.
        params_file_of_habitat: PyRadiomics param file for the habitat image.
        raw_img_folder: Root directory of raw images.
        habitats_map_folder: Root directory of habitat maps.
        out_dir: Output directory for CSVs and logs.
        n_processes: Number of worker processes; defaults to cpu_count // 2.
        habitat_pattern: Glob pattern for matching habitat map files.
        voxel_cutoff: Minimum voxel count for MSI small-region filtering.
        plugin_configs: Mapping of optional plugin name to its config object.
    """

    def __init__(
        self,
        params_file_of_non_habitat: Optional[str] = None,
        params_file_of_habitat: Optional[str] = None,
        raw_img_folder: Optional[str] = None,
        habitats_map_folder: Optional[str] = None,
        out_dir: Optional[str] = None,
        n_processes: Optional[int] = None,
        habitat_pattern: Optional[str] = None,
        voxel_cutoff: int = 10,
        plugin_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.params_file_of_non_habitat = params_file_of_non_habitat
        self.params_file_of_habitat = params_file_of_habitat
        self.raw_img_folder = raw_img_folder
        self.habitats_map_folder = habitats_map_folder
        self.out_dir = out_dir
        self._habitat_pattern = habitat_pattern
        self.voxel_cutoff = voxel_cutoff
        self.n_habitats: Optional[int] = None
        self.save_every_n_files = 5

        if n_processes is None:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.n_processes = min(n_processes, multiprocessing.cpu_count() - 2)

        # Build the unified plugin map: built-in first, then optional.
        self._all_plugins: Dict[str, HabitatFeaturePluginBase] = {}
        self._all_plugins.update(self._build_builtin_plugins())
        self._all_plugins.update(self._build_optional_plugins(plugin_configs or {}))

        self._setup_logging()

    # ------------------------------------------------------------------
    # Plugin construction
    # ------------------------------------------------------------------

    def _build_builtin_plugins(self) -> Dict[str, HabitatFeaturePluginBase]:
        """Instantiate all built-in feature plugins with the analyzer's config.

        Returns:
            Dict mapping feature-type name to plugin instance.
        """
        from .builtin_plugins import (
            EachHabitatPlugin,
            ITHPlugin,
            MSIPlugin,
            NonRadiomicsPlugin,
            TraditionalRadiomicsPlugin,
            WholeHabitatPlugin,
        )

        return {
            "non_radiomics": NonRadiomicsPlugin(),
            "traditional": TraditionalRadiomicsPlugin(
                params_file=self.params_file_of_non_habitat
            ),
            "whole_habitat": WholeHabitatPlugin(
                params_file=self.params_file_of_habitat
            ),
            "each_habitat": EachHabitatPlugin(
                params_file=self.params_file_of_non_habitat
            ),
            "msi": MSIPlugin(voxel_cutoff=self.voxel_cutoff),
            "ith_score": ITHPlugin(),
        }

    @staticmethod
    def _build_optional_plugins(
        plugin_configs: Dict[str, Any],
    ) -> Dict[str, HabitatFeaturePluginBase]:
        """Instantiate registered optional plugins from a config mapping.

        Args:
            plugin_configs: Mapping of plugin name to its config object.

        Returns:
            Dict mapping plugin name to plugin instance.
        """
        bootstrap_optional_plugins()
        return {name: build_plugin(name, config) for name, config in plugin_configs.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure logging; reuse CLI handler when already set up."""
        from habit.utils.log_utils import LoggerManager, get_module_logger, setup_logger

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        manager = LoggerManager()
        if manager.get_log_file() is not None:
            self.logger = get_module_logger("habitat.analyzer")
            self.logger.info("Using existing logging configuration from CLI entry point")
        else:
            self.logger = setup_logger(
                name="habitat.analyzer",
                output_dir=self.out_dir,
                log_filename="processing.log",
                level=logging.INFO,
            )

        self._log_file_path = manager.get_log_file()
        self._log_level = logging.INFO
        self.logger.info("Logging setup completed")

    def _ensure_logging_in_subprocess(self) -> None:
        """Restore logging in a spawned child process (Windows compatibility)."""
        from habit.utils.log_utils import restore_logging_in_subprocess

        if logging.getLogger("habit").handlers:
            return
        if hasattr(self, "_log_file_path") and self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _get_n_habitats_from_csv(self) -> int:
        """Read habitat count from habitats.csv / habitats.parquet.

        Falls back to interactive prompt when the file is missing.

        Returns:
            Number of distinct habitat labels (positive integer).
        """
        n_habitats = FeatureUtils.get_n_habitats_from_csv(self.habitats_map_folder)
        if n_habitats is not None:
            return n_habitats

        self.logger.warning(
            "Unable to read habitat count from file; please enter manually"
        )
        while True:
            try:
                value = int(input("Please enter the number of habitats (integer): ").strip())
                if value > 0:
                    self.logger.info("User entered number of habitats: %s", value)
                    return value
                self.logger.warning("Please enter a positive integer")
            except ValueError:
                self.logger.warning("Invalid input, please enter an integer")

    def get_mask_and_raw_files(
        self,
    ) -> tuple:
        """Discover all raw-image, habitat-map, and mask file paths.

        Returns:
            Tuple of (images_paths, habitat_paths, mask_paths).
        """
        images_paths, mask_paths = get_image_and_mask_paths(self.raw_img_folder)

        habitat_paths: Dict[str, str] = {}
        for subj_path in Path(self.habitats_map_folder).glob(self._habitat_pattern):
            key = subj_path.name.replace(
                self._habitat_pattern.replace("*", ""), ""
            )
            habitat_paths[key] = str(subj_path)

        return images_paths, habitat_paths, mask_paths

    # ------------------------------------------------------------------
    # Per-subject extraction (runs inside multiprocessing workers)
    # ------------------------------------------------------------------

    def process_subject(
        self,
        subj: str,
        images_paths: Dict[str, Dict[str, str]],
        habitat_paths: Dict[str, str],
        mask_paths: Optional[Dict[str, str]] = None,
        feature_types: Optional[List[str]] = None,
        n_habitats: Optional[int] = None,
    ) -> tuple:
        """Extract all requested features for a single subject.

        Dispatches to each active plugin; all feature-type logic lives in
        the plugin classes.  Called inside a multiprocessing worker.

        Args:
            subj: Subject identifier.
            images_paths: Mapping {subject_id: {modality_name: file_path}}.
            habitat_paths: Mapping {subject_id: habitat_map_path}.
            mask_paths: Optional mapping {subject_id: {modality: mask_path}}.
            feature_types: List of feature type names to extract.
            n_habitats: Number of habitat labels (pre-computed by run()).

        Returns:
            Tuple (subj, subject_features_dict).
        """
        self._ensure_logging_in_subprocess()
        from habit.utils.log_utils import get_module_logger
        logger = get_module_logger("habitat.analyzer")

        if feature_types is None:
            feature_types = get_default_feature_types()
        validate_feature_types(feature_types)

        ctx = SubjectExtractionContext(
            subj=subj,
            habitat_path=habitat_paths[subj],
            image_paths=images_paths.get(subj, {}),
            mask_paths=(mask_paths or {}).get(subj),
            n_habitats=n_habitats,
            logger=logger,
        )

        subject_features: Dict[str, Any] = {}
        for plugin_name, plugin in self._all_plugins.items():
            if plugin_name not in feature_types:
                continue
            try:
                subject_features[plugin.subject_data_key] = plugin.extract_subject(ctx)
            except Exception as exc:
                logger.error(
                    "Error extracting %s features for subject %s: %s",
                    plugin_name, subj, exc,
                )
                subject_features[plugin.subject_data_key] = {"error": str(exc)}

        return subj, subject_features

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        images_paths: Dict[str, Dict[str, str]],
        habitat_paths: Dict[str, str],
        mask_paths: Optional[Dict[str, str]] = None,
        feature_types: Optional[List[str]] = None,
        n_habitats: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run per-subject extraction in parallel for all subjects.

        Args:
            images_paths: {subject_id: {modality_name: file_path}}.
            habitat_paths: {subject_id: habitat_map_path}.
            mask_paths: Optional {subject_id: {modality: mask_path}}.
            feature_types: Feature type names to extract.
            n_habitats: Pre-computed habitat count passed to each worker.

        Returns:
            Dict {subject_id: {subject_data_key: features}}.
        """
        features: Dict[str, Any] = {}
        subjs = list(set(images_paths.keys()) & set(habitat_paths.keys()))

        if not subjs:
            self.logger.error(
                "No matching subjects found between raw images and habitat maps"
            )
            return features

        self.logger.info(
            "Starting feature extraction for %s subjects using %s processes",
            len(subjs), self.n_processes,
        )

        process_func = partial(
            self.process_subject,
            images_paths=images_paths,
            habitat_paths=habitat_paths,
            mask_paths=mask_paths,
            feature_types=feature_types,
            n_habitats=n_habitats,
        )

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            pb = CustomTqdm(total=len(subjs), desc="Extracting Features")
            subject_iter = iter_until_cancelled(
                pool.imap_unordered(process_func, subjs),
                pool=pool,
            )
            for subj, subject_features in subject_iter:
                features[subj] = subject_features
                pb.update(1)
            pb.close()

        return features

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        feature_types: Optional[List[str]] = None,
        n_habitats: Optional[int] = None,
    ) -> "HabitatMapAnalyzer":
        """Run the complete feature extraction pipeline.

        1. Resolve n_habitats (auto-detect if not given).
        2. Discover all file paths.
        3. Extract per-subject features in parallel.
        4. Call export_batch() on each active plugin to write CSVs.

        Args:
            feature_types: Feature type names to extract.  Defaults to all
                registered types.
            n_habitats: Number of habitat labels.  Auto-detected from the
                habitat results table when omitted.

        Returns:
            Self, for method chaining.
        """
        if not self.out_dir:
            raise ValueError("Output directory must be specified to run the analysis")

        if feature_types is None:
            feature_types = get_default_feature_types()
        validate_feature_types(feature_types)

        # Resolve n_habitats once before spawning workers.
        if n_habitats is None:
            n_habitats = self._get_n_habitats_from_csv()
        self.n_habitats = n_habitats
        self.logger.info("Using habitat count: %s", n_habitats)

        images_paths, habitat_paths, mask_paths = self.get_mask_and_raw_files()

        feature_data = self.extract_features(
            images_paths, habitat_paths, mask_paths,
            feature_types=feature_types,
            n_habitats=n_habitats,
        )

        self.logger.info(
            "Feature extraction completed for %s subjects", len(feature_data)
        )
        self.data = feature_data

        # Export: loop over plugins — no feature-type-specific branching.
        export_ctx = BatchExportContext(
            out_dir=self.out_dir,
            n_habitats=n_habitats,
            habitat_paths=habitat_paths,
            logger=self.logger,
            n_processes=self.n_processes,
        )

        for plugin_name, plugin in self._all_plugins.items():
            if plugin_name not in feature_types:
                continue
            try:
                plugin.export_batch(feature_data, export_ctx)
            except Exception as exc:
                self.logger.error(
                    "Error exporting %s features: %s", plugin_name, exc
                )
            if plugin.should_visualize():
                try:
                    plugin.visualize_batch(feature_data, export_ctx)
                except Exception as exc:
                    self.logger.error(
                        "Error visualising %s features: %s", plugin_name, exc
                    )

        return self
