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
"""
Result publishing helpers for habitat analysis.

These helpers keep CSV/image artefact publishing out of the main
``HabitatAnalysis`` lifecycle and avoid hidden in-place mutation of the result
DataFrame supplied by callers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from habit.utils.habitats_results_io import save_habitats_results

from ..config_schemas import HabitatAnalysisConfig, ResultColumns
from .habitat_image_writer import HabitatImageWriter

if TYPE_CHECKING:
    from .feature_service import FeatureService


def canonical_csv_column_order(df: pd.DataFrame) -> List[str]:
    """
    Return habitats.csv column order: metadata columns first, features after.
    """
    meta_order = [
        ResultColumns.SUBJECT,
        ResultColumns.SUPERVOXEL,
        ResultColumns.HABITATS,
        ResultColumns.COUNT,
    ]
    meta_cols = [c for c in meta_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols]
    return meta_cols + other_cols


class HabitatResultPublisher:
    """Publish habitat-analysis results as CSV and optional label images."""

    def __init__(
        self,
        config: HabitatAnalysisConfig,
        habitat_image_writer: HabitatImageWriter,
        logger: logging.Logger,
        pipeline_getter: Callable[[], Optional[Any]],
        feature_service_getter: Optional[Callable[[], Optional["FeatureService"]]] = None,
    ) -> None:
        self.config = config
        self.habitat_image_writer = habitat_image_writer
        self.logger = logger
        self._pipeline_getter = pipeline_getter
        self._feature_service_getter = feature_service_getter

    def publish(self, results_df: pd.DataFrame, mode: str) -> None:
        """Write configured result artefacts for a train or predict run."""
        if self.config.verbose:
            self.logger.info(f"Saving {mode} results...")

        self._write_csv(results_df)
        self._write_images_if_enabled(results_df)

    def _write_csv(self, results_df: pd.DataFrame) -> None:
        canonical_order = canonical_csv_column_order(results_df)
        ordered_df = results_df.loc[:, canonical_order]
        save_habitats_results(
            ordered_df,
            self.config.out_dir,
            self.config.habitats_results_format,
            logger=self.logger if self.config.verbose else None,
        )

    def _write_images_if_enabled(self, results_df: pd.DataFrame) -> None:
        clustering_mode = self.config.HabitatSegmentation.clustering_mode
        if clustering_mode == 'one_step':
            if self.config.verbose:
                self.logger.info(
                    "One-Step mode: habitat maps were saved during clustering."
                )
            return

        if not self.config.save_images:
            return

        # two_step reads existing *_supervoxel.nrrd files; only direct_pooling
        # (voxel-level rows) needs mask arrays in memory for image export.
        if clustering_mode == "direct_pooling":
            self._populate_mask_cache_for_results(results_df)
        else:
            self.habitat_image_writer.mask_info_cache = {}

        self.habitat_image_writer.results_df = results_df.copy(deep=False)
        self.habitat_image_writer.save_all_habitat_images(failed_subjects=[])

    def _populate_mask_cache_for_results(self, results_df: pd.DataFrame) -> None:
        """
        Build ``mask_info_cache`` for NRRD export without relying on pkl payloads.

        Legacy pipelines may still carry an in-memory cache; missing subjects are
        loaded on demand from ``FeatureService.mask_paths``.
        """
        pipeline = self._pipeline_getter()
        cache: Dict[str, Any] = dict(getattr(pipeline, "mask_info_cache", None) or {})

        if ResultColumns.SUBJECT not in results_df.columns:
            self.habitat_image_writer.mask_info_cache = cache
            return

        subject_ids = sorted(set(results_df[ResultColumns.SUBJECT].astype(str)))
        feature_service = (
            self._feature_service_getter()
            if self._feature_service_getter is not None
            else None
        )
        if feature_service is None:
            self.habitat_image_writer.mask_info_cache = cache
            return

        for subject_id in subject_ids:
            if subject_id in cache and cache[subject_id].get("mask_array") is not None:
                continue
            try:
                cache[subject_id] = feature_service.load_mask_info(subject_id)
            except (KeyError, ValueError, OSError) as exc:
                self.logger.warning(
                    "Failed to load mask for subject %s from data paths: %s",
                    subject_id,
                    exc,
                )

        self.habitat_image_writer.mask_info_cache = cache
