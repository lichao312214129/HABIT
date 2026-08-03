# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Built-in habitat feature extraction handlers.

Each class wraps an existing computation class and implements
BaseHabitatFeature, enabling uniform dispatch in HabitatMapAnalyzer.

Registered names (used in YAML feature_types lists):
    - non_radiomics   : region count and volume ratio per habitat label
    - traditional     : PyRadiomics on the raw image within the whole ROI
    - whole_habitat   : PyRadiomics on the multi-label habitat map itself
    - each_habitat    : PyRadiomics on the raw image per individual habitat
    - msi             : Multiregional spatial interaction (MSI) features
    - ith_score       : Intratumoral Heterogeneity score

To add a new feature type, create a new subclass here (or in a separate file),
decorate it with @HabitatFeatureFactory.register('your_name'), and implement
extract_subject() + export_batch().  No changes to HabitatMapAnalyzer needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from habit.core.habitat_analysis.feature_registry import (
    BatchExportContext,
    BaseHabitatFeature,
    HabitatFeatureFactory,
    SubjectExtractionContext,
)
from habit.utils.progress_utils import CustomTqdm

from .basic_features import BasicFeatureExtractor
from .feature_utils import FeatureUtils
from .habitat_radiomics import HabitatRadiomicsExtractor
from .ith_features import ITHFeatureExtractor
from .msi_features import MSIFeatureExtractor


# ---------------------------------------------------------------------------
# non_radiomics: region count + volume ratio per habitat label
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("non_radiomics")
class NonRadiomicsFeature(BaseHabitatFeature):
    """Basic spatial features: disconnected region count and volume ratio.

    Wraps BasicFeatureExtractor.
    """

    subject_data_key = "non_radiomics_features"
    output_csv_name = "habitat_basic_features.csv"
    progress_desc = "Basic Habitat Features"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for basic habitat features."""
        return "non_radiomics"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._extractor = BasicFeatureExtractor()

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract region count and volume ratio for each habitat label.

        Args:
            ctx: Per-subject context; only ctx.habitat_path is needed.

        Returns:
            Dict with 'num_habitats' and per-label 'num_regions'/'volume_ratio'.
        """
        return self._extractor.get_non_radiomics_features(ctx.habitat_path)

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[pd.DataFrame]:
        """Aggregate non-radiomics features across all subjects and save CSV.

        Args:
            data: Per-subject feature dict.
            ctx: Batch context with out_dir, n_habitats, logger.

        Returns:
            DataFrame with one row per subject, or None on failure.
        """
        subjs = list(data.keys())
        n_habitats = ctx.n_habitats

        col_num_regions = [f"{i}_num_regions" for i in range(1, n_habitats + 1)]
        col_volume_ratio = [f"{i}_volume_ratio" for i in range(1, n_habitats + 1)]
        columns = ["num_habitats"] + col_num_regions + col_volume_ratio
        df = pd.DataFrame(index=subjs, columns=columns)

        pb = CustomTqdm(total=len(subjs), desc=self.progress_desc)
        for subj in subjs:
            pb.update(1)
            try:
                features = data[subj].get(self.subject_data_key, {})
                df.loc[subj, "num_habitats"] = features.get("num_habitats", 0)
                for hab_id in range(1, n_habitats + 1):
                    hab = features.get(hab_id, {})
                    df.loc[subj, f"{hab_id}_num_regions"] = hab.get("num_regions", 0)
                    df.loc[subj, f"{hab_id}_volume_ratio"] = hab.get("volume_ratio", 0.0)
            except Exception as exc:
                ctx.logger.error(
                    "Error exporting basic features for subject %s: %s", subj, exc
                )
                df.loc[subj, :] = np.nan
        pb.close()

        out_file = os.path.join(ctx.out_dir, self.output_csv_name)
        df.to_csv(out_file, index=True)
        ctx.logger.info("Basic habitat features saved to %s", out_file)
        return df


# ---------------------------------------------------------------------------
# traditional: PyRadiomics on the raw image within the whole ROI mask
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("traditional")
class TraditionalRadiomicsFeature(BaseHabitatFeature):
    """PyRadiomics features extracted from the raw image within the ROI.

    The habitat map is binarised to form a single ROI mask.
    Wraps HabitatRadiomicsExtractor.extract_tranditional_radiomics().
    """

    subject_data_key = "tranditional_radiomics_features"
    output_csv_name = "raw_image_radiomics.csv"
    progress_desc = "Traditional Radiomics"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for traditional radiomics."""
        return "traditional"

    def __init__(self, params_file: Optional[str] = None) -> None:
        """
        Args:
            params_file: Path to the PyRadiomics parameter YAML for the raw image.
        """
        super().__init__()
        self.params_file = params_file

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract traditional radiomics for each image modality of one subject.

        Args:
            ctx: Per-subject context; uses habitat_path and image_paths.

        Returns:
            Dict mapping image name to its radiomics feature dict.
        """
        results: Dict[str, Any] = {}
        for img_name, img_path in ctx.image_paths.items():
            try:
                results[img_name] = HabitatRadiomicsExtractor.extract_tranditional_radiomics(
                    img_path, ctx.habitat_path, ctx.subj, self.params_file
                )
            except Exception as exc:
                ctx.logger.error(
                    "Error extracting traditional radiomics for subject %s, image %s: %s",
                    ctx.subj, img_name, exc,
                )
                results[img_name] = {"error": str(exc)}
        return results

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[pd.DataFrame]:
        """Flatten per-modality radiomics into a wide CSV (one row per subject).

        Args:
            data: Per-subject feature dict.
            ctx: Batch context with out_dir and logger.

        Returns:
            Wide DataFrame or None.
        """
        subjs = list(data.keys())
        rows: list = []

        pb = CustomTqdm(total=len(subjs), desc=self.progress_desc)
        for subj in subjs:
            pb.update(1)
            try:
                modality_features = data[subj].get(self.subject_data_key, {})
                imgs = list(modality_features.keys())
                df_mod = pd.DataFrame(
                    [modality_features[img] for img in imgs], index=imgs
                )
                df_mod = df_mod.loc[:, ~df_mod.columns.str.contains("diagnostic")]
                # Flatten: column names become "{feature}_of_{modality}"
                wide_cols = [
                    f"{col}_of_{idx}"
                    for idx in df_mod.index
                    for col in df_mod.columns
                ]
                wide_df = pd.DataFrame([df_mod.values.flatten()], columns=wide_cols)
                rows.append(wide_df)
            except Exception as exc:
                ctx.logger.error(
                    "Error exporting traditional radiomics for subject %s: %s", subj, exc
                )
                if rows:
                    rows.append(FeatureUtils.create_empty_dataframe_like(rows[0], index=[0]))
        pb.close()

        if not rows:
            ctx.logger.error("No valid traditional radiomics data to export")
            return None

        result = pd.concat(rows)
        result.index = subjs
        out_file = os.path.join(ctx.out_dir, self.output_csv_name)
        result.to_csv(out_file, index=True)
        ctx.logger.info("Traditional radiomics features saved to %s", out_file)
        return result


# ---------------------------------------------------------------------------
# whole_habitat: PyRadiomics on the multi-label habitat map itself
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("whole_habitat")
class WholeHabitatFeature(BaseHabitatFeature):
    """PyRadiomics features of the habitat map treated as a single binary mask.

    Wraps HabitatRadiomicsExtractor.extract_radiomics_features_for_whole_habitat().
    """

    subject_data_key = "radiomics_features_of_whole_habitat_map"
    output_csv_name = "whole_habitat_radiomics.csv"
    progress_desc = "Whole Habitat Radiomics"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for whole-habitat radiomics."""
        return "whole_habitat"

    def __init__(self, params_file: Optional[str] = None) -> None:
        """
        Args:
            params_file: Path to the PyRadiomics parameter YAML for the habitat image.
        """
        super().__init__()
        self.params_file = params_file

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract radiomics from the whole habitat map for one subject.

        Args:
            ctx: Per-subject context; uses habitat_path.

        Returns:
            Dict of radiomics feature name → value.
        """
        return HabitatRadiomicsExtractor.extract_radiomics_features_for_whole_habitat(
            ctx.habitat_path, self.params_file
        )

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[pd.DataFrame]:
        """Save whole-habitat radiomics features to CSV.

        Args:
            data: Per-subject feature dict.
            ctx: Batch context.

        Returns:
            DataFrame or None.
        """
        subjs = list(data.keys())
        rows: list = []

        pb = CustomTqdm(total=len(subjs), desc=self.progress_desc)
        for subj in subjs:
            pb.update(1)
            try:
                features = data[subj].get(self.subject_data_key, {})
                df_row = pd.DataFrame.from_dict(features, orient="index").T
                df_row = df_row.loc[:, ~df_row.columns.str.contains("diagnostic")]
                rows.append(df_row)
            except Exception as exc:
                ctx.logger.error(
                    "Error exporting whole habitat radiomics for subject %s: %s", subj, exc
                )
                if rows:
                    rows.append(FeatureUtils.create_empty_dataframe_like(rows[0], index=[0]))
        pb.close()

        if not rows:
            ctx.logger.error("No valid whole habitat radiomics data to export")
            return None

        result = pd.concat(rows)
        result.index = subjs
        out_file = os.path.join(ctx.out_dir, self.output_csv_name)
        result.to_csv(out_file, index=True)
        ctx.logger.info("Whole habitat radiomics features saved to %s", out_file)
        return result


# ---------------------------------------------------------------------------
# each_habitat: PyRadiomics on the raw image per individual habitat label
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("each_habitat")
class EachHabitatFeature(BaseHabitatFeature):
    """PyRadiomics extracted from the raw image within each habitat label.

    Produces one CSV per habitat label (habitat_1_radiomics.csv, …) plus a
    habitat_count.csv indicating which subjects contain each habitat.
    Wraps HabitatRadiomicsExtractor.extract_radiomics_features_from_each_habitat().
    """

    subject_data_key = "radiomics_features_from_each_habitat"
    output_csv_name = "habitat_1_radiomics.csv"   # representative; actual names are per-label
    progress_desc = "Each Habitat Radiomics"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for per-habitat radiomics."""
        return "each_habitat"

    def __init__(self, params_file: Optional[str] = None) -> None:
        """
        Args:
            params_file: Path to the PyRadiomics parameter YAML for the raw image.
        """
        super().__init__()
        self.params_file = params_file

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Extract radiomics per habitat label for each image modality.

        Args:
            ctx: Per-subject context; uses habitat_path and image_paths.

        Returns:
            Dict mapping image name to {habitat_id: radiomics_dict}.
        """
        results: Dict[str, Any] = {}
        for img_name, img_path in ctx.image_paths.items():
            try:
                results[img_name] = (
                    HabitatRadiomicsExtractor.extract_radiomics_features_from_each_habitat(
                        ctx.habitat_path, img_path, ctx.subj, self.params_file
                    )
                )
            except Exception as exc:
                ctx.logger.error(
                    "Error extracting each-habitat radiomics for subject %s, image %s: %s",
                    ctx.subj, img_name, exc,
                )
                results[img_name] = {"error": str(exc)}
        return results

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """Save per-habitat-label CSVs and a habitat-count CSV.

        Args:
            data: Per-subject feature dict.
            ctx: Batch context with out_dir, n_habitats, logger.

        Returns:
            Dict mapping habitat_id to its DataFrame, or None.
        """
        subjs = list(data.keys())
        n_habitats = ctx.n_habitats

        per_habitat: Dict[int, list] = {i + 1: [] for i in range(n_habitats)}
        habitat_count = pd.DataFrame(
            np.zeros((len(subjs), n_habitats)),
            index=subjs,
            columns=[np.arange(1, n_habitats + 1)],
        )

        pb = CustomTqdm(total=n_habitats, desc=self.progress_desc)
        for hab_id in per_habitat:
            pb.update(1)
            for i, subj in enumerate(subjs):
                try:
                    hab_data = data[subj].get(self.subject_data_key, {})
                    if i == 0:
                        imgs = list(hab_data.keys())
                    rows_per_modal: list = []
                    for j_img, img in enumerate(imgs):
                        if hab_id == 1 and j_img == 0:
                            col_names = list(hab_data.get(img, {}).keys())
                            habitat_count.loc[subj, col_names] = 1
                        feature = hab_data.get(img, {}).get(hab_id)
                        if feature is not None:
                            df_feat = pd.DataFrame.from_dict(feature, orient="index").T
                            rows_per_modal.append(df_feat)
                    if rows_per_modal:
                        df_concat = pd.concat(rows_per_modal)
                        df_concat.index = imgs
                        df_concat = df_concat.loc[
                            :, ~df_concat.columns.str.contains("diagnostic")
                        ]
                        wide_cols = [
                            f"{col}_of_{idx}"
                            for idx in df_concat.index
                            for col in df_concat.columns
                        ]
                        wide_row = pd.DataFrame(
                            [df_concat.values.flatten()], columns=wide_cols, index=[subj]
                        )
                        per_habitat[hab_id].append(wide_row)
                except Exception as exc:
                    ctx.logger.error(
                        "Error exporting habitat %s radiomics for subject %s: %s",
                        hab_id, subj, exc,
                    )
                    if per_habitat[hab_id]:
                        per_habitat[hab_id].append(
                            FeatureUtils.create_empty_dataframe_like(
                                per_habitat[hab_id][0], index=[subj]
                            )
                        )

            if per_habitat[hab_id]:
                per_habitat[hab_id] = pd.concat(per_habitat[hab_id])
                out_file = os.path.join(ctx.out_dir, f"habitat_{hab_id}_radiomics.csv")
                per_habitat[hab_id].to_csv(out_file, index=True)
                ctx.logger.info("Habitat %s radiomics saved to %s", hab_id, out_file)
            else:
                ctx.logger.error("No valid radiomics data for habitat %s", hab_id)
        pb.close()

        # Write habitat_count CSV
        habitat_count.columns = [f"has_habitat_{i}" for i in range(1, n_habitats + 1)]
        count_file = os.path.join(ctx.out_dir, "habitat_count.csv")
        habitat_count.to_csv(count_file, index=True)
        ctx.logger.info("Habitat count information saved to %s", count_file)

        return per_habitat


# ---------------------------------------------------------------------------
# msi: Multiregional spatial interaction (MSI)
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("msi")
class MSIFeature(BaseHabitatFeature):
    """Multiregional spatial interaction (MSI) features derived from the habitat map.

    Wraps MSIFeatureExtractor.
    """

    subject_data_key = "msi_features"
    output_csv_name = "msi_features.csv"
    progress_desc = "MSI Features"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for MSI features."""
        return "msi"

    def __init__(self, voxel_cutoff: int = 10) -> None:
        """
        Args:
            voxel_cutoff: Minimum voxel count used to filter small regions.
        """
        super().__init__()
        self._extractor = MSIFeatureExtractor(voxel_cutoff=voxel_cutoff)

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Compute the MSI feature vector for one subject.

        Args:
            ctx: Per-subject context; uses habitat_path, n_habitats, and subj.

        Returns:
            Dict of MSI feature name → value.
        """
        return self._extractor.extract_MSI_features(
            ctx.habitat_path, ctx.n_habitats, ctx.subj
        )

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[pd.DataFrame]:
        """Save MSI features to CSV.

        Args:
            data: Per-subject feature dict.
            ctx: Batch context with out_dir and logger.

        Returns:
            DataFrame or None.
        """
        subjs = list(data.keys())
        rows: list = []

        pb = CustomTqdm(total=len(subjs), desc=self.progress_desc)
        for subj in subjs:
            pb.update(1)
            try:
                features = data[subj].get(self.subject_data_key, {})
                if "error" in features:
                    ctx.logger.error(
                        "MSI extraction error for subject %s: %s", subj, features["error"]
                    )
                    if rows:
                        rows.append(
                            FeatureUtils.create_empty_dataframe_like(rows[0], index=[subj])
                        )
                    continue
                df_row = pd.DataFrame.from_dict(features, orient="index").T
                df_row.index = [subj]
                rows.append(df_row)
            except Exception as exc:
                ctx.logger.error("Error exporting MSI features for subject %s: %s", subj, exc)
                if rows:
                    rows.append(
                        FeatureUtils.create_empty_dataframe_like(rows[0], index=[subj])
                    )
        pb.close()

        if not rows:
            ctx.logger.error("No valid MSI features data to export")
            return None

        result = pd.concat(rows)
        out_file = os.path.join(ctx.out_dir, self.output_csv_name)
        result.to_csv(out_file, index=True)
        ctx.logger.info("MSI features saved to %s", out_file)
        return result


# ---------------------------------------------------------------------------
# ith_score: Intratumoral Heterogeneity score
# ---------------------------------------------------------------------------

@HabitatFeatureFactory.register("ith_score")
class ITHFeature(BaseHabitatFeature):
    """Intratumoral Heterogeneity (ITH) score from the habitat map.

    Wraps ITHFeatureExtractor.
    """

    subject_data_key = "ith_features"
    output_csv_name = "ith_scores.csv"
    progress_desc = "ITH Features"

    @classmethod
    def feature_name(cls) -> str:
        """Return the canonical factory name for ITH features."""
        return "ith_score"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._extractor = ITHFeatureExtractor()

    def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
        """Compute the ITH score for one subject.

        Args:
            ctx: Per-subject context; uses habitat_path only.

        Returns:
            Dict with 'ith_score' and related per-habitat statistics.
        """
        return self._extractor.extract_ith_features(ctx.habitat_path)

    def export_batch(
        self,
        data: Dict[str, Dict[str, Any]],
        ctx: BatchExportContext,
    ) -> Optional[pd.DataFrame]:
        """Save ITH scores to CSV.

        Args:
            data: Per-subject feature dict.
            ctx: Batch context with out_dir and logger.

        Returns:
            DataFrame or None.
        """
        subjs = list(data.keys())
        collected: Dict[str, Any] = {}

        pb = CustomTqdm(total=len(subjs), desc=self.progress_desc)
        for subj in subjs:
            pb.update(1)
            try:
                value = data[subj].get(self.subject_data_key)
                if value is not None:
                    collected[subj] = value
            except Exception as exc:
                ctx.logger.error(
                    "Error exporting ITH features for subject %s: %s", subj, exc
                )
        pb.close()

        if not collected:
            ctx.logger.error("No valid ITH features data to export")
            return None

        result = pd.DataFrame.from_dict(collected, orient="index")
        out_file = os.path.join(ctx.out_dir, self.output_csv_name)
        result.to_csv(out_file)
        ctx.logger.info("ITH features saved to %s", out_file)
        return result


# Backward-compatible aliases for callers that imported the old plugin names.
# New code should use the ``*Feature`` class names above.
NonRadiomicsPlugin = NonRadiomicsFeature
TraditionalRadiomicsPlugin = TraditionalRadiomicsFeature
WholeHabitatPlugin = WholeHabitatFeature
EachHabitatPlugin = EachHabitatFeature
MSIPlugin = MSIFeature
ITHPlugin = ITHFeature
