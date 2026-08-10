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
"""L1 I/O helpers for habitat feature-extraction recipes.

Assembles in-memory :class:`~habit.contracts.subject.Subject` /
:class:`~habit.contracts.habitat.HabitatMap` inputs from the v0.1 directory
layout, and writes feature CSVs in the historical ``habit extract`` layout
so CLI consumers keep seeing the same files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.adapters.image_refs import FileImageRef
from habit.contracts.geometry import Geometry
from habit.contracts.habitat import HabitatMap
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.exceptions import DataFormatError, HABITAPIError
from habit.utils.habitats_results_io import (
    find_habitats_results_file,
    load_habitats_results,
)
from habit.utils.progress_utils import CustomTqdm

__all__ = [
    "discover_habitat_map_paths",
    "load_extract_cohort",
    "read_habitat_map",
    "resolve_n_habitats",
    "write_extract_feature_csvs",
]

_LOG = logging.getLogger(__name__)

#: v0.1 CLI CSV stems keyed by domain registry name.
_FAMILY_CSV_STEM: Mapping[str, str] = {
    "msi": "msi_features",
    "ith_score": "ith_scores",
    "non_radiomics": "habitat_basic_features",
    "traditional": "raw_image_radiomics",
    "whole_habitat": "whole_habitat_radiomics",
    "graph": "habitat_graph_features",
}

#: Domain ITH summary columns remapped to the v0.1 CSV names.
_ITH_CSV_RENAMES: Mapping[str, str] = {
    "ith_num_habitats": "num_habitats",
    "ith_total_area": "total_area",
}


def _require_simpleitk() -> Any:
    """Import SimpleITK lazily so the adapter layer stays light to import."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:  # pragma: no cover - present in CI
        raise HABITAPIError(
            "SimpleITK is required to read habitat map files from disk."
        ) from exc
    return sitk


def _first_file_in(directory: Path) -> Optional[Path]:
    """Return the first non-hidden file inside a convention subdirectory."""
    if not directory.is_dir():
        return None
    files = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    )
    if not files:
        return None
    if len(files) > 1:
        _LOG.warning("Multiple files in %s; using %s", directory, files[0].name)
    return files[0]


def load_extract_cohort(
    raw_img_folder: Union[str, Path],
    *,
    images_folder: str = "images",
    name: Optional[str] = None,
) -> Cohort:
    """
    Build a cohort from ``raw_img_folder/images/<subject>/<modality>/``.

    Unlike :class:`~habit.adapters.directory.DirectoryDataSource`, masks are
    not required: habitat feature extractors derive the ROI from the habitat
    map. Every modality folder that contains a file is kept (per-subject).

    Args:
        raw_img_folder: Root directory holding the ``images`` subdirectory.
        images_folder: Name of the images subdirectory under the root.
        name: Optional cohort name.

    Returns:
        Cohort with lazy :class:`~habit.adapters.image_refs.FileImageRef`
        images, sorted by subject id.

    Raises:
        DataFormatError: If the images tree is missing or yields no subjects.
    """
    root = Path(raw_img_folder)
    images_root = root / images_folder
    if not images_root.is_dir():
        raise DataFormatError(
            f"Images folder not found: {images_root}. Expected layout "
            f"{root}/images/<subject>/<modality>/<file>."
        )

    subjects: List[Subject] = []
    for subject_dir in sorted(
        (
            entry
            for entry in images_root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        ),
        key=lambda entry: entry.name,
    ):
        images: Dict[str, FileImageRef] = {}
        for modality_dir in sorted(
            (
                entry
                for entry in subject_dir.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ),
            key=lambda entry: entry.name,
        ):
            file_path = _first_file_in(modality_dir)
            if file_path is None:
                continue
            images[modality_dir.name] = FileImageRef(
                file_path, is_mask=False, role_name=modality_dir.name
            )
        if not images:
            _LOG.warning(
                "Subject %s has no image files under %s; skipped.",
                subject_dir.name,
                subject_dir,
            )
            continue
        subjects.append(
            Subject(subject_id=subject_dir.name, images=images, masks={})
        )

    if not subjects:
        raise DataFormatError(
            f"No subjects with images found under {images_root}."
        )
    return Cohort(subjects, name=name)


def discover_habitat_map_paths(
    habitats_map_folder: Union[str, Path],
    habitat_pattern: str = "*_habitats.nrrd",
) -> Dict[str, Path]:
    """
    Discover habitat map files and map them to subject ids.

    Subject ids are derived by stripping the non-wildcard portion of
    ``habitat_pattern`` from the filename, matching
    :class:`~habit.compat.engines.habitat_extraction.habitat_features.habitat_analyzer.HabitatMapAnalyzer`.

    Args:
        habitats_map_folder: Directory containing ``*_habitats.nrrd`` files.
        habitat_pattern: Glob pattern for habitat map filenames.

    Returns:
        Mapping of subject id to absolute habitat map path.
    """
    folder = Path(habitats_map_folder)
    if not folder.is_dir():
        raise DataFormatError(
            f"Habitat map folder not found: {folder}."
        )
    suffix = habitat_pattern.replace("*", "")
    paths: Dict[str, Path] = {}
    for path in sorted(folder.glob(habitat_pattern)):
        if not path.is_file():
            continue
        subject_id = path.name.replace(suffix, "")
        if not subject_id:
            _LOG.warning(
                "Could not derive subject id from habitat file %s; skipped.",
                path.name,
            )
            continue
        paths[subject_id] = path.resolve()
    return paths


def read_habitat_map(
    path: Union[str, Path],
    *,
    subject_id: str,
    habitat_ids: Sequence[int],
    model_id: str = "extract",
) -> HabitatMap:
    """
    Load one habitat label map from disk into a :class:`HabitatMap`.

    Args:
        path: NRRD / NIfTI / MetaImage path written by ``habit get-habitat``.
        subject_id: Owning subject id.
        habitat_ids: Canonical habitat ids the extractors should emit columns
            for (usually ``1..n_habitats``).
        model_id: Model identifier recorded on the map; extract from precomputed
            maps uses a placeholder because the original model file may be absent.

    Returns:
        In-memory habitat map with geometry taken from the file metadata.
    """
    sitk = _require_simpleitk()
    image = sitk.ReadImage(str(path))
    labels = np.asarray(sitk.GetArrayFromImage(image))
    if not np.issubdtype(labels.dtype, np.integer):
        labels = np.rint(labels).astype(np.int32)
    else:
        labels = labels.astype(np.int32, copy=False)
    size_xyz = tuple(int(v) for v in image.GetSize())
    geometry = Geometry(
        shape=tuple(reversed(size_xyz)),
        spacing=tuple(float(v) for v in image.GetSpacing()),
        origin=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )
    return HabitatMap(
        subject_id=subject_id,
        label_array=labels,
        geometry=geometry,
        model_id=model_id,
        habitat_ids=tuple(int(v) for v in habitat_ids),
        provenance=Provenance(
            produced_by="adapters.read_habitat_map",
            spec_fingerprint="habitat_map_file",
            notes={"path": str(Path(path).resolve())},
        ),
    )


def resolve_n_habitats(
    habitats_map_folder: Union[str, Path],
    n_habitats: Optional[int],
) -> int:
    """
    Resolve the habitat count for column alignment across the cohort.

    Prefers an explicit ``n_habitats``. Otherwise reads ``habitats.parquet`` /
    ``habitats.csv`` from the habitat output folder (same source as v0.1).

    Args:
        habitats_map_folder: Directory that may contain a habitats results table.
        n_habitats: Explicit count from config, or ``None`` to auto-detect.

    Returns:
        Positive habitat count.

    Raises:
        ValueError: If the count cannot be determined.
    """
    if n_habitats is not None:
        count = int(n_habitats)
        if count < 1:
            raise ValueError(f"n_habitats must be >= 1; got {count}.")
        return count

    results_path = find_habitats_results_file(habitats_map_folder)
    if results_path is None:
        raise ValueError(
            "Unable to determine the number of habitats automatically. "
            "Provide FeatureExtractionConfig.n_habitats explicitly, or write "
            "habitats.parquet / habitats.csv (via StudyResult.save) into "
            f"habitats_map_folder: {str(habitats_map_folder)!r}."
        )
    frame = load_habitats_results(results_path)
    for column in ("habitats", "habitat", "label"):
        if column in frame.columns:
            unique = int(pd.Series(frame[column]).nunique())
            if unique < 1:
                break
            return unique
    raise ValueError(
        f"Could not infer n_habitats from {results_path}; expected a "
        "'habitats' column."
    )


def _frame_with_subject_index(table: FeatureTable) -> pd.DataFrame:
    """Return a DataFrame indexed by subject id for v0.1 CSV layout."""
    frame = table.frame.copy()
    if "subject" in frame.columns:
        frame = frame.set_index("subject")
    frame.index.name = None
    return frame


def _write_simple_family(
    root: Path,
    family: str,
    tables: Sequence[FeatureTable],
    *,
    logger: logging.Logger,
) -> Optional[str]:
    """Concatenate one-row tables and write the family's CSV."""
    if not tables:
        logger.error("No tables to export for feature family %s", family)
        return None
    frames = [_frame_with_subject_index(table) for table in tables]
    if family == "ith_score":
        frames = [frame.rename(columns=dict(_ITH_CSV_RENAMES)) for frame in frames]
    result = pd.concat(frames, axis=0)
    stem = _FAMILY_CSV_STEM[family]
    destination = root / f"{stem}.csv"
    result.to_csv(destination, index=True)
    logger.info("%s features saved to %s", family, destination)
    return str(destination)


def _write_each_habitat_family(
    root: Path,
    tables: Sequence[FeatureTable],
    n_habitats: int,
    *,
    logger: logging.Logger,
) -> List[str]:
    """
    Split the wide each_habitat FeatureTable into per-habitat CSVs.

    Domain columns are ``habitat_{id}_{feature}_of_{modality}`` plus
    ``has_habitat_{id}``. The v0.1 layout writes one CSV per habitat with the
    ``habitat_{id}_`` prefix stripped, plus ``habitat_count.csv``.
    """
    if not tables:
        logger.error("No tables to export for feature family each_habitat")
        return []

    written: List[str] = []
    frames = [_frame_with_subject_index(table) for table in tables]
    combined = pd.concat(frames, axis=0)

    count_cols = [f"has_habitat_{hid}" for hid in range(1, n_habitats + 1)]
    count_frame = pd.DataFrame(index=combined.index)
    for col in count_cols:
        if col in combined.columns:
            count_frame[col] = combined[col].fillna(0).astype(float)
        else:
            count_frame[col] = 0.0
    count_path = root / "habitat_count.csv"
    count_frame.to_csv(count_path, index=True)
    logger.info("Habitat count information saved to %s", count_path)
    written.append(str(count_path))

    prefix_re = re.compile(r"^habitat_(\d+)_(.+)$")
    pb = CustomTqdm(total=n_habitats, desc="Each Habitat Radiomics")
    for hid in range(1, n_habitats + 1):
        pb.update(1)
        renamed: Dict[str, str] = {}
        for column in combined.columns:
            match = prefix_re.match(str(column))
            if match is None:
                continue
            if int(match.group(1)) != hid:
                continue
            renamed[column] = match.group(2)
        if not renamed:
            logger.error("No valid radiomics data for habitat %s", hid)
            continue
        habitat_frame = combined.loc[:, list(renamed.keys())].rename(columns=renamed)
        destination = root / f"habitat_{hid}_radiomics.csv"
        habitat_frame.to_csv(destination, index=True)
        logger.info("Habitat %s radiomics saved to %s", hid, destination)
        written.append(str(destination))
    pb.close()
    return written


def write_extract_feature_csvs(
    root: Union[str, Path],
    family_tables: Mapping[str, Sequence[FeatureTable]],
    *,
    n_habitats: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Persist domain feature tables in the v0.1 ``habit extract`` CSV layout.

    Args:
        root: Destination directory (created if absent). Named ``root`` to
            match :class:`~habit.adapters.writers.DirectoryResultWriter`.
        family_tables: Mapping of registry feature name to per-subject tables
            in cohort order.
        n_habitats: Habitat count used for ``each_habitat`` / count CSVs.
        logger: Optional logger; defaults to this module's logger.

    Returns:
        Paths of files written.
    """
    log = logger or _LOG
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for family, tables in family_tables.items():
        if family == "each_habitat":
            written.extend(
                _write_each_habitat_family(
                    destination, list(tables), n_habitats, logger=log
                )
            )
            continue
        if family in _FAMILY_CSV_STEM:
            path = _write_simple_family(
                destination, family, list(tables), logger=log
            )
            if path is not None:
                written.append(path)
            continue
        # Unknown families: write a joined wide CSV named after the family.
        if not tables:
            continue
        frames = [_frame_with_subject_index(table) for table in tables]
        result = pd.concat(frames, axis=0)
        # Distinct name from the ``Optional[str]`` above: this branch builds a
        # Path, and reusing ``path`` would conflate the two types.
        family_path = destination / f"{family}_features.csv"
        result.to_csv(family_path, index=True)
        log.info("Feature family %s saved to %s", family, family_path)
        written.append(str(family_path))
    return written
