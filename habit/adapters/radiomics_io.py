"""L1 filesystem adapters for the standalone radiomics workflow."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from habit.exceptions import DataFormatError

__all__ = [
    "RadiomicsFilePair",
    "RadiomicsFeatureRow",
    "discover_radiomics_file_pairs",
    "write_radiomics_feature_tables",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RadiomicsFilePair:
    """One image/mask pair discoverable by the standalone workflow."""

    subject_id: str
    modality: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class RadiomicsFeatureRow:
    """Numeric radiomics output for one subject, modality, and mask label."""

    subject_id: str
    modality: str
    label: int
    values: Mapping[str, float]


def _first_file(directory: Path) -> Optional[Path]:
    """Return the deterministic first visible file in a modality directory."""
    files = sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))
    if not files:
        return None
    if len(files) > 1:
        _LOG.warning("Multiple files in %s; using %s", directory, files[0].name)
    return files[0]


def _directory_pairs(root: Path) -> Tuple[RadiomicsFilePair, ...]:
    """Discover pairs in the documented ``images/`` plus ``masks/`` layout."""
    images_root, masks_root = root / "images", root / "masks"
    if not images_root.is_dir() or not masks_root.is_dir():
        raise DataFormatError(
            f"Radiomics input must contain images/ and masks/: {root}."
        )
    pairs: List[RadiomicsFilePair] = []
    for subject_dir in sorted(path for path in images_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        mask_subject = masks_root / subject_dir.name
        if not mask_subject.is_dir():
            continue
        for image_modality in sorted(path for path in subject_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            image_path = _first_file(image_modality)
            mask_path = _first_file(mask_subject / image_modality.name) if (mask_subject / image_modality.name).is_dir() else None
            if image_path is not None and mask_path is not None:
                pairs.append(RadiomicsFilePair(subject_dir.name, image_modality.name, image_path, mask_path))
    return tuple(pairs)


def _manifest_pairs(path: Path) -> Tuple[RadiomicsFilePair, ...]:
    """Discover pairs from the legacy explicit ``images``/``masks`` YAML manifest."""
    from habit.utils.config_loader import load_config

    raw = load_config(str(path), resolve_paths=False) or {}
    if not isinstance(raw, Mapping):
        raise DataFormatError(f"Radiomics manifest must be a mapping: {path}.")
    auto = str(raw.get("auto_select_first_file", "true")).casefold() in {"true", "1", "yes", "y"}

    def resolve(value: Any) -> Path:
        candidate = Path(str(value))
        candidate = candidate if candidate.is_absolute() else path.parent / candidate
        if auto and candidate.is_dir():
            selected = _first_file(candidate)
            if selected is None:
                raise DataFormatError(f"No file in manifest directory: {candidate}.")
            return selected
        return candidate

    images = raw.get("images", {})
    masks = raw.get("masks", {})
    if not isinstance(images, Mapping) or not isinstance(masks, Mapping):
        raise DataFormatError("Radiomics manifest images and masks must be mappings.")
    pairs: List[RadiomicsFilePair] = []
    for subject_id, image_modalities in images.items():
        mask_modalities = masks.get(subject_id, {})
        if not isinstance(image_modalities, Mapping) or not isinstance(mask_modalities, Mapping):
            continue
        for modality, image_value in image_modalities.items():
            if modality in mask_modalities:
                pairs.append(RadiomicsFilePair(str(subject_id), str(modality), resolve(image_value), resolve(mask_modalities[modality])))
    return tuple(pairs)


def discover_radiomics_file_pairs(
    source: str | Path,
    *,
    modalities: Optional[Sequence[str]] = None,
) -> Tuple[RadiomicsFilePair, ...]:
    """Discover and validate image/mask pairs, optionally selecting modalities."""
    path = Path(source)
    pairs = _manifest_pairs(path) if path.is_file() and path.suffix.casefold() in {".yaml", ".yml"} else _directory_pairs(path)
    selected = tuple(str(value) for value in modalities) if modalities is not None else None
    if selected is not None:
        present = {pair.modality for pair in pairs}
        missing = sorted(set(selected) - present)
        if missing:
            raise DataFormatError(f"Configured process_image_types are absent from radiomics input: {missing}.")
        pairs = tuple(pair for pair in pairs if pair.modality in selected)
    if not pairs:
        raise DataFormatError(f"No matching image/mask radiomics pairs found under {path}.")
    return tuple(sorted(pairs, key=lambda pair: (pair.subject_id, pair.modality)))


def _destination(root: Path, stem: str, export_format: str, timestamp: Optional[str]) -> Path:
    """Build one stable output path, appending UTC timestamp only when requested."""
    suffix = f"_{timestamp}" if timestamp else ""
    return root / f"{stem}{suffix}.{export_format}"


def _write_frame(frame: pd.DataFrame, path: Path, export_format: str) -> None:
    """Write a frame in one explicitly supported public format."""
    if export_format == "csv":
        frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    elif export_format == "json":
        frame.to_json(path, orient="records")
    elif export_format == "pickle":
        frame.to_pickle(path)
    else:  # schema validation guards this; retain a defensive adapter error.
        raise DataFormatError(f"Unsupported radiomics export format: {export_format}.")


def write_radiomics_feature_tables(
    root: str | Path,
    rows: Iterable[RadiomicsFeatureRow],
    *,
    export_by_image_type: bool,
    export_combined: bool,
    export_format: str,
    add_timestamp: bool,
    timestamp: Optional[str] = None,
    target_labels: Optional[Sequence[int]] = None,
    partial: bool = False,
) -> Dict[str, Path]:
    """Write v2 standalone radiomics exports and return their named artifacts."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    timestamp = (
        timestamp
        if add_timestamp
        else None
    )
    if add_timestamp and timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    labels = sorted({row.label for row in materialized})
    multi_label = len(target_labels if target_labels is not None else labels) > 1
    written: Dict[str, Path] = {}
    for label in labels:
        label_rows = [row for row in materialized if row.label == label]
        if export_by_image_type:
            for modality in sorted({row.modality for row in label_rows}):
                frame = pd.DataFrame(
                    [{"ID": row.subject_id, **dict(row.values)} for row in label_rows if row.modality == modality]
                )
                frame = frame.sort_values("ID", kind="stable").reset_index(drop=True)
                suffix = f"_label_{label}" if multi_label else ""
                partial_suffix = "_partial" if partial else ""
                path = _destination(destination, f"radiomics_features_{modality}{suffix}{partial_suffix}", export_format, timestamp)
                _write_frame(frame, path, export_format)
                written[f"{modality}:label:{label}"] = path
        if export_combined:
            by_subject: Dict[str, Dict[str, float]] = {}
            for row in label_rows:
                payload = by_subject.setdefault(row.subject_id, {})
                payload.update({f"{row.modality}_{name}": value for name, value in row.values.items()})
            frame = pd.DataFrame([{"ID": subject_id, **values} for subject_id, values in sorted(by_subject.items())])
            suffix = f"_label_{label}" if multi_label else ""
            partial_suffix = "_partial" if partial else ""
            path = _destination(destination, f"radiomics_features_all{suffix}{partial_suffix}", export_format, timestamp)
            _write_frame(frame, path, export_format)
            written[f"all:label:{label}"] = path
    return written
