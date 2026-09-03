"""Filesystem adapter for the batch image-preprocessing recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import SimpleITK as sitk

from habit.adapters.image_refs import FileImageRef
from habit.contracts.subject import Subject
from habit.exceptions import DataFormatError
from habit.utils.config_loader import load_config

__all__ = ["PreprocessingInput", "PreprocessingIOAdapter"]


@dataclass(frozen=True)
class PreprocessingInput:
    """One lazily materialised subject plus its modality-specific output paths."""

    subject: Subject
    output_modalities: Tuple[str, ...]


def _first_file(path: Path, *, auto_select_first_file: bool) -> Optional[Path]:
    """Resolve a manifest path, preserving the legacy first-file convention."""
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    files = [entry for entry in path.iterdir() if entry.is_file() and not entry.name.startswith(".")]
    if not files:
        return None
    if auto_select_first_file:
        return files[0]
    return path


class PreprocessingIOAdapter:
    """Discover legacy directory/manifest inputs and write NIfTI batch results."""

    def __init__(
        self,
        *,
        source_root: str,
        destination_root: str,
        auto_select_first_file: bool,
    ) -> None:
        self.source_root = Path(source_root)
        self.destination_root = Path(destination_root)
        self.auto_select_first_file = bool(auto_select_first_file)

    def discover(self) -> List[PreprocessingInput]:
        """Build lazy Subjects from either the HABIT directory layout or a manifest."""
        images, masks = self._read_paths()
        inputs: List[PreprocessingInput] = []
        for subject_id, image_paths in images.items():
            subject_images = {
                modality: FileImageRef(path, is_mask=False, role_name=modality)
                for modality, path in image_paths.items()
            }
            subject_masks = {
                modality: FileImageRef(path, is_mask=True, role_name=modality)
                for modality, path in masks.get(subject_id, {}).items()
            }
            if not subject_images:
                continue
            inputs.append(
                PreprocessingInput(
                    subject=Subject(
                        subject_id=str(subject_id),
                        images=subject_images,
                        masks=subject_masks,
                    ),
                    output_modalities=tuple(subject_images),
                )
            )
        return inputs

    def write(
        self,
        subject: Subject,
        *,
        stage_name: Optional[str] = None,
        modalities: Optional[Sequence[str]] = None,
    ) -> None:
        """Write processed images and same-name masks to the legacy NIfTI tree."""
        root = (
            self.destination_root / "processed_images"
            if stage_name is None
            else self.destination_root / stage_name
        )
        if stage_name is None:
            # The retired BatchProcessor always created this directory, even
            # when a pipeline produced no metadata sidecars. Preserve that
            # observable output-tree contract for callers that archive it.
            (root / "metadata").mkdir(parents=True, exist_ok=True)
        selected = list(modalities) if modalities is not None else list(subject.images)
        for modality in selected:
            image_dir = root / "images" / subject.subject_id / modality
            image_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                subject.image(modality).to_sitk(),
                str(image_dir / f"{modality}.nii.gz"),
            )
            if modality in subject.masks:
                mask_dir = root / "masks" / subject.subject_id / modality
                mask_dir.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(
                    subject.mask(modality).to_sitk(),
                    str(mask_dir / f"mask_{modality}.nii.gz"),
                )

    def _read_paths(self) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Dict[str, Path]]]:
        """Read resolved image/mask mappings from a manifest or directory tree."""
        if self.source_root.is_file() and self.source_root.suffix.lower() in {".yaml", ".yml"}:
            raw = load_config(self.source_root)
            return self._resolve_manifest_paths(raw.get("images", {})), self._resolve_manifest_paths(
                raw.get("masks", {})
            )
        return self._scan_directory("images"), self._scan_directory("masks", optional=True)

    def _resolve_manifest_paths(
        self,
        mapping: Mapping[str, Mapping[str, str]],
    ) -> Dict[str, Dict[str, Path]]:
        """Resolve selected manifest files while retaining subject/modality keys."""
        output: Dict[str, Dict[str, Path]] = {}
        for subject_id, modalities in mapping.items():
            resolved: Dict[str, Path] = {}
            for modality, raw_path in modalities.items():
                path = _first_file(
                    Path(raw_path),
                    auto_select_first_file=self.auto_select_first_file,
                )
                if path is not None:
                    resolved[str(modality)] = path
            output[str(subject_id)] = resolved
        return output

    def _scan_directory(
        self,
        folder: str,
        *,
        optional: bool = False,
    ) -> Dict[str, Dict[str, Path]]:
        """Scan ``<root>/<folder>/<subject>/<modality>/<file>`` without loading voxels."""
        root = self.source_root / folder
        if not root.is_dir():
            if optional:
                return {}
            raise DataFormatError(f"Input image directory does not exist: {root}")
        output: Dict[str, Dict[str, Path]] = {}
        for subject_dir in (entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")):
            modalities: Dict[str, Path] = {}
            for modality_dir in (
                entry for entry in subject_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
            ):
                path = _first_file(
                    modality_dir,
                    auto_select_first_file=self.auto_select_first_file,
                )
                if path is not None:
                    modalities[modality_dir.name] = path
            output[subject_dir.name] = modalities
        return output
