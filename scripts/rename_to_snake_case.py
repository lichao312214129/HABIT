#!/usr/bin/env python3
"""
One-shot rename script: unify HABIT config keys and identifiers to snake_case.

Excludes PyRadiomics-native paths (torchradiomics fork, config/radiomics/*.yaml)
where upstream setting names (kernel_radius, binWidth, ...) must stay camelCase.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_PARTS = {
    "torchradiomics",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "build",
    "dist",
    ".eggs",
}
SKIP_FILE_PREFIXES = (ROOT / "config" / "radiomics",)

EXTENSIONS = {".py", ".yaml", ".yml", ".rst", ".md", ".csv"}

# Protect PascalCase class/type names before block-key renames.
CLASS_PLACEHOLDERS: tuple[tuple[str, str], ...] = (
    ("FeatureConstructionConfig", "FeatureConstructionConfig"),
    ("HabitatSegmentationConfig", "HabitatSegmentationConfig"),
    ("PreprocessingConfigurator", "PreprocessingConfigurator"),
    ("PreprocessingStepConfig", "PreprocessingStepConfig"),
    ("PreprocessingMethod", "PreprocessingMethod"),
    ("PreprocessingConfig", "PreprocessingConfig"),
)

HABIT_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("supervoxel_union_bbox_crop", "supervoxel_union_bbox_crop"),
    ("supervoxel_pad_distance", "supervoxel_pad_distance"),
    ("use_supervoxel_cext", "use_supervoxel_cext"),
    ("use_torch_radiomics", "use_torch_radiomics"),
    ("feature_construction", "feature_construction"),
    ("habitat_segmentation", "habitat_segmentation"),
    ("supervoxel_batch", "supervoxel_batch"),
    ("supervoxel_id", "supervoxel_id"),
    ("torch_gpu_count", "torch_gpu_count"),
    ("gpu_slot_index", "gpu_slot_index"),
    ("kernel_radius", "kernel_radius"),
    ("voxel_batch", "voxel_batch"),
    ("torch_device", "torch_device"),
    ("torch_gpus", "torch_gpus"),
    ("torch_dtype", "torch_dtype"),
    ("subject_id_column", "subject_id_column"),
    ("subject_id", "subject_id"),
)

PYRADIOMICS_SETTINGS_KEYS: tuple[tuple[str, str], ...] = (
    ("'kernel_radius'", "'kernel_radius'"),
    ('"kernel_radius"', '"kernel_radius"'),
    ("'geometry_tolerance'", "'geometryTolerance'"),
    ('"geometry_tolerance"', '"geometryTolerance"'),
    ("'voxel_batch'", "'voxel_batch'"),
    ('"voxel_batch"', '"voxel_batch"'),
)

PYRADIOMICS_BOUNDARY_FILES = {
    ROOT / "habit/core/habitat_analysis/clustering_features/voxel_radiomics_extractor.py",
    ROOT / "habit/core/habitat_analysis/clustering_features/supervoxel_radiomics_extractor.py",
}


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIR_PARTS:
            return True
    for prefix in SKIP_FILE_PREFIXES:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False


def protect_classes(text: str) -> str:
    for old, placeholder in CLASS_PLACEHOLDERS:
        text = text.replace(old, placeholder)
    return text


def restore_classes(text: str) -> str:
    for old, placeholder in CLASS_PLACEHOLDERS:
        text = text.replace(placeholder, old)
    return text


def rename_preprocessing_block_key(text: str) -> str:
    """Rename YAML/schema block key Preprocessing -> preprocessing (not class names)."""
    text = re.sub(
        r"(^|\n)(\s*)Preprocessing:",
        r"\1\2preprocessing:",
        text,
    )
    text = text.replace('data.get("preprocessing")', 'data.get("preprocessing")')
    text = text.replace("data.get('preprocessing')", "data.get('preprocessing')")
    text = text.replace('out["preprocessing"]', 'out["preprocessing"]')
    text = text.replace("out['preprocessing']", "out['preprocessing']")
    text = re.sub(
        r"^\s+Preprocessing:\s*Dict",
        "    preprocessing: Dict",
        text,
        flags=re.MULTILINE,
    )
    text = text.replace(".preprocessing", ".preprocessing")
    return text


def rename_result_column_literals(text: str) -> str:
    """Rename hard-coded pipeline column headers to snake_case."""
    replacements = (
        ('"subject"', '"subject"'),
        ("'subject'", "'subject'"),
        ('"supervoxel"', '"supervoxel"'),
        ("'supervoxel'", "'supervoxel'"),
        ('"habitats"', '"habitats"'),
        ("'habitats'", "'habitats'"),
        ('"count"', '"count"'),
        ("'count'", "'count'"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def fix_result_columns_class(text: str, path: Path) -> str:
    if path.name == "habitat.py" and "schemas/workflows" in str(path).replace("\\", "/"):
        text = text.replace('SUBJECT = "subject"', 'SUBJECT = "subject"')
        text = text.replace('SUPERVOXEL = "supervoxel"', 'SUPERVOXEL = "supervoxel"')
        text = text.replace('COUNT = "count"', 'COUNT = "count"')
        text = text.replace('HABITATS = "habitats"', 'HABITATS = "habitats"')
    return text


def restore_pyradiomics_settings_keys(text: str) -> str:
    for old, new in PYRADIOMICS_SETTINGS_KEYS:
        text = text.replace(old, new)
    return text


def transform(text: str, path: Path) -> str:
    text = protect_classes(text)
    for old, new in HABIT_KEY_REPLACEMENTS:
        text = text.replace(old, new)
    text = rename_preprocessing_block_key(text)
    text = rename_result_column_literals(text)
    text = restore_classes(text)
    text = fix_result_columns_class(text, path)
    if path in PYRADIOMICS_BOUNDARY_FILES:
        text = restore_pyradiomics_settings_keys(text)
    return text


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = transform(original, path)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in EXTENSIONS or should_skip(path):
            continue
        if process_file(path):
            changed.append(path)
    print(f"Updated {len(changed)} files.")
    for p in sorted(changed):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
