#!/usr/bin/env python3
"""
One-shot helper: migrate habitat radiomics YAML blocks to explicit-binding style.

Updates voxel_radiomics / supervoxel_radiomics demo configs to:
* omit params_file (bundled preset fallback)
* list override keys in method parentheses
* remove orphan params under mean_voxel_features()

Run from repo root:
  python scripts/migrate_feature_expression_configs.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VOXEL_BLOCK_OLD = """    method: concat(voxel_radiomics(delay2, params_file, kernel_radius))
    params:
      params_file: ../radiomics/params_voxel_radiomics.yaml
      kernel_radius: 3
      voxel_batch: 1000
      use_torch_radiomics: auto"""

VOXEL_BLOCK_NEW = """    method: concat(voxel_radiomics(delay2, kernel_radius))
    params:
      kernel_radius: 3
      # params_file omitted -> bundled CT R3B12 preset (habit/resources/radiomics/params_voxel_radiomics.yaml)
      # voxel_batch / use_torch_radiomics omitted -> method defaults (1000, auto)"""

SUPERVOXEL_ORPHAN_OLD = """    method: mean_voxel_features()
    params:
      params_file: ../radiomics/params_supervoxel_radiomics.yaml
      supervoxel_union_bbox_crop: true
      use_supervoxel_cext: auto"""

SUPERVOXEL_ORPHAN_NEW = """    method: mean_voxel_features()
    params: {}"""

SUPERVOXEL_RAD_OLD = """    method: concat(supervoxel_radiomics(delay2, params_file))
    params:
      params_file: ../radiomics/params_supervoxel_radiomics.yaml
      supervoxel_union_bbox_crop: true
      use_supervoxel_cext: auto
      use_torch_radiomics: auto"""

SUPERVOXEL_RAD_NEW = """    method: concat(supervoxel_radiomics(delay2, supervoxel_union_bbox_crop, use_supervoxel_cext, use_torch_radiomics))
    params:
      supervoxel_union_bbox_crop: true
      use_supervoxel_cext: auto
      use_torch_radiomics: auto
      # params_file omitted -> bundled full-set preset (habit/resources/radiomics/params_supervoxel_radiomics.yaml)"""

MULTI_VOXEL_OLD = """    method: concat(voxel_radiomics(delay2), voxel_radiomics(delay3))
    params:
      params_file: ../radiomics/params_voxel_radiomics.yaml
      kernel_radius: 3
      voxel_batch: 1000
      use_torch_radiomics: auto
      torch_gpus: []"""

MULTI_VOXEL_NEW = """    method: concat(voxel_radiomics(delay2, kernel_radius), voxel_radiomics(delay3, kernel_radius))
    params:
      kernel_radius: 3
      # params_file omitted -> bundled CT R3B12 preset"""

TRIPLE_VOXEL_OLD = """    method: concat(voxel_radiomics(delay2), voxel_radiomics(delay3), voxel_radiomics(delay5))
    params:
      params_file: ../radiomics/params_voxel_radiomics.yaml
      kernel_radius: 3
      voxel_batch: 1000
      use_torch_radiomics: auto
      torch_gpus: []"""

TRIPLE_VOXEL_NEW = """    method: concat(voxel_radiomics(delay2, kernel_radius), voxel_radiomics(delay3, kernel_radius), voxel_radiomics(delay5, kernel_radius))
    params:
      kernel_radius: 3
      # params_file omitted -> bundled CT R3B12 preset"""


def _apply_replacements(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    habitat_dir = ROOT / "config" / "habitat"
    changed: list[str] = []

    voxel_files = list(habitat_dir.glob("*voxel_radiomics*.yaml"))
    for path in voxel_files:
        if _apply_replacements(
            path,
            [(VOXEL_BLOCK_OLD, VOXEL_BLOCK_NEW), (SUPERVOXEL_ORPHAN_OLD, SUPERVOXEL_ORPHAN_NEW)],
        ):
            changed.append(str(path.relative_to(ROOT)))

    for name in (
        "config_habitat_two_step_supervoxel_radiomics_train.yaml",
        "config_habitat_two_step_supervoxel_radiomics_predict.yaml",
    ):
        path = habitat_dir / name
        if path.is_file() and _apply_replacements(path, [(SUPERVOXEL_RAD_OLD, SUPERVOXEL_RAD_NEW)]):
            changed.append(str(path.relative_to(ROOT)))

    for name, old, new in (
        ("config_habitat_two_step_supervoxel_slic.yaml", MULTI_VOXEL_OLD, MULTI_VOXEL_NEW),
        ("config_habitat_direct_pooling_gmm_aic.yaml", TRIPLE_VOXEL_OLD, TRIPLE_VOXEL_NEW),
    ):
        path = habitat_dir / name
        if path.is_file() and _apply_replacements(path, [(old, new)]):
            changed.append(str(path.relative_to(ROOT)))

    print(f"Updated {len(changed)} file(s):")
    for item in changed:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
