# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.
"""
Check a HABIT YAML config: syntax first, then optional schema validation.

Intended for clinicians who edit YAML by hand and want a dry-run before a long job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Type

import click
import yaml

from habit.common import (
    echo_error,
    echo_success,
    exit_with_error,
    format_config_load_error,
    require_config_path,
)

# workflow alias -> lazy loader for the typed config class
_WORKFLOW_LOADERS: Dict[str, Callable[[], Type]] = {
    "preprocess": lambda: __import__(
        "habit.core.schemas.workflows.preprocessing", fromlist=["PreprocessingConfig"]
    ).PreprocessingConfig,
    "habitat": lambda: __import__(
        "habit.core.schemas.workflows.habitat", fromlist=["HabitatAnalysisConfig"]
    ).HabitatAnalysisConfig,
    "extract": lambda: __import__(
        "habit.core.schemas.workflows.habitat", fromlist=["FeatureExtractionConfig"]
    ).FeatureExtractionConfig,
    "radiomics": lambda: __import__(
        "habit.core.schemas.workflows.habitat", fromlist=["RadiomicsConfig"]
    ).RadiomicsConfig,
    "model": lambda: __import__(
        "habit.core.schemas.workflows.ml", fromlist=["MLConfig"]
    ).MLConfig,
    "cv": lambda: __import__(
        "habit.core.schemas.workflows.ml", fromlist=["MLConfig"]
    ).MLConfig,
    "compare": lambda: __import__(
        "habit.core.schemas.workflows.ml", fromlist=["ModelComparisonConfig"]
    ).ModelComparisonConfig,
    "icc": lambda: __import__(
        "habit.core.machine_learning.feature_selectors.icc.config",
        fromlist=["ICCConfig"],
    ).ICCConfig,
    "retest": lambda: __import__(
        "habit.core.schemas.workflows.ml", fromlist=["TestRetestConfig"]
    ).TestRetestConfig,
    "sort-dicom": lambda: __import__(
        "habit.core.schemas.workflows.dicom_sort", fromlist=["DicomSortConfig"]
    ).DicomSortConfig,
}


def _guess_workflow(config_path: Path) -> Optional[str]:
    """
    Guess workflow type from path fragments (best-effort for doctor UX).

    Directory names are preferred over filename tokens so that e.g.
    ``machine_learning/config_machine_learning_radiomics.yaml`` maps to
    ``model``, not ``radiomics``.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Workflow alias or None when unknown.
    """
    parts = [p.lower() for p in config_path.parts]
    name = config_path.name.lower()

    # Prefer parent directory conventions used under config/.
    dir_rules = (
        ("preprocessing", "preprocess"),
        ("dicom_sort", "sort-dicom"),
        ("feature_extraction", "extract"),
        ("machine_learning", "model"),
        ("model_comparison", "compare"),
        ("habitat", "habitat"),
        ("radiomics", "radiomics"),
        ("auxiliary", "icc"),
    )
    for needle, alias in dir_rules:
        if needle in parts:
            # K-fold configs live under machine_learning/ but use habit cv.
            if alias == "model" and "kfold" in name:
                return "cv"
            # test-retest templates under auxiliary/
            if alias == "icc" and ("retest" in name or "test_retest" in name):
                return "retest"
            return alias

    # Filename fallbacks when the file is copied outside config/.
    name_rules = (
        ("preprocess", "preprocess"),
        ("kfold", "cv"),
        ("habitat", "habitat"),
        ("extract_features", "extract"),
        ("radiomics", "radiomics"),
        ("machine_learning", "model"),
        ("model_comparison", "compare"),
        ("icc", "icc"),
        ("retest", "retest"),
        ("sort_dicom", "sort-dicom"),
    )
    for needle, alias in name_rules:
        if needle in name:
            return alias
    return None


def run_check_config(
    config_path: str,
    workflow: Optional[str] = None,
    syntax_only: bool = False,
) -> None:
    """
    Validate YAML syntax and optionally the workflow schema.

    Args:
        config_path: Path to configuration YAML.
        workflow: Optional workflow alias (preprocess, habitat, model, ...).
            When omitted, guessed from the file path when possible.
        syntax_only: When True, validate YAML syntax only. Use this for
            input manifests and PyRadiomics parameter presets, which are
            referenced by a workflow config rather than run directly.
    """
    path = Path(require_config_path(config_path))
    if not path.is_file():
        exit_with_error(f"Error: 找不到配置文件 / Config not found: {path}")

    # 1) YAML syntax
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        exit_with_error(format_config_load_error(exc, str(path)))
    except OSError as exc:
        exit_with_error(f"Error: 无法读取配置文件 / Cannot read config: {exc}")

    if data is None:
        exit_with_error(
            "Error: 配置文件为空 / Config file is empty.\n"
            "请确认文件有内容，且不是只有注释。"
        )
    if not isinstance(data, dict):
        exit_with_error(
            "Error: 配置根节点必须是键值对（mapping）/"
            "Root of YAML must be a mapping (key: value).\n"
            f"当前类型 / got: {type(data).__name__}"
        )

    click.echo(f"✓ YAML 语法正确 / YAML syntax OK: {path}")

    # 2) Schema (optional)
    if syntax_only:
        echo_success("YAML 语法检查通过 / YAML syntax check passed")
        return

    alias = (workflow or _guess_workflow(path) or "").strip().lower()
    if not alias:
        click.echo(
            "提示: 未指定 --workflow，且无法从路径推断工作流；"
            "已跳过字段校验。可用例如: habit check-config -c FILE -w model"
        )
        echo_success("配置语法检查通过（未做字段校验）")
        return

    if alias not in _WORKFLOW_LOADERS:
        exit_with_error(
            f"Error: 未知工作流类型 / Unknown workflow: {alias}\n"
            f"可选 / choices: {', '.join(sorted(_WORKFLOW_LOADERS))}"
        )

    config_cls = _WORKFLOW_LOADERS[alias]()
    try:
        config_cls.from_file(str(path))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        echo_error(format_config_load_error(exc, str(path)))
        exit_with_error(
            f"字段校验失败 / Schema validation failed (workflow={alias}).\n"
            "请对照配置文件中「★ 必改」项与中文注释修改。"
        )

    echo_success(f"配置检查通过 / Config OK (workflow={alias})")
