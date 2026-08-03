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
"""
End-to-end CLI tests for every HABIT pipeline config under ``config/``.

Auxiliary YAML files (file lists, radiomics parameter templates, etc.) are
excluded — only configs that can be passed to ``habit <command> -c`` are
included.

Run inside conda ``py310`` (see ``.cursor/rules/runtime-environment.mdc``):

    pytest tests/test_all_configs.py -m integration
    pytest tests/test_all_configs.py -m "integration and not slow"  # schema only
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import pytest
import yaml
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

# Filename prefixes / names that are NOT CLI entry configs.
_AUXILIARY_NAME_PREFIXES: Tuple[str, ...] = (
    "files_",
    "file_habitat",
)
_AUXILIARY_EXACT_NAMES: frozenset[str] = frozenset({"image_files.yaml"})


@dataclass(frozen=True)
class PipelineConfigSpec:
    """One runnable pipeline YAML and its CLI invocation."""

    rel_path: str
    command: str
    extra_args: Tuple[str, ...] = ()
    skip_on_windows: bool = False
    skip_unless_paths_exist: Tuple[str, ...] = ()
    skip_reason: str = ""
    pytest_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.pytest_id:
            object.__setattr__(self, "pytest_id", Path(self.rel_path).stem)

    @property
    def abs_path(self) -> Path:
        return PROJECT_ROOT / self.rel_path

    def cli_args(self) -> list[str]:
        """Build argv fragment: ``[command, '-c', path, *extra_args]``."""
        return [self.command, "-c", self.rel_path, *self.extra_args]


def _is_auxiliary_config(path: Path) -> bool:
    """Return True when ``path`` is a helper YAML, not a pipeline entry config."""
    name = path.name
    if name in _AUXILIARY_EXACT_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in _AUXILIARY_NAME_PREFIXES):
        return True
    # Radiomics folder: only ``config_traditional_radiomics.yaml`` is a CLI config.
    if path.parent.name == "radiomics" and not name.startswith("config_traditional"):
        return True
    return False


def _ml_cli_mode(yaml_path: Path) -> Tuple[str, Tuple[str, ...]]:
    """
    Infer ``habit model`` vs ``habit cv`` and ``--mode`` for ML configs.

    Returns:
        (command, extra_args) e.g. ``("model", ("-m", "train"))`` or ``("cv", ())``.
    """
    name = yaml_path.name.lower()
    if "kfold" in name:
        return "cv", ()
    if "predict" in name:
        return "model", ("-m", "predict")
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if isinstance(raw, dict) and raw.get("run_mode") == "predict":
        return "model", ("-m", "predict")
    return "model", ("-m", "train")


def _infer_spec(path: Path) -> Optional[PipelineConfigSpec]:
    """Map a config file path to its CLI command and options."""
    if _is_auxiliary_config(path):
        return None

    rel = path.relative_to(PROJECT_ROOT).as_posix()
    name = path.name
    subdir = path.parent.name

    skip_wsl = "_wsl" in name
    skip_reason_wsl = "WSL-only config (uses ~/habit_data paths)" if skip_wsl else ""

    if subdir == "preprocessing":
        if not (name.startswith("config_preprocessing") or name.startswith("config_image")):
            return None
        return PipelineConfigSpec(
            rel_path=rel,
            command="preprocess",
            skip_on_windows=skip_wsl,
            skip_reason=skip_reason_wsl,
        )

    if subdir == "dicom_sort":
        return PipelineConfigSpec(rel_path=rel, command="sort-dicom")

    if subdir == "habitat":
        if not name.startswith("config_habitat"):
            return None
        return PipelineConfigSpec(
            rel_path=rel,
            command="get-habitat",
            skip_on_windows=skip_wsl,
            skip_reason=skip_reason_wsl,
        )

    if subdir == "feature_extraction":
        return PipelineConfigSpec(rel_path=rel, command="extract")

    if subdir == "machine_learning":
        command, extra = _ml_cli_mode(path)
        skip_paths: Tuple[str, ...] = ()
        skip_reason = skip_reason_wsl
        if command == "model" and extra == ("-m", "predict"):
            # Predict configs reference a trained pipeline under demo_data/results.
            skip_paths = ("demo_data/results/ml",)
            skip_reason = (
                skip_reason
                or "Predict config requires prior train output under demo_data/results/ml"
            )
        return PipelineConfigSpec(
            rel_path=rel,
            command=command,
            extra_args=extra,
            skip_on_windows=skip_wsl,
            skip_unless_paths_exist=skip_paths,
            skip_reason=skip_reason,
        )

    if subdir == "model_comparison":
        return PipelineConfigSpec(rel_path=rel, command="compare")

    if subdir == "auxiliary":
        if name.startswith("config_icc"):
            return PipelineConfigSpec(rel_path=rel, command="icc")
        if name.startswith("config_test_retest"):
            return PipelineConfigSpec(rel_path=rel, command="retest")
        return None

    if subdir == "radiomics":
        return PipelineConfigSpec(rel_path=rel, command="radiomics")

    return None


def discover_pipeline_configs() -> list[PipelineConfigSpec]:
    """Collect all CLI-runnable configs under ``config/``, sorted by path."""
    specs: list[PipelineConfigSpec] = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        spec = _infer_spec(path)
        if spec is not None:
            specs.append(spec)
    return specs


PIPELINE_CONFIGS: list[PipelineConfigSpec] = discover_pipeline_configs()


def _schema_loader(spec: PipelineConfigSpec) -> Callable[[str], Any]:
    """Return the ``from_file`` callable for schema validation of ``spec``."""
    loaders: dict[str, Tuple[str, str]] = {
        "preprocess": (
            "habit.core.preprocessing.config_schemas",
            "PreprocessingConfig",
        ),
        "sort-dicom": (
            "habit.core.dicom_sort",
            "DicomSortConfig",
        ),
        "get-habitat": (
            "habit.core.habitat_analysis.config_schemas",
            "HabitatAnalysisConfig",
        ),
        "model": (
            "habit.core.machine_learning.config_schemas",
            "MLConfig",
        ),
        "cv": (
            "habit.core.machine_learning.config_schemas",
            "MLConfig",
        ),
        "compare": (
            "habit.core.machine_learning.config_schemas",
            "ModelComparisonConfig",
        ),
        "icc": (
            "habit.core.machine_learning.feature_selectors.icc.config",
            "ICCConfig",
        ),
        "retest": (
            "habit.core.machine_learning.config_schemas",
            "TestRetestConfig",
        ),
        "radiomics": (
            "habit.core.habitat_analysis.config_schemas",
            "RadiomicsConfig",
        ),
    }
    if spec.command == "extract":
        from habit.core.habitat_analysis.feature_extraction_loader import (
            load_feature_extraction_config_from_file,
        )

        def _load_extract(path: str) -> Any:
            config, _plugins = load_feature_extraction_config_from_file(path)
            return config

        return _load_extract

    module_name, class_name = loaders[spec.command]
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls.from_file  # type: ignore[no-any-return]


def _path_exists(path: Path) -> bool:
    """Return True when ``path`` exists as a file or directory."""
    return path.is_file() or path.is_dir()


def _collect_required_input_paths(config: Any, spec: PipelineConfigSpec) -> list[Path]:
    """
    Collect filesystem paths that must exist before a CLI E2E run can succeed.

    Production templates often reference user-specific absolute paths; when those
    paths are missing on the current machine we skip E2E (schema tests still run).
    """
    paths: list[Path] = []
    command = spec.command

    if command == "preprocess":
        data_dir = getattr(config, "data_dir", None)
        if data_dir:
            paths.append(Path(str(data_dir)))

    elif command == "sort-dicom":
        data_dir = getattr(config, "data_dir", None) or getattr(config, "input_dir", None)
        if data_dir:
            paths.append(Path(str(data_dir)))

    elif command == "get-habitat":
        data_dir = getattr(config, "data_dir", None)
        if data_dir:
            paths.append(Path(str(data_dir)))
        run_mode = getattr(config, "run_mode", "train")
        if run_mode == "predict":
            pipeline_path = getattr(config, "pipeline_path", None)
            if pipeline_path:
                paths.append(Path(str(pipeline_path)))

    elif command == "extract":
        for attr in ("raw_img_folder", "habitats_map_folder"):
            value = getattr(config, attr, None)
            if value:
                paths.append(Path(str(value)))

    elif command in ("model", "cv"):
        for item in getattr(config, "input", []) or []:
            path_value = getattr(item, "path", None) if not isinstance(item, dict) else item.get("path")
            if path_value:
                paths.append(Path(str(path_value)))
        run_mode = getattr(config, "run_mode", None)
        if run_mode == "predict" or spec.extra_args == ("-m", "predict"):
            pipeline_path = getattr(config, "pipeline_path", None)
            if pipeline_path:
                paths.append(Path(str(pipeline_path)))

    elif command == "compare":
        for item in getattr(config, "files_config", []) or []:
            path_value = getattr(item, "path", None) if not isinstance(item, dict) else item.get("path")
            if path_value:
                paths.append(Path(str(path_value)))

    elif command == "icc":
        input_cfg = getattr(config, "input", None)
        if input_cfg is not None:
            input_type = getattr(input_cfg, "type", None)
            if input_type == "files":
                for group in getattr(input_cfg, "file_groups", []) or []:
                    for file_path in group:
                        paths.append(Path(str(file_path)))
            elif input_type == "directories":
                for dir_path in getattr(input_cfg, "dir_list", []) or []:
                    paths.append(Path(str(dir_path)))

    elif command == "retest":
        for attr in ("test_habitat_table", "retest_habitat_table", "input_dir"):
            value = getattr(config, attr, None)
            if value:
                paths.append(Path(str(value)))

    elif command == "radiomics":
        paths_cfg = getattr(config, "paths", None)
        if paths_cfg is not None:
            images_folder = getattr(paths_cfg, "images_folder", None)
            if images_folder:
                paths.append(Path(str(images_folder)))
        elif getattr(config, "images_folder", None):
            paths.append(Path(str(config.images_folder)))

    return paths


def _e2e_prerequisites_met(spec: PipelineConfigSpec) -> tuple[bool, str]:
    """
    Check whether local input data exists for an end-to-end CLI run.

    Returns:
        (ready, skip_reason) — when ``ready`` is False, ``skip_reason`` explains why.
    """
    try:
        loader = _schema_loader(spec)
        config = loader(str(spec.abs_path))
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot load config for E2E prerequisite check: {exc}"

    required = _collect_required_input_paths(config, spec)
    if not required:
        return True, ""

    missing = [p for p in required if not _path_exists(p)]
    if missing:
        preview = ", ".join(str(p) for p in missing[:3])
        suffix = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        return False, f"Input data not available on this machine: {preview}{suffix}"
    return True, ""


def _apply_skip_guards(spec: PipelineConfigSpec, *, check_e2e_data: bool = False) -> None:
    """Skip the current test when platform or prerequisite paths are missing."""
    if spec.skip_on_windows and sys.platform == "win32":
        pytest.skip(spec.skip_reason or "Skipped on Windows")
    for rel_dir in spec.skip_unless_paths_exist:
        if not (PROJECT_ROOT / rel_dir).exists():
            pytest.skip(spec.skip_reason or f"Missing prerequisite: {rel_dir}")
    if not spec.abs_path.is_file():
        pytest.skip(f"Config not found: {spec.abs_path}")
    if check_e2e_data:
        ready, reason = _e2e_prerequisites_met(spec)
        if not ready:
            pytest.skip(reason)


@pytest.mark.integration
class TestAllConfigSchemas:
    """Fast validation: every pipeline YAML must load into its typed schema."""

    @pytest.mark.parametrize(
        "spec",
        PIPELINE_CONFIGS,
        ids=[s.pytest_id for s in PIPELINE_CONFIGS],
    )
    def test_config_schema_loads(
        self,
        spec: PipelineConfigSpec,
        cwd_repo_root: None,
    ) -> None:
        _apply_skip_guards(spec)
        loader = _schema_loader(spec)
        config = loader(str(spec.abs_path))
        assert config is not None


@pytest.mark.integration
@pytest.mark.slow
class TestAllConfigCLI:
    """Full CLI smoke test: ``habit <command> -c <config>`` for every pipeline YAML."""

    @pytest.mark.parametrize(
        "spec",
        PIPELINE_CONFIGS,
        ids=[s.pytest_id for s in PIPELINE_CONFIGS],
    )
    def test_cli_run_exits_zero(
        self,
        spec: PipelineConfigSpec,
        cwd_repo_root: None,
    ) -> None:
        _apply_skip_guards(spec, check_e2e_data=True)
        runner = CliRunner()
        result = runner.invoke(cli, spec.cli_args(), catch_exceptions=False)
        assert result.exit_code == 0, (
            f"CLI failed for {spec.rel_path}\n"
            f"command: habit {' '.join(spec.cli_args())}\n"
            f"output:\n{result.output}"
        )


def test_pipeline_registry_not_empty() -> None:
    """Sanity check: discovery must find at least one runnable config."""
    assert len(PIPELINE_CONFIGS) >= 20, (
        f"Expected many pipeline configs, found {len(PIPELINE_CONFIGS)}"
    )
