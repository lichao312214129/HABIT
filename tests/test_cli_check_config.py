"""Tests for the clinician-facing ``habit check-config`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner
import yaml
from yaml.constructor import ConstructorError

from habit.cli import cli


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects ambiguous duplicate YAML mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    """Construct one mapping and raise when a key appears more than once."""
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key!r})",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def test_all_config_yaml_files_have_unique_keys() -> None:
    """Every config YAML parses without duplicate mapping keys."""
    yaml_files = sorted((PROJECT_ROOT / "config").rglob("*.yaml"))

    assert yaml_files
    for yaml_file in yaml_files:
        yaml.load(
            yaml_file.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )


def test_check_config_validates_workflow_schema() -> None:
    """A runnable workflow config passes syntax and typed-schema validation."""
    config_path = PROJECT_ROOT / "config/preprocessing/config_preprocessing_minimal.yaml"
    result = CliRunner().invoke(
        cli,
        ["check-config", "-c", str(config_path), "-w", "preprocess"],
    )

    assert result.exit_code == 0, result.output
    assert "配置检查通过" in result.output
    assert "workflow=preprocess" in result.output


def test_check_config_supports_standalone_yaml_syntax_only() -> None:
    """A PyRadiomics preset can be checked without applying a workflow schema."""
    preset_path = PROJECT_ROOT / "config/radiomics/parameter.yaml"
    result = CliRunner().invoke(
        cli,
        ["check-config", "-c", str(preset_path), "--syntax-only"],
    )

    assert result.exit_code == 0, result.output
    assert "YAML syntax check passed" in result.output


def test_check_config_explains_yaml_syntax_errors(tmp_path: Path) -> None:
    """Malformed YAML returns a bilingual diagnostic with a nearby line number."""
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("input:\n\tpath: broken\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["check-config", "-c", str(invalid_config), "--syntax-only"],
    )

    assert result.exit_code == 1
    assert "YAML 修改提示" in result.output
    assert "出错位置约在" in result.output
