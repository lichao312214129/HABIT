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
"""v0↔v1 migration equivalence tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from habit.cli import cli
from habit.spec.legacy import LegacyConfigAdapter, migrate_yaml, validate_v1_document
from habit.spec.specs import HabitatSpec
from tests.cloud_behavior.helpers import (
    habitat_label_digest,
    parse_yaml,
    run_two_step_on_tree,
    spec_from_v0_payload,
)
from tests.cloud_behavior.synthetic_data import minimal_v0_habitat_yaml


@pytest.mark.integration
def test_v0_v1_migration_spec_fingerprint_and_labels_match(
    synthetic_tree: Path,
    v0_config_path: Path,
    tmp_path: Path,
) -> None:
    """
    Migrating the minimal v0 config must yield an equivalent v1 document.

    ``validate_v1_document`` must pass on the migrated YAML, the v1 spec
    fingerprint must equal the v0 translation fingerprint, and both specs
    must produce identical habitat label digests on the synthetic cohort.
    """
    v0_payload = parse_yaml(v0_config_path)
    adapter = LegacyConfigAdapter()
    v0_translation = adapter.translate(v0_payload, "habitat")
    validate_v1_document(v0_translation.document, workflow="habitat")
    spec_v0 = HabitatSpec.from_dict(v0_translation.document["spec"])

    v1_path = tmp_path / "config_habitat_minimal.v1.yaml"
    migrate_yaml(v0_config_path, v1_path, workflow="habitat")
    v1_payload = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    assert isinstance(v1_payload, dict)
    validate_v1_document(v1_payload, workflow="habitat")
    spec_v1 = HabitatSpec.from_dict(v1_payload["spec"])

    assert spec_v0.fingerprint() == spec_v1.fingerprint()

    result_v0 = run_two_step_on_tree(synthetic_tree, spec_v0)
    result_v1 = run_two_step_on_tree(synthetic_tree, spec_v1)
    assert habitat_label_digest(result_v0.habitat_maps) == habitat_label_digest(
        result_v1.habitat_maps
    )


@pytest.mark.integration
def test_migrate_config_cli_produces_valid_v1_document(
    synthetic_tree: Path,
    tmp_path: Path,
) -> None:
    """
    ``habit migrate-config`` writes a v1 file that passes ``validate_v1_document``.

    Args:
        synthetic_tree: Synthetic cohort root fixture.
        tmp_path: Pytest temporary directory.
    """
    v0_path = tmp_path / "config_habitat_minimal.yaml"
    v0_path.write_text(
        minimal_v0_habitat_yaml(synthetic_tree, tmp_path / "out"),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["migrate-config", "-c", str(v0_path)])
    assert result.exit_code == 0, result.output

    v1_path = tmp_path / "config_habitat_minimal.v1.yaml"
    v1_payload = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    assert isinstance(v1_payload, dict)
    validate_v1_document(v1_payload, workflow="habitat")

    # Cross-check against the in-memory adapter used by the recipe layer.
    assert HabitatSpec.from_dict(v1_payload["spec"]).fingerprint() == spec_from_v0_payload(
        parse_yaml(v0_path)
    ).fingerprint()
