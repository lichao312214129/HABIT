# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Contract tests for symbols exposed by ``import habit``."""

from __future__ import annotations

import subprocess
import sys

import pytest

from habit.api.registry import PUBLIC_API_SYMBOLS


@pytest.mark.unit
def test_public_api_symbols_match_registry() -> None:
    """``habit.__all__`` must stay aligned with the canonical registry."""
    import habit

    assert habit.__all__[0] == "__version__"
    assert set(habit.__all__[1:]) == set(PUBLIC_API_SYMBOLS)


@pytest.mark.unit
def test_api_subpackage_exports_the_registered_symbols() -> None:
    """The documented ``habit.api`` facade must match the top-level API."""
    import habit.api

    assert set(habit.api.__all__) == set(PUBLIC_API_SYMBOLS)
    for symbol in PUBLIC_API_SYMBOLS:
        assert getattr(habit.api, symbol) is getattr(__import__("habit"), symbol)


@pytest.mark.unit
def test_version_is_string() -> None:
    """``habit.__version__`` is available without lazy resolution."""
    import habit

    assert isinstance(habit.__version__, str)
    assert habit.__version__


@pytest.mark.unit
@pytest.mark.parametrize("symbol", PUBLIC_API_SYMBOLS)
def test_public_symbol_importable(symbol: str) -> None:
    """Every registered public symbol resolves on ``getattr(habit, ...)``."""
    import habit

    obj = getattr(habit, symbol)
    assert obj is not None


@pytest.mark.unit
def test_import_habit_does_not_load_radiomics() -> None:
    """Bare ``import habit`` in a fresh interpreter must not import PyRadiomics."""
    import subprocess
    import sys

    script = (
        "import sys, habit\n"
        "assert habit.__version__\n"
        "print('radiomics_loaded', 'radiomics' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout or "") + (
        completed.stderr or ""
    )
    assert "radiomics_loaded False" in completed.stdout


@pytest.mark.unit
def test_run_preprocess_delegates_to_core_runner() -> None:
    """Public runner is an alias of the existing core entry point."""
    from unittest.mock import MagicMock, patch

    import habit

    with patch("habit.core.preprocessing.run.run_preprocess_from_config") as mock_run:
        config = MagicMock()
        logger = MagicMock()
        habit.run_preprocess(config, logger=logger)
        mock_run.assert_called_once_with(config, logger=logger)


@pytest.mark.unit
def test_load_feature_extraction_config_delegates_to_plugin_aware_loader() -> None:
    """Public loader preserves optional plugin configuration blocks from YAML."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import habit

    expected_config = MagicMock()
    expected_plugins = {"graph": MagicMock()}
    with patch(
        "habit.core.habitat_analysis.feature_extraction_loader."
        "load_feature_extraction_config_from_file",
        return_value=(expected_config, expected_plugins),
    ) as mock_load:
        actual_config, actual_plugins = habit.load_feature_extraction_config(
            Path("feature_config.yaml")
        )

    mock_load.assert_called_once_with(Path("feature_config.yaml"))
    assert actual_config is expected_config
    assert actual_plugins is expected_plugins


@pytest.mark.unit
def test_run_feature_extraction_passes_plugin_configs() -> None:
    """Public feature runner forwards optional plugin settings to the core runner."""
    from unittest.mock import MagicMock, patch

    import habit

    config = MagicMock()
    plugins = {"graph": MagicMock()}
    with patch(
        "habit.core.habitat_analysis.run.run_feature_extraction_from_config"
    ) as mock_run:
        habit.run_feature_extraction(config, plugin_configs=plugins)

    mock_run.assert_called_once_with(
        config,
        logger=None,
        plugin_configs=plugins,
    )


@pytest.mark.unit
def test_run_test_retest_analysis_maps_and_processes_images() -> None:
    """Public test-retest runner must match labels before writing mapped images."""
    from unittest.mock import MagicMock, patch

    import habit

    config = MagicMock()
    config.test_habitat_table = "test.csv"
    config.retest_habitat_table = "retest.csv"
    config.features = ["feature_a"]
    config.similarity_method = "pearson"
    config.input_dir = "input"
    config.out_dir = "output"
    config.processes = 2
    expected_mapping = {2: 1}
    logger = MagicMock()

    with patch(
        "habit.core.machine_learning.feature_selectors.icc."
        "habitat_test_retest_mapper.find_habitat_mapping",
        return_value=expected_mapping,
    ) as mock_find, patch(
        "habit.core.machine_learning.feature_selectors.icc."
        "habitat_test_retest_mapper.batch_process_files"
    ) as mock_batch:
        actual_mapping = habit.run_test_retest_analysis(config, logger=logger)

    assert actual_mapping == expected_mapping
    mock_find.assert_called_once_with(
        "test.csv",
        "retest.csv",
        ["feature_a"],
        "pearson",
    )
    mock_batch.assert_called_once_with("input", expected_mapping, "output", 2)
    logger.info.assert_called_once_with(
        "Computed test-retest habitat mapping: %s",
        expected_mapping,
    )


@pytest.mark.unit
def test_config_from_file_via_public_api(tmp_path, cwd_repo_root: None) -> None:
    """Config classes exposed at top level retain ``from_file`` factory."""
    from pathlib import Path

    import habit

    cfg_path = Path("config/preprocessing/config_preprocessing_demo.yaml")
    if not cfg_path.is_file():
        pytest.skip(f"Demo config not found: {cfg_path}")

    config = habit.PreprocessingConfig.from_file(str(cfg_path))
    assert config.out_dir
