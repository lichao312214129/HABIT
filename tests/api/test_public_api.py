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
def test_public_exceptions_share_one_documented_hierarchy() -> None:
    """Public callers can catch stable HABIT errors without deep imports."""
    import habit
    from habit.exceptions import DataFormatError, HabitError, NotFittedError
    from sklearn.exceptions import NotFittedError as SklearnNotFittedError

    assert issubclass(habit.HABITAPIError, DataFormatError)
    assert issubclass(habit.HABITAPIError, HabitError)
    assert NotFittedError is SklearnNotFittedError


@pytest.mark.unit
def test_run_preprocess_delegates_to_core_runner() -> None:
    """Public preprocessing accepts a dictionary and validates it before delegation."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import habit

    with patch("habit.core.preprocessing.run.run_preprocess_from_config") as mock_run:
        logger = MagicMock()
        result = habit.run_preprocess(
            {"data_dir": "input", "out_dir": "output"},
            logger=logger,
        )

    delegated_config = mock_run.call_args.args[0]
    assert isinstance(delegated_config, habit.PreprocessingConfig)
    assert delegated_config.data_dir == "input"
    mock_run.assert_called_once_with(delegated_config, logger=logger)
    assert result.artifact("output_dir") == Path("output")


@pytest.mark.unit
def test_public_runner_rejects_invalid_dictionary_before_core_execution() -> None:
    """Dictionary validation must happen at the public API boundary."""
    from unittest.mock import patch

    import habit

    with patch("habit.core.preprocessing.run.run_preprocess_from_config") as mock_run:
        with pytest.raises(habit.ConfigurationError):
            habit.run_preprocess({"data_dir": "input"})

    mock_run.assert_not_called()


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
    """Public feature runner accepts dictionaries and explicit plugin settings."""
    from unittest.mock import MagicMock, patch

    import habit

    plugins = {"graph": MagicMock()}
    with patch(
        "habit.core.habitat_analysis.run.run_feature_extraction_from_config"
    ) as mock_run:
        habit.run_feature_extraction(
            {
                "raw_img_folder": "raw",
                "habitats_map_folder": "habitats",
                "out_dir": "features",
                "feature_types": ["non_radiomics"],
            },
            plugin_configs=plugins,
        )

    delegated_config = mock_run.call_args.args[0]
    assert isinstance(delegated_config, habit.FeatureExtractionConfig)
    mock_run.assert_called_once_with(
        delegated_config,
        logger=None,
        plugin_configs=plugins,
    )


@pytest.mark.unit
def test_run_test_retest_analysis_maps_and_processes_images() -> None:
    """Public test-retest runner validates mappings before writing images."""
    from unittest.mock import MagicMock, patch

    import habit

    config = {
        "test_habitat_table": "test.csv",
        "retest_habitat_table": "retest.csv",
        "features": ["feature_a"],
        "similarity_method": "pearson",
        "input_dir": "input",
        "out_dir": "output",
        "processes": 2,
    }
    expected_mapping = {2: 1}
    logger = MagicMock()

    with (
        patch(
            "habit.core.machine_learning.feature_selectors.icc."
            "habitat_test_retest_mapper.find_habitat_mapping",
            return_value=expected_mapping,
        ) as mock_find,
        patch(
            "habit.core.machine_learning.feature_selectors.icc."
            "habitat_test_retest_mapper.batch_process_files"
        ) as mock_batch,
    ):
        result = habit.run_test_retest_analysis(config, logger=logger)

    assert result.data == expected_mapping
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
