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
def test_show_versions_returns_software_fingerprint() -> None:
    """``show_versions`` is callable and mirrors ``software_fingerprint``."""
    import habit
    from habit.contracts.provenance import software_fingerprint

    versions = habit.show_versions()
    assert callable(habit.show_versions)
    assert isinstance(versions, dict)
    assert "habit" in versions
    assert isinstance(versions["habit"], str)
    assert versions["habit"]
    assert all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in versions.items()
    )
    assert versions == dict(software_fingerprint())


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
def test_kernels_and_viz_symbols_are_registered() -> None:
    """The registry mirrors ``habit.kernels.__all__`` and ``habit.viz.__all__``."""
    import habit.kernels
    import habit.viz

    registered = set(PUBLIC_API_SYMBOLS)
    assert set(habit.kernels.__all__) <= registered
    assert set(habit.viz.__all__) <= registered


@pytest.mark.unit
def test_compat_interop_symbols_are_registered() -> None:
    """The sklearn/MONAI/nnU-Net interop entry points are public API."""
    import habit.compat.monai
    import habit.compat.nnunet

    registered = set(PUBLIC_API_SYMBOLS)
    assert {"as_estimator", "as_transformer", "as_classifier"} <= registered
    assert set(habit.compat.monai.__all__) <= registered
    assert set(habit.compat.nnunet.__all__) <= registered


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
    # NotFittedError has ONE source (habit.exceptions) per the v1.0 plan; it
    # subclasses sklearn's so sklearn interop isinstance checks keep working.
    assert issubclass(NotFittedError, SklearnNotFittedError)
    assert issubclass(NotFittedError, HabitError)


@pytest.mark.unit
def test_run_preprocess_delegates_to_core_runner() -> None:
    """Public preprocessing accepts a dictionary and validates it before delegation."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import habit

    with patch("habit.compat.preprocess_runner.run_preprocess_from_config") as mock_run:
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

    with patch("habit.compat.preprocess_runner.run_preprocess_from_config") as mock_run:
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
        "habit.compat.feature_extraction_loader."
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
    """Public feature runner forwards plugin settings into the L4 recipe."""
    from unittest.mock import MagicMock, patch

    import habit

    plugins = {"graph": MagicMock()}
    with patch(
        "habit.recipes.features.extract_habitat_features"
    ) as mock_run:
        mock_run.return_value = MagicMock(run_id="extract-run")
        habit.run_feature_extraction(
            {
                "raw_img_folder": "raw",
                "habitats_map_folder": "habitats",
                "out_dir": "features",
                "feature_types": ["non_radiomics", "graph"],
            },
            plugin_configs=plugins,
        )

    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["plugin_configs"] == plugins
    assert mock_run.call_args.kwargs["logger"] is None
    config_arg = mock_run.call_args.args[0]
    assert isinstance(config_arg, dict)
    assert "graph" in config_arg["feature_types"]


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
            "habit.compat.test_retest_mapper.find_habitat_mapping",
            return_value=expected_mapping,
        ) as mock_find,
        patch("habit.compat.test_retest_mapper.batch_process_files") as mock_batch,
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
