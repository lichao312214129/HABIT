import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pytest

from habit.core.common.service_configurator import ServiceConfigurator
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig, ResultColumns
from habit.core.habitat_analysis.strategies import get_strategy
from habit.utils.io_utils import get_image_and_mask_paths

_ONE_STEP_CACHE: Optional[Tuple[Path, List[str]]] = None
_OUTPUT_ROOT = Path("F:/work/habit_project/demo_data/results")
_CONFIG_PATH = Path("F:/work/habit_project/demo_data/config_habitat_one_step.yaml")


def _resolve_output_root(strategy_name: str) -> Path:
    """
    Resolve a persistent output root for tests.

    This test uses a fixed output directory so saved images can be inspected.
    """
    base_dir = _OUTPUT_ROOT / strategy_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _get_one_step_trained_pipeline() -> Tuple[Path, List[str]]:
    """
    Train a one-step pipeline once and reuse the saved artifact.

    This helper keeps the test suite simple: one test trains, another loads.
    """
    global _ONE_STEP_CACHE
    if _ONE_STEP_CACHE is not None and _ONE_STEP_CACHE[0].exists():
        return _ONE_STEP_CACHE

    config = HabitatAnalysisConfig.from_file(str(_CONFIG_PATH))
    config.verbose = False
    base_dir = _resolve_output_root("one_step")
    config.out_dir = str(base_dir / "train")

    assert config.HabitatsSegmention.clustering_mode == "one_step"

    images_paths, mask_paths = get_image_and_mask_paths(config.data_dir)
    if not images_paths or not mask_paths:
        pytest.skip("Demo data paths are missing for one_step pipeline.")

    configurator = ServiceConfigurator(config, logger=logging.getLogger("test_one_step_train"))
    analysis = configurator.create_habitat_analysis()

    subjects = list(images_paths.keys())
    results = analysis.run(subjects=subjects, save_results_csv=True)

    assert not results.empty
    assert ResultColumns.HABITATS in results.columns

    pipeline_path = Path(config.out_dir) / "habitat_pipeline.pkl"
    assert pipeline_path.exists()

    _ONE_STEP_CACHE = (pipeline_path, subjects)
    return _ONE_STEP_CACHE


def test_one_step_train_pipeline() -> None:
    """Train a one-step pipeline and verify the artifact is saved."""
    pipeline_path, _ = _get_one_step_trained_pipeline()
    assert pipeline_path.exists()


def test_one_step_predict_with_trained_pipeline() -> None:
    """
    Load a trained one-step pipeline and run prediction.

    This test validates that a pre-trained pipeline can be loaded and used
    in transform-only mode with updated output settings.
    """
    # Load pipeline path (must exist from training)
    base_dir = _resolve_output_root("one_step")
    pipeline_path = base_dir / "train" / "habitat_pipeline.pkl"
    
    if not pipeline_path.exists():
        pytest.skip(f"Pipeline not found at {pipeline_path}. Run test_one_step_train_pipeline first.")
    
    # Load config and get subjects
    config = HabitatAnalysisConfig.from_file(str(_CONFIG_PATH))
    assert config.HabitatsSegmention.clustering_mode == "one_step"
    
    images_paths, mask_paths = get_image_and_mask_paths(config.data_dir)
    if not images_paths or not mask_paths:
        pytest.skip("Demo data paths are missing for one_step pipeline.")
    subjects = list(images_paths.keys())

    # Setup prediction config
    predict_config = HabitatAnalysisConfig.from_file(str(_CONFIG_PATH))
    predict_config.out_dir = str(base_dir / "predict")
    predict_config.plot_curves = False
    predict_config.save_images = True

    predict_configurator = ServiceConfigurator(
        predict_config, logger=logging.getLogger("test_one_step_predict_infer")
    )
    predict_analysis = predict_configurator.create_habitat_analysis()

    strategy_cls = get_strategy(predict_config.HabitatsSegmention.clustering_mode)
    strategy = strategy_cls(predict_analysis)
    predict_results = strategy.run(
        subjects=subjects,
        save_results_csv=True,
        load_from=str(pipeline_path)
    )

    assert not predict_results.empty
    assert ResultColumns.HABITATS in predict_results.columns



if __name__ == '__main__':
    test_one_step_predict_with_trained_pipeline()