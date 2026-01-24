import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

from habit.core.common.service_configurator import ServiceConfigurator
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig, ResultColumns
from habit.core.habitat_analysis.strategies import get_strategy
from habit.utils.io_utils import get_image_and_mask_paths


def _resolve_output_root(output_root: Optional[str], strategy_name: str) -> Path:
    """
    Resolve output root directory for train/predict runs.

    If output_root is not provided, use HABITAT_RUN_OUTPUT_DIR or fallback to ./demo_data/results.
    """
    resolved_root = output_root or os.environ.get("HABITAT_RUN_OUTPUT_DIR") or "demo_data/results"
    base_dir = Path(resolved_root) / strategy_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _load_subjects(config: HabitatAnalysisConfig) -> List[str]:
    """
    Load subject ids from the configured data directory.
    """
    images_paths, mask_paths = get_image_and_mask_paths(config.data_dir)
    if not images_paths or not mask_paths:
        raise FileNotFoundError("Demo data paths are missing for habitat pipeline.")
    return list(images_paths.keys())


def train_pipeline(config_path: str, output_root: Optional[str]) -> Path:
    """
    Train a habitat pipeline and return the saved pipeline path.
    """
    config = HabitatAnalysisConfig.from_file(config_path)
    strategy_name = config.HabitatsSegmention.clustering_mode
    base_dir = _resolve_output_root(output_root, strategy_name)
    config.out_dir = str(base_dir / "train")
    config.verbose = False

    subjects = _load_subjects(config)

    configurator = ServiceConfigurator(config, logger=logging.getLogger("habitat_train"))
    analysis = configurator.create_habitat_analysis()

    strategy_cls = get_strategy(config.HabitatsSegmention.clustering_mode)
    strategy = strategy_cls(analysis)
    results = strategy.run(subjects=subjects, save_results_csv=True)

    if results.empty or ResultColumns.HABITATS not in results.columns:
        raise ValueError("Training produced empty results or missing habitat labels.")

    pipeline_path = Path(config.out_dir) / "habitat_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError("Training completed but pipeline file was not saved.")

    return pipeline_path


def predict_with_pipeline(
    config_path: str,
    output_root: Optional[str],
    pipeline_path: Optional[str],
) -> None:
    """
    Load a trained pipeline and run prediction.
    """
    config = HabitatAnalysisConfig.from_file(config_path)
    strategy_name = config.HabitatsSegmention.clustering_mode
    base_dir = _resolve_output_root(output_root, strategy_name)

    pipeline = Path(pipeline_path) if pipeline_path else base_dir / "train" / "habitat_pipeline.pkl"
    if not pipeline.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline}. Run train first.")

    predict_config = HabitatAnalysisConfig.from_file(config_path)
    predict_config.out_dir = str(base_dir / "predict")
    predict_config.plot_curves = False
    predict_config.save_images = True

    subjects = _load_subjects(predict_config)

    configurator = ServiceConfigurator(
        predict_config, logger=logging.getLogger("habitat_predict")
    )
    analysis = configurator.create_habitat_analysis()

    strategy_cls = get_strategy(predict_config.HabitatsSegmention.clustering_mode)
    strategy = strategy_cls(analysis)
    results = strategy.run(
        subjects=subjects,
        save_results_csv=True,
        load_from=str(pipeline)
    )

    if results.empty or ResultColumns.HABITATS not in results.columns:
        raise ValueError("Prediction produced empty results or missing habitat labels.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run habitat analysis train/predict.")
    parser.add_argument("--config", required=True, help="Path to habitat config YAML.")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Run mode: train or predict.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for outputs (defaults to HABITAT_RUN_OUTPUT_DIR or demo_data/results).",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Path to a trained pipeline file (predict mode only).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        pipeline_path = train_pipeline(args.config, args.output_root)
        print(f"Pipeline saved to: {pipeline_path}")
    else:
        predict_with_pipeline(args.config, args.output_root, args.pipeline)
        print("Prediction completed.")


if __name__ == "__main__":
    main()
