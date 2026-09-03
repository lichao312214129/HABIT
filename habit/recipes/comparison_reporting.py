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
"""Filesystem sinks for model-comparison artefacts (L4).

Pure plotters live in :mod:`habit.viz.classification`. Domain evaluation lives
in :mod:`habit.evaluation.comparison`. This module turns a
:class:`~habit.evaluation.comparison.ComparisonResult` into the on-disk
layout expected by ``habit compare`` (combined CSV, per-split figures,
``delong_results.json``, ``metrics/metrics.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from habit.evaluation.comparison import ComparisonResult, ModelArrays
from habit.evaluation.panel import clean_binary_predictions
from habit.viz.classification import (
    plot_calibration,
    plot_decision_curve,
    plot_precision_recall,
    plot_roc,
)

__all__ = [
    "write_comparison_artifacts",
    "write_comparison_figures",
    "write_json",
]


def write_comparison_artifacts(
    result: ComparisonResult,
    destination: Union[str, Path],
    *,
    visualization: Any = None,
    merged_save_name: str = "combined_predictions.csv",
    write_merged: bool = True,
    delong_save_name: str = "delong_results.json",
    write_delong: bool = True,
    write_metrics: bool = True,
    split_enabled: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """
    Persist a :class:`ComparisonResult` under ``destination``.

    Args:
        result: In-memory comparison outcome.
        destination: Output root directory.
        visualization: Optional comparison visualization config namespace
            (``roc`` / ``dca`` / ``calibration`` / ``pr_curve`` items).
        merged_save_name: Filename for the combined prediction CSV.
        write_merged: Whether to write the merged CSV.
        delong_save_name: Filename for per-group DeLong JSON.
        write_delong: Whether to write DeLong JSON files.
        write_metrics: Whether to write ``metrics/metrics.json``.
        split_enabled: When True, figures/DeLong land under per-split subdirs.
        logger: Optional logger.

    Returns:
        Mapping of artefact key -> written path.
    """
    log = logger or logging.getLogger(__name__)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    if write_merged:
        merged_path = root / merged_save_name
        result.merged.frame.to_csv(merged_path, index=False)
        artifacts["combined_predictions"] = merged_path
        log.info("Merged predictions written to %s", merged_path)

    for group_name, models_data in result.groups.items():
        if split_enabled and result.merged.split_column:
            group_dir = root / str(group_name)
            group_label: Optional[str] = str(group_name)
        else:
            group_dir = root
            group_label = None
        group_dir.mkdir(parents=True, exist_ok=True)

        if visualization is not None:
            written = write_comparison_figures(
                models_data,
                group_dir,
                visualization=visualization,
                group_name=group_label,
                logger=log,
            )
            artifacts.update(written)

        if write_delong:
            rows = result.delong_by_group.get(str(group_name), ())
            if rows:
                delong_path = group_dir / delong_save_name
                write_json(delong_path, list(rows))
                artifacts[f"delong:{group_name}"] = delong_path
                log.info("DeLong results written to %s", delong_path)

    if write_metrics and result.metrics:
        metrics_path = root / "metrics" / "metrics.json"
        write_json(metrics_path, result.metrics)
        artifacts["metrics"] = metrics_path
        log.info("Metrics written to %s", metrics_path)

    return artifacts


def write_comparison_figures(
    models_data: Mapping[str, ModelArrays],
    group_dir: Union[str, Path],
    *,
    visualization: Any,
    group_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """
    Render multi-model ROC / DCA / calibration / PR via ``habit.viz``.

    Args:
        models_data: Model name -> ``(y_true, y_prob, y_pred?)``.
        group_dir: Directory that receives the PDF files.
        visualization: Config namespace with per-plot enable/title/save_name.
        group_name: Optional split label used only in default titles.
        logger: Optional logger.

    Returns:
        Mapping of figure key -> written path.
    """
    log = logger or logging.getLogger(__name__)
    out_dir = Path(group_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = _curves_panel(models_data)
    if not curves:
        log.warning("No finite prediction pairs available for figures in %s", out_dir)
        return {}

    title_suffix = f"[{group_name}] " if group_name else ""
    written: Dict[str, Path] = {}

    def _save(fig: Any, filename: str, key: str) -> None:
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight")
        # No optional-dependency gate here on purpose: ``fig`` was produced by
        # habit.viz, which already went through require("matplotlib"), so
        # matplotlib is provably importable at this point. The except clause
        # only guards against a backend teardown error while closing.
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # noqa: BLE001
            pass
        written[key] = path
        log.info("Figure written to %s", path)

    roc_cfg = visualization.roc
    if getattr(roc_cfg, "enabled", True):
        title = roc_cfg.title or f"{title_suffix}ROC Curves Comparison"
        fig = plot_roc(curves=curves, title=str(title))
        _save(fig, roc_cfg.save_name or "roc_curves.pdf", f"roc:{group_name}")

    dca_cfg = visualization.dca
    if getattr(dca_cfg, "enabled", True):
        title = dca_cfg.title or f"{title_suffix}Decision Curve"
        fig = plot_decision_curve(curves=curves, title=str(title))
        _save(
            fig,
            dca_cfg.save_name or "decision_curves.pdf",
            f"dca:{group_name}",
        )

    cal_cfg = visualization.calibration
    if getattr(cal_cfg, "enabled", True):
        title = cal_cfg.title or f"{title_suffix}Calibration Curves"
        n_bins = int(cal_cfg.n_bins or 10)
        fig = plot_calibration(curves=curves, title=str(title), n_bins=n_bins)
        _save(
            fig,
            cal_cfg.save_name or "calibration_curves.pdf",
            f"calibration:{group_name}",
        )

    pr_cfg = visualization.pr_curve
    if getattr(pr_cfg, "enabled", True):
        title = pr_cfg.title or f"{title_suffix}Precision-Recall Curves"
        fig = plot_precision_recall(curves=curves, title=str(title))
        _save(
            fig,
            pr_cfg.save_name or "precision_recall_curves.pdf",
            f"pr:{group_name}",
        )

    return written


def write_json(path: Union[str, Path], payload: Any) -> Path:
    """
    Write JSON with NumPy scalars coerced to Python builtins.

    Args:
        path: Destination file path.
        payload: JSON-serialisable (after coercion) object.

    Returns:
        Resolved path that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=4, ensure_ascii=False)
    return out


def _curves_panel(models_data: Mapping[str, ModelArrays]) -> Dict[str, tuple]:
    """Build a name -> (y_true, y_prob) panel with NaN rows dropped per model."""
    panel: Dict[str, tuple] = {}
    for name, (y_true, y_prob, _) in models_data.items():
        cleaned = clean_binary_predictions(y_true, y_prob)
        if cleaned.y_true.size < 2:
            continue
        if np.unique(cleaned.y_true).size < 2:
            continue
        panel[name] = (cleaned.y_true, cleaned.y_prob)
    return panel


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy / Path values for ``json.dump``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
