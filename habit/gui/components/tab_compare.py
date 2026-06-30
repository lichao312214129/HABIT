# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Model comparison tab for Gradio GUI."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_compare import run_compare
from habit.core.machine_learning.config_schemas import ModelComparisonConfig
from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.path_picker import PathPickerRegistry
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    abs_path,
    extract_validation_msgs,
    load_config_yaml,
    load_gui_draft,
    open_directory,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    select_local_path,
    translate_pydantic_error,
)

MAX_MODELS: int = 6
LOAD_YAML_FIXED_OUTPUTS: int = 15


def _model_slot(index: int) -> Dict[str, Any]:
    """Create a fixed comparison model input slot."""
    letter = chr(65 + index)
    with gr.Group():
        enabled = gr.Checkbox(label=f"Enable model #{index + 1}", value=index < 2)
        path = gr.Textbox(label=f"path #{index + 1}", value="")
        with gr.Row():
            browse = gr.Button(f"Browse #{index + 1}", scale=1)
        model_name = gr.Textbox(label=f"model_name #{index + 1}", value=f"Model_{letter}")
        with gr.Row():
            subject_id_col = gr.Textbox(label=f"subject_id_col #{index + 1}", value="Subject")
            label_col = gr.Textbox(label=f"label_col #{index + 1}", value="label")
        with gr.Row():
            prob_col = gr.Textbox(label=f"prob_col #{index + 1}", value="probability")
            pred_col = gr.Textbox(label=f"pred_col #{index + 1} (optional)", value="predicted_label")
        split_col = gr.Textbox(label=f"split_col #{index + 1} (optional)", value="")
    return {
        "enabled": enabled,
        "path": path,
        "browse": browse,
        "model_name": model_name,
        "subject_id_col": subject_id_col,
        "label_col": label_col,
        "prob_col": prob_col,
        "pred_col": pred_col,
        "split_col": split_col,
    }


def render_compare_tab(
    demo: Optional[Any] = None,
    path_picker: PathPickerRegistry | None = None,
    project_root_state: Optional[Any] = None,
) -> None:
    """Render multi-model comparison tab."""
    _ = (demo, path_picker, project_root_state)
    gr.Markdown("Compare multiple model prediction CSVs (ROC, DCA, calibration, DeLong test).")

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing comparison YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    with gr.Row():
        output_dir = gr.Textbox(label="output_dir *", scale=4)
        out_btn = gr.Button("Browse", scale=1)

    gr.Markdown("### Model prediction files (enable at least 2)")
    slots: List[Dict[str, Any]] = [_model_slot(i) for i in range(MAX_MODELS)]

    with gr.Group():
        gr.Markdown("### Merged data and split")
        merged_enabled = gr.Checkbox(label="merged_data.enabled", value=True)
        merged_name = gr.Textbox(label="merged_data.save_name", value="combined_predictions.csv")
        split_enabled = gr.Checkbox(label="split.enabled", value=False)

    with gr.Group():
        gr.Markdown("### Visualization")
        plot_roc = gr.Checkbox(label="visualization.roc.enabled", value=True)
        plot_dca = gr.Checkbox(label="visualization.dca.enabled", value=True)
        plot_cal = gr.Checkbox(label="visualization.calibration.enabled", value=True)
        plot_pr = gr.Checkbox(label="visualization.pr_curve.enabled", value=True)
        cal_bins = gr.Number(label="visualization.calibration.n_bins", value=10, precision=0)

    with gr.Group():
        gr.Markdown("### Statistics")
        delong_enabled = gr.Checkbox(label="delong_test.enabled", value=True)
        basic_metrics = gr.Checkbox(label="metrics.basic_metrics.enabled", value=True)
        youden_metrics = gr.Checkbox(label="metrics.youden_metrics.enabled", value=True)
        target_metrics_enabled = gr.Checkbox(label="metrics.target_metrics.enabled", value=False)
        target_sensitivity = gr.Number(label="target sensitivity", value=0.7)
        target_specificity = gr.Number(label="target specificity", value=0.7)

    with gr.Row():
        submit_btn = gr.Button("Validate and run comparison", variant="primary")
        stop_btn = gr.Button("Stop", interactive=False)
    log_output = render_console_log(lines=15, elem_id="habit-log-compare")
    delong_table = gr.JSON(label="DeLong results (if enabled)")
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def browse_file() -> Any:
        p = select_local_path("file", "Select file")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    out_btn.click(browse_folder, outputs=output_dir)
    for slot in slots:
        slot["browse"].click(browse_file, outputs=slot["path"])

    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            # Keep fallback updates aligned with existing_yaml.change(..., outputs=load_yaml_outputs).
            # The fixed section has 15 widgets before model slots, and each slot contributes 8 widgets.
            return [noop] * (LOAD_YAML_FIXED_OUTPUTS + MAX_MODELS * 8)
        loaded = load_config_yaml(path)
        if not loaded:
            # Return the same number of updates as outputs to avoid Gradio output-count mismatch.
            return [noop] * (LOAD_YAML_FIXED_OUTPUTS + MAX_MODELS * 8)

        files = loaded.get("files_config", []) or []
        merged = loaded.get("merged_data", {}) or {}
        split = loaded.get("split", {}) or {}
        vis = loaded.get("visualization", {}) or {}
        delong = loaded.get("delong_test", {}) or {}
        metrics = loaded.get("metrics", {}) or {}
        basic = metrics.get("basic_metrics", {}) or {}
        youden = metrics.get("youden_metrics", {}) or {}
        target = metrics.get("target_metrics", {}) or {}
        targets = target.get("targets", {}) or {}

        updates: List[Any] = [
            loaded.get("output_dir", ""),
            merged.get("enabled", True),
            merged.get("save_name", "combined_predictions.csv"),
            split.get("enabled", False),
            vis.get("roc", {}).get("enabled", True),
            vis.get("dca", {}).get("enabled", True),
            vis.get("calibration", {}).get("enabled", True),
            vis.get("pr_curve", {}).get("enabled", True),
            int(vis.get("calibration", {}).get("n_bins", 10)),
            delong.get("enabled", True),
            basic.get("enabled", True),
            youden.get("enabled", True),
            target.get("enabled", False),
            float(targets.get("sensitivity", 0.7)),
            float(targets.get("specificity", 0.7)),
        ]

        for i in range(MAX_MODELS):
            if i < len(files):
                f = files[i]
                updates.extend([
                    True,
                    f.get("path", ""),
                    f.get("model_name") or f.get("name", f"Model_{i+1}"),
                    f.get("subject_id_col", "Subject"),
                    f.get("label_col", "label"),
                    f.get("prob_col", "probability"),
                    f.get("pred_col", "predicted_label"),
                    f.get("split_col", "") or "",
                ])
            else:
                updates.extend([False, "", f"Model_{chr(65+i)}", "Subject", "label", "probability", "predicted_label", ""])
        return updates

    slot_outputs: List[Any] = []
    for slot in slots:
        slot_outputs.extend([
            slot["enabled"], slot["path"], slot["model_name"],
            slot["subject_id_col"], slot["label_col"], slot["prob_col"], slot["pred_col"], slot["split_col"],
        ])

    load_yaml_outputs = [
        output_dir, merged_enabled, merged_name, split_enabled,
        plot_roc, plot_dca, plot_cal, plot_pr, cal_bins,
        delong_enabled, basic_metrics, youden_metrics, target_metrics_enabled,
        target_sensitivity, target_specificity,
    ] + slot_outputs

    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=load_yaml_outputs)

    def collect_slot_values(*args: Any) -> List[Dict[str, Any]]:
        """Unpack flat slot widget values into structured rows."""
        rows: List[Dict[str, Any]] = []
        chunk = 8
        for i in range(MAX_MODELS):
            base = i * chunk
            rows.append({
                "enabled": args[base],
                "path": args[base + 1],
                "model_name": args[base + 2],
                "subject_id_col": args[base + 3],
                "label_col": args[base + 4],
                "prob_col": args[base + 5],
                "pred_col": args[base + 6],
                "split_col": args[base + 7],
            })
        return rows

    def run_compare_pipeline(
        out_val: str,
        merged_en: bool,
        merged_save: str,
        split_en: bool,
        roc_en: bool,
        dca_en: bool,
        cal_en: bool,
        pr_en: bool,
        cal_bins_val: float,
        delong_en: bool,
        basic_en: bool,
        youden_en: bool,
        target_en: bool,
        target_sens: float,
        target_spec: float,
        *slot_args: Any,
    ):
        if not out_val:
            yield "❌ output_dir is required.", gr.update(visible=False), None
            return

        # Resolve relative paths to absolute before saving YAML.
        out_abs = abs_path(out_val)

        slot_rows = collect_slot_values(*slot_args)
        files_config: List[Dict[str, Any]] = []
        for row in slot_rows:
            if not row["enabled"]:
                continue
            if not row["path"]:
                yield "❌ Enabled model slot missing path.", gr.update(visible=False), None
                return
            entry: Dict[str, Any] = {
                "path": abs_path(row["path"]),
                "model_name": row["model_name"],
                "subject_id_col": row["subject_id_col"],
                "label_col": row["label_col"],
                "prob_col": row["prob_col"],
            }
            if row["pred_col"]:
                entry["pred_col"] = row["pred_col"]
            if row["split_col"]:
                entry["split_col"] = row["split_col"]
            files_config.append(entry)

        if len(files_config) < 2:
            yield "❌ Enable at least 2 models with valid paths.", gr.update(visible=False), None
            return

        config_data: Dict[str, Any] = {
            "output_dir": out_abs,
            "files_config": files_config,
            "merged_data": {"enabled": merged_en, "save_name": merged_save},
            "split": {"enabled": split_en},
            "visualization": {
                "roc": {"enabled": roc_en, "save_name": "roc_curves.png", "title": "ROC Curves"},
                "dca": {"enabled": dca_en, "save_name": "decision_curves.png", "title": "Decision Curves"},
                "calibration": {
                    "enabled": cal_en,
                    "save_name": "calibration_curves.png",
                    "title": "Calibration Curves",
                    "n_bins": int(cal_bins_val),
                },
                "pr_curve": {"enabled": pr_en, "save_name": "precision_recall_curves.png", "title": "Precision-Recall Curves"},
            },
            "delong_test": {"enabled": delong_en, "save_name": "delong_results.json"},
            "metrics": {
                "basic_metrics": {"enabled": basic_en},
                "youden_metrics": {"enabled": youden_en},
                "target_metrics": {
                    "enabled": target_en,
                    "targets": {
                        "sensitivity": float(target_sens),
                        "specificity": float(target_spec),
                    },
                },
            },
        }

        try:
            ModelComparisonConfig(**config_data)
            os.makedirs(out_abs, exist_ok=True)
            gui_path = str(Path(out_abs) / "config_compare_gui.yaml")
            save_config_yaml(config_data, gui_path)
            save_gui_draft("compare", gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running comparison...", gr.update(visible=False), None

            for log_text in run_background_job(
                run_compare,
                kwargs={"config_file": gui_path},
                log_file=Path(out_abs) / "processing.log",
            ):
                delong_data: Optional[Dict[str, Any]] = None
                if delong_en:
                    delong_path = Path(out_abs) / "delong_results.json"
                    if delong_path.exists():
                        with open(delong_path, "r", encoding="utf-8") as fh:
                            delong_data = json.load(fh)
                yield log_text, gr.update(visible=True), delong_data
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False), None
        except Exception as exc:  # noqa: BLE001
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs), gr.update(visible=False), None
            else:
                yield f"❌ Failed: {exc}", gr.update(visible=False), None

    submit_inputs = [
        output_dir, merged_enabled, merged_name, split_enabled,
        plot_roc, plot_dca, plot_cal, plot_pr, cal_bins,
        delong_enabled, basic_metrics, youden_metrics, target_metrics_enabled,
        target_sensitivity, target_specificity,
    ] + slot_outputs

    submit_btn.click(
        job_start_button_updates,
        outputs=[submit_btn, stop_btn],
    ).then(
        run_compare_pipeline,
        inputs=submit_inputs,
        outputs=[log_output, open_dir_btn, delong_table],
    ).then(
        job_end_button_updates,
        outputs=[submit_btn, stop_btn],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])
    open_dir_btn.click(lambda p: open_directory(p), inputs=output_dir)

    if demo is not None:
        # Restore only the YAML path; existing_yaml.change() then reloads all fields.
        def _restore_compare_path() -> str:
            return load_gui_draft("compare") or ""
        demo.load(_restore_compare_path, inputs=[], outputs=[existing_yaml])
