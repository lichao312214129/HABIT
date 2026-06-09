# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Habitat feature extraction tab for Gradio GUI."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_extract_features import run_extract_features
from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    abs_path,
    extract_validation_msgs,
    load_config_yaml,
    open_directory,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
)

FEATURE_TYPE_CHOICES: List[str] = [
    "traditional",
    "non_radiomics",
    "whole_habitat",
    "each_habitat",
    "msi",
    "ith_score",
]


def render_extract_tab() -> None:
    """Render habitat feature extraction tab."""
    gr.Markdown(
        "Extract radiomics and habitat metrics from habitat maps (`*_habitats.nrrd`) "
        "and original images; outputs CSV tables for machine learning."
    )

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing extraction YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Paths")
        with gr.Row():
            raw_img_folder = gr.Textbox(label="raw_img_folder *", scale=4)
            raw_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            habitats_map_folder = gr.Textbox(label="habitats_map_folder *", scale=4)
            hab_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="out_dir *", scale=4)
            out_btn = gr.Button("Browse", scale=1)

    with gr.Group():
        gr.Markdown("### 2. Radiomics parameter files")
        with gr.Row():
            params_non_hab = gr.Textbox(
                label="params_file_of_non_habitat *",
                value="../radiomics/parameter.yaml",
            )
            params_hab = gr.Textbox(
                label="params_file_of_habitat *",
                value="../radiomics/parameter_habitat.yaml",
            )

    with gr.Group():
        gr.Markdown("### 3. Extraction controls")
        with gr.Row():
            habitat_pattern = gr.Textbox(label="habitat_pattern *", value="*_habitats.nrrd")
            n_processes = gr.Number(label="n_processes", value=4, minimum=1, maximum=32, step=1)
        feature_types = gr.CheckboxGroup(
            label="feature_types *",
            choices=FEATURE_TYPE_CHOICES,
            value=["traditional", "non_radiomics", "whole_habitat", "msi", "ith_score"],
        )
        n_habitats = gr.Number(
            label="n_habitats (0 = auto-detect)",
            value=0,
            minimum=0,
            maximum=20,
            step=1,
        )
        debug = gr.Checkbox(label="debug", value=False)

    submit_btn = gr.Button("Validate and run feature extraction", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=15, interactive=False)
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def browse_file() -> Any:
        p = select_local_path("file", "Select YAML")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    raw_btn.click(browse_folder, outputs=raw_img_folder)
    hab_btn.click(browse_folder, outputs=habitats_map_folder)
    out_btn.click(browse_folder, outputs=out_dir)

    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            return [noop] * 11
        loaded = load_config_yaml(path)
        if not loaded:
            return [noop] * 11
        nh = loaded.get("n_habitats")
        return [
            loaded.get("raw_img_folder", ""),
            loaded.get("habitats_map_folder", ""),
            loaded.get("out_dir", ""),
            loaded.get("params_file_of_non_habitat", "../radiomics/parameter.yaml"),
            loaded.get("params_file_of_habitat", "../radiomics/parameter_habitat.yaml"),
            loaded.get("habitat_pattern", "*_habitats.nrrd"),
            int(loaded.get("n_processes", 4)),
            loaded.get("feature_types", FEATURE_TYPE_CHOICES),
            int(nh) if nh else 0,
            loaded.get("debug", False),
        ]

    existing_yaml.change(
        load_yaml,
        inputs=existing_yaml,
        outputs=[
            raw_img_folder, habitats_map_folder, out_dir,
            params_non_hab, params_hab, habitat_pattern, n_processes,
            feature_types, n_habitats, debug,
        ],
    )

    def run_extract(
        raw_val: str,
        hab_val: str,
        out_val: str,
        p_non: str,
        p_hab: str,
        pattern: str,
        n_proc: float,
        feats: List[str],
        n_hab: float,
        debug_val: bool,
    ):
        if not raw_val or not hab_val or not out_val:
            yield "❌ raw_img_folder, habitats_map_folder, and out_dir are required.", gr.update(visible=False)
            return
        if not feats:
            yield "❌ Select at least one feature_types entry.", gr.update(visible=False)
            return

        # Resolve relative paths to absolute before saving to YAML.
        out_abs = abs_path(out_val)

        config_data: Dict[str, Any] = {
            "raw_img_folder": abs_path(raw_val),
            "habitats_map_folder": abs_path(hab_val),
            "out_dir": out_abs,
            "params_file_of_non_habitat": abs_path(p_non) if p_non and p_non.strip() else p_non,
            "params_file_of_habitat": abs_path(p_hab) if p_hab and p_hab.strip() else p_hab,
            "habitat_pattern": pattern,
            "n_processes": int(n_proc),
            "feature_types": list(feats),
            "n_habitats": int(n_hab) if int(n_hab) > 0 else None,
            "debug": debug_val,
        }

        try:
            FeatureExtractionConfig(**config_data)
            os.makedirs(out_abs, exist_ok=True)
            gui_path = str(Path(out_abs) / "config_extract_gui.yaml")
            save_config_yaml(config_data, gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running extraction...", gr.update(visible=False)
            for log_text in run_background_job(
                run_extract_features,
                kwargs={"config_file": gui_path},
                log_file=Path(out_abs) / "processing.log",
            ):
                yield log_text, gr.update(visible=True)
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs), gr.update(visible=False)
            else:
                yield f"❌ Failed: {exc}", gr.update(visible=False)

    submit_btn.click(
        run_extract,
        inputs=[
            raw_img_folder, habitats_map_folder, out_dir,
            params_non_hab, params_hab, habitat_pattern, n_processes,
            feature_types, n_habitats, debug,
        ],
        outputs=[log_output, open_dir_btn],
    )
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)
