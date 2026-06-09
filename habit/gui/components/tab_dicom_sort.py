# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""DICOM sort tab for Gradio GUI."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from pydantic import ValidationError

from habit.core.dicom_sort import DicomSortConfig, run_dicom_sort
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    abs_path,
    extract_validation_msgs,
    load_config_yaml,
    open_directory,
    parse_comma_list,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
)
from habit.utils.log_utils import setup_logger, stop_queue_listener

FORMAT_TEMPLATES: Dict[str, str] = {
    "Default clinical (subj_%n_%g_%x/%s_%d/%r_%o.dcm)": "subj_%n_%g_%x/%s_%d/%r_%o.dcm",
    "Series description (%d_%s)": "%d_%s",
    "Patient name + series (%n_%s)": "%n_%s",
    "Description + protocol + series (%d_%p_%s)": "%d_%p_%s",
    "Custom": "",
}


def render_dicom_sort_tab() -> None:
    """Render DICOM sort/rename tab."""
    gr.Markdown("Sort and rename raw DICOM folders using dcm2niix.")

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing DICOM sort YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Paths")
        with gr.Row():
            data_dir = gr.Textbox(label="Input DICOM folder *", scale=4)
            data_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="Output folder *", scale=4)
            out_btn = gr.Button("Browse", scale=1)

    with gr.Group():
        gr.Markdown("### 2. Naming format (-f)")
        template = gr.Dropdown(label="Template", choices=list(FORMAT_TEMPLATES.keys()), value=list(FORMAT_TEMPLATES.keys())[0])
        f_pattern = gr.Textbox(label="dcm2niix -f pattern *", value=FORMAT_TEMPLATES[list(FORMAT_TEMPLATES.keys())[0]])

    with gr.Accordion("Advanced (optional)", open=False):
        dcm2niix_path = gr.Textbox(label="dcm2niix_path", value="")
        output_dir = gr.Textbox(label="output_dir (overrides out_dir for dcm2niix -o)", value="")
        extra_args = gr.Textbox(label="extra_args (comma-separated)", value="")

    submit_btn = gr.Button("Validate and run DICOM sort", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=15, interactive=False)
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def on_template_change(name: str) -> str:
        val = FORMAT_TEMPLATES.get(name, "")
        return val if val else "subj_%n_%g_%x/%s_%d/%r_%o.dcm"

    template.change(on_template_change, inputs=template, outputs=f_pattern)

    def browse_file() -> Any:
        p = select_local_path("file", "Select YAML")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    data_btn.click(browse_folder, outputs=data_dir)
    out_btn.click(browse_folder, outputs=out_dir)

    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            return [noop] * 8
        loaded = load_config_yaml(path)
        if not loaded:
            return [noop] * 8
        f_val = loaded.get("f") or loaded.get("filename_format") or ""
        extra = loaded.get("extra_args", [])
        return [
            loaded.get("data_dir", ""),
            loaded.get("out_dir", ""),
            "Custom" if f_val not in FORMAT_TEMPLATES.values() else next(
                (k for k, v in FORMAT_TEMPLATES.items() if v == f_val), "Custom"
            ),
            f_val,
            loaded.get("dcm2niix_path", "") or "",
            loaded.get("output_dir", "") or "",
            ", ".join(extra) if isinstance(extra, list) else "",
        ]

    existing_yaml.change(
        load_yaml,
        inputs=existing_yaml,
        outputs=[data_dir, out_dir, template, f_pattern, dcm2niix_path, output_dir, extra_args],
    )

    def run_sort(
        data_dir_val: str,
        out_dir_val: str,
        f_val: str,
        dcm2niix_path_val: str,
        output_dir_val: str,
        extra_args_val: str,
    ):
        if not data_dir_val or not out_dir_val or not f_val:
            yield "❌ data_dir, out_dir, and -f pattern are required.", gr.update(visible=False)
            return

        # Resolve any relative paths to absolute so the saved YAML is portable
        # and the pipeline's own path-resolution does not mis-map them.
        data_dir_abs = abs_path(data_dir_val)
        out_dir_abs = abs_path(out_dir_val)

        config_data: Dict[str, Any] = {
            "data_dir": data_dir_abs,
            "out_dir": out_dir_abs,
            "f": f_val,
            "extra_args": parse_comma_list(extra_args_val),
        }
        if dcm2niix_path_val.strip():
            config_data["dcm2niix_path"] = abs_path(dcm2niix_path_val.strip())
        if output_dir_val.strip():
            config_data["output_dir"] = abs_path(output_dir_val.strip())

        try:
            config = DicomSortConfig(**config_data)
            os.makedirs(out_dir_abs, exist_ok=True)
            gui_path = str(Path(out_dir_abs) / "config_dicom_sort_gui.yaml")
            save_config_yaml(config_data, gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running DICOM sort...", gr.update(visible=False)

            def job() -> None:
                logger = setup_logger(
                    name="cli.dicom_sort",
                    output_dir=Path(out_dir_abs),
                    log_filename="processing.log",
                    level=logging.INFO,
                )
                try:
                    run_dicom_sort(config, logger=logger)
                finally:
                    stop_queue_listener()

            for log_text in run_background_job(
                job,
                log_file=Path(out_dir_abs) / "processing.log",
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
        run_sort,
        inputs=[data_dir, out_dir, f_pattern, dcm2niix_path, output_dir, extra_args],
        outputs=[log_output, open_dir_btn],
    )
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)
