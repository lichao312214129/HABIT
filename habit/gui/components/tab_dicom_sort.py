# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""DICOM sort tab for Gradio GUI (schema-driven)."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
from pydantic import ValidationError

from habit.core.dicom_sort import DicomSortConfig, run_dicom_sort
from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.path_picker import PathPickerRegistry
from habit.gui.pipeline_runner import run_background_job
from habit.gui.schema_form import SchemaForm
from habit.gui.utils import (
    abs_path,
    extract_validation_msgs,
    load_config_yaml,
    load_gui_draft,
    open_directory,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    translate_pydantic_error,
    user_visible_path,
)
from habit.gui.project.step_hooks import finalize_step_from_log, mark_step_running
from habit.gui.step_integration import register_project_path_fill
from habit.gui.step_registry import register_step_paths
from habit.utils.log_utils import setup_logger, stop_queue_listener

FORMAT_TEMPLATES: Dict[str, str] = {
    "Default clinical (subj_%n_%g_%x/%s_%d/%r_%o.dcm)": "subj_%n_%g_%x/%s_%d/%r_%o.dcm",
    "Series description (%d_%s)": "%d_%s",
    "Patient name + series (%n_%s)": "%n_%s",
    "Description + protocol + series (%d_%p_%s)": "%d_%p_%s",
    "Custom": "",
}


def render_dicom_sort_tab(
    demo=None,
    path_picker: PathPickerRegistry | None = None,
    project_root_state: Any | None = None,
) -> None:
    """Render DICOM sort/rename tab (schema-driven, zero hardcoded widgets)."""
    picker = path_picker if path_picker is not None else PathPickerRegistry()
    gr.Markdown(
        "**Step 1 — DICOM Sort:** Organize raw DICOM into a structured folder layout. "
        "Next step (Preprocessing) can convert sorted DICOM to NIfTI."
    )

    # --- YAML load ---
    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing DICOM sort YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)
    picker.add(browse_cfg_btn, existing_yaml, pick="file")

    # --- Schema-driven form (auto-generates all widgets from DicomSortConfig) ---
    form = SchemaForm.build(
        DicomSortConfig,
        exclude={"filename_format"},
        group_order=["Paths", "Naming", "Advanced"],
        open_groups={"Paths", "Naming"},
        path_picker=picker,
    )

    # --- Convenience: format template dropdown (fills 'f' field) ---
    f_widget = form.widget("f")
    template = gr.Dropdown(
        label="Quick format template",
        choices=list(FORMAT_TEMPLATES.keys()),
        value=list(FORMAT_TEMPLATES.keys())[0],
        info="Select a template to auto-fill the -f pattern above",
    )

    def on_template_change(name: str) -> str:
        val = FORMAT_TEMPLATES.get(name, "")
        return val if val else "subj_%n_%g_%x/%s_%d/%r_%o.dcm"

    template.change(on_template_change, inputs=template, outputs=f_widget)

    # --- Submit / Stop ---
    with gr.Row():
        submit_btn = gr.Button("Validate and run DICOM sort", variant="primary")
        stop_btn = gr.Button("Stop", interactive=False)
    log_output = render_console_log(lines=15, elem_id="habit-log-dicom")
    open_dir_btn = gr.Button("Open output folder", visible=False)

    # --- YAML load handler (schema-driven populate) ---
    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        resolved = abs_path(path) if path and str(path).strip() else ""
        if not resolved or not os.path.exists(resolved):
            return [noop] * len(form.outputs())
        loaded = load_config_yaml(resolved)
        if not loaded:
            return [noop] * len(form.outputs())
        # Backward compat: filename_format → f
        if "f" not in loaded and "filename_format" in loaded:
            loaded["f"] = loaded["filename_format"]
        # Convert paths to user-visible format
        for key in ("data_dir", "out_dir", "dcm2niix_path", "output_dir"):
            if key in loaded and loaded[key]:
                loaded[key] = user_visible_path(str(loaded[key]))
        return form.populate(loaded)

    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=form.outputs())

    # --- Run handler (schema-driven collect) ---
    def run_sort(project_root: str, *values: Any):
        config_data = form.collect(*values)
        data_dir_val = config_data.get("data_dir", "")
        out_dir_val = config_data.get("out_dir", "")
        f_val = config_data.get("f", "")

        if not data_dir_val or not out_dir_val or not f_val:
            yield "❌ data_dir, out_dir, and -f pattern are required.", gr.update(visible=False)
            return

        if project_root:
            mark_step_running(project_root, "dicom_sort")

        data_dir_rt = abs_path(data_dir_val)
        out_dir_rt = abs_path(out_dir_val)

        # Runtime config (absolute paths)
        runtime_data: Dict[str, Any] = dict(config_data)
        runtime_data["data_dir"] = data_dir_rt
        runtime_data["out_dir"] = out_dir_rt
        for key in ("dcm2niix_path", "output_dir"):
            val = runtime_data.get(key)
            if val and str(val).strip():
                runtime_data[key] = abs_path(str(val).strip())
            else:
                runtime_data.pop(key, None)

        # YAML data (user-visible paths)
        yaml_data: Dict[str, Any] = dict(config_data)
        yaml_data["data_dir"] = user_visible_path(data_dir_val)
        yaml_data["out_dir"] = user_visible_path(out_dir_val)
        for key in ("dcm2niix_path", "output_dir"):
            val = yaml_data.get(key)
            if val and str(val).strip():
                yaml_data[key] = user_visible_path(str(val).strip())
            else:
                yaml_data.pop(key, None)

        try:
            config = DicomSortConfig(**runtime_data)
            os.makedirs(out_dir_rt, exist_ok=True)
            gui_path = str(Path(out_dir_rt) / "config_dicom_sort_gui.yaml")
            save_config_yaml(yaml_data, gui_path)
            save_gui_draft("dicom_sort", gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running DICOM sort...", gr.update(visible=False)

            def job() -> None:
                logger = setup_logger(
                    name="cli.dicom_sort",
                    output_dir=Path(out_dir_rt),
                    log_filename="processing.log",
                    level=logging.INFO,
                )
                try:
                    run_dicom_sort(config, logger=logger)
                finally:
                    stop_queue_listener()

            last_log = ""
            for log_text in run_background_job(
                job,
                log_file=Path(out_dir_rt) / "processing.log",
            ):
                last_log = log_text
                yield log_text, gr.update(visible=True)
            finalize_step_from_log(
                project_root,
                "dicom_sort",
                last_log,
                config_path=gui_path,
                output_dir=out_dir_rt,
            )
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
        job_start_button_updates,
        outputs=[submit_btn, stop_btn],
    ).then(
        run_sort,
        inputs=[project_root_state] + form.inputs(),
        outputs=[log_output, open_dir_btn],
    ).then(
        job_end_button_updates,
        outputs=[submit_btn, stop_btn],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])
    open_dir_btn.click(lambda p: open_directory(abs_path(p)), inputs=form.widget("out_dir"))

    # --- Step integration ---
    if project_root_state is not None:
        register_step_paths(
            "dicom_sort", ["data_dir", "out_dir"],
            [form.widget("data_dir"), form.widget("out_dir")],
        )
        register_project_path_fill(
            project_root_state, "dicom_sort", ["data_dir", "out_dir"],
            [form.widget("data_dir"), form.widget("out_dir")],
        )

    if path_picker is None:
        picker.finalize()

    if demo is not None:
        def _restore_dicom_path() -> str:
            draft = load_gui_draft("dicom_sort") or ""
            return user_visible_path(draft) if draft else ""

        demo.load(_restore_dicom_path, inputs=[], outputs=[existing_yaml])
