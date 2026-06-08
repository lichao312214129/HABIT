# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Preprocessing tab component for Gradio GUI.
Provides clinician-friendly inputs for image preprocessing pipelines.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_preprocess import run_preprocess
from habit.core.preprocessing.config_schemas import PreprocessingConfig
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    dict_to_yaml_block,
    load_config_yaml,
    open_directory,
    parse_comma_list,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
    yaml_block_to_dict,
)

KNOWN_STEPS: List[str] = [
    "n4_correction",
    "resample",
    "zscore_normalization",
    "registration",
    "histogram_standardization",
    "adaptive_histogram_equalization",
    "reorientation",
    "dcm2nii",
]
STEP_LABELS: Dict[str, str] = {
    "n4_correction": "N4 bias correction",
    "resample": "Resample",
    "zscore_normalization": "Z-score normalization",
    "registration": "Registration",
    "histogram_standardization": "Histogram standardization",
    "adaptive_histogram_equalization": "Adaptive histogram equalization",
    "reorientation": "Reorientation",
    "dcm2nii": "DICOM to NIfTI (dcm2nii)",
}
DEFAULT_STEP_ORDER: str = "n4_correction, resample, zscore_normalization, registration"


def _parse_step_order(order_text: str) -> List[str]:
    """Parse and validate the user-defined preprocessing step execution order."""
    steps: List[str] = parse_comma_list(order_text)
    if not steps:
        raise ValueError("Step execution order cannot be empty.")
    unknown: List[str] = [s for s in steps if s not in KNOWN_STEPS]
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {', '.join(unknown)}")
    return steps


def _toggle_backend_panels(backend: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Show ANTs/SimpleITK vs elastix-specific registration fields."""
    is_elastix: bool = backend == "elastix"
    return (
        gr.update(visible=not is_elastix),
        gr.update(visible=is_elastix),
        gr.update(visible=backend == "simpleitk"),
    )


def render_preprocess_tab() -> None:
    """Render the image preprocessing tab inside a parent Gradio Blocks context."""
    gr.Markdown(
        "Configure a custom preprocessing pipeline (registration, resampling, intensity "
        "normalization, etc.) and batch-process all subjects."
    )

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing preprocessing YAML (optional)", scale=4)
        browse_prep_config_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Paths and system settings")
        with gr.Row():
            data_dir = gr.Textbox(label="Input data root directory *", scale=4)
            prep_data_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="Output directory *", scale=4)
            prep_out_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            processes = gr.Number(label="Parallel workers (CPU)", value=1, minimum=1, maximum=32, step=1)
            auto_select = gr.Checkbox(label="Auto-select first file in modality folder", value=True)
            random_state = gr.Number(label="Random state", value=42, precision=0)

    with gr.Group():
        gr.Markdown("### 2. Pipeline step order")
        step_order_text = gr.Textbox(
            label="Execution order (comma-separated; top-to-bottom = run order)",
            value=DEFAULT_STEP_ORDER,
        )
        gr.Markdown(
            f"Supported steps: `{', '.join(KNOWN_STEPS)}`. "
            "Steps without a dedicated form below can be supplied via **Extra preprocessing YAML**."
        )

    with gr.Group():
        gr.Markdown("### 3. Built-in step parameters")
        en_n4 = gr.Checkbox(label=f"Enable {STEP_LABELS['n4_correction']}", value=False)
        with gr.Column(visible=False) as box_n4:
            n4_images = gr.Textbox(label="Modalities (comma-separated)", value="T1, T2, FLAIR")
            n4_levels = gr.Number(label="num_fitting_levels", value=4, minimum=1, maximum=6, step=1)

        en_res = gr.Checkbox(label=f"Enable {STEP_LABELS['resample']}", value=False)
        with gr.Column(visible=False) as box_res:
            res_images = gr.Textbox(label="Modalities (comma-separated)", value="T1, T2, DWI, ADC")
            res_spacing = gr.Textbox(label="target_spacing (x, y, z mm)", value="1.0, 1.0, 1.0")

        en_zs = gr.Checkbox(label=f"Enable {STEP_LABELS['zscore_normalization']}", value=False)
        with gr.Column(visible=False) as box_zs:
            zs_images = gr.Textbox(label="Modalities (comma-separated)", value="T1, T2")
            zs_mask = gr.Checkbox(label="only_inmask", value=False)
            zs_mask_key = gr.Textbox(label="mask_key (optional)", value="")

        en_reg = gr.Checkbox(label=f"Enable {STEP_LABELS['registration']}", value=False)
        with gr.Column(visible=False) as box_reg:
            reg_images = gr.Textbox(label="images (comma-separated; must include fixed_image)", value="T2WI, ADC")
            reg_fixed = gr.Textbox(label="fixed_image *", value="T2WI")
            reg_backend = gr.Dropdown(
                label="backend",
                choices=["ants", "simpleitk", "elastix"],
                value="ants",
            )
            with gr.Column(visible=True) as box_ants_sitk:
                with gr.Row():
                    reg_transform = gr.Dropdown(
                        label="type_of_transform",
                        choices=["SyNRA", "SyN", "Rigid", "Affine", "Translation"],
                        value="SyNRA",
                    )
                    reg_metric = gr.Dropdown(
                        label="metric",
                        choices=["MI", "CC", "MeanSquares"],
                        value="MI",
                    )
                reg_optimizer = gr.Textbox(label="optimizer (optional)", value="")
            with gr.Column(visible=False) as box_elastix:
                elastix_parameter_files = gr.Textbox(
                    label="elastix_parameter_files (path or comma-separated list) *",
                    value="",
                )
                with gr.Row():
                    elastix_path = gr.Textbox(label="elastix_path (optional)", value="")
                    transformix_path = gr.Textbox(label="transformix_path (optional)", value="")
                elastix_threads = gr.Number(label="elastix_threads (optional)", value=0, precision=0)
            with gr.Accordion("SimpleITK-only tuning (backend=simpleitk)", open=False, visible=False) as box_sitk:
                sitk_bins = gr.Number(label="number_of_histogram_bins", value=50, precision=0)
                sitk_sampling = gr.Number(label="metric_sampling_percentage", value=0.01)
                sitk_shrink = gr.Textbox(label="shrink_factors_per_level", value="4, 2, 1")
                sitk_sigmas = gr.Textbox(label="smoothing_sigmas_per_level", value="2.1, 1.0, 0.0")
                sitk_lr = gr.Number(label="learning_rate", value=1.0)
                sitk_iters = gr.Number(label="number_of_iterations", value=100, precision=0)
                sitk_bspline_mesh = gr.Number(label="bspline_mesh_size", value=8, precision=0)
                sitk_bspline_order = gr.Number(label="bspline_order", value=3, precision=0)
            reg_mask = gr.Checkbox(label="use_mask", value=False)
            reg_replace_mask = gr.Checkbox(label="replace_by_fixed_image_mask", value=True)

    with gr.Accordion("Extra preprocessing YAML (optional advanced steps / overrides)", open=False):
        extra_prep_yaml = gr.Textbox(
            label="Additional Preprocessing blocks (YAML mapping)",
            lines=8,
            placeholder=(
                "histogram_standardization:\n"
                "  images: [T1, T2]\n"
                "  percentiles: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]\n"
            ),
        )

    with gr.Group():
        gr.Markdown("### 4. Intermediate outputs")
        save_intermediate = gr.Checkbox(label="save_intermediate", value=False)
        intermediate_steps = gr.CheckboxGroup(
            label="intermediate_steps (empty = save all enabled steps)",
            choices=KNOWN_STEPS,
            value=[],
        )

    submit_btn = gr.Button("Validate and run preprocessing", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=18, interactive=False)
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def toggle_vis(checked: bool) -> Dict[str, Any]:
        return gr.update(visible=checked)

    for checkbox, panel in [
        (en_n4, box_n4),
        (en_res, box_res),
        (en_zs, box_zs),
        (en_reg, box_reg),
    ]:
        checkbox.change(toggle_vis, inputs=checkbox, outputs=panel)

    reg_backend.change(
        _toggle_backend_panels,
        inputs=reg_backend,
        outputs=[box_ants_sitk, box_elastix, box_sitk],
    )

    def browse_file() -> Any:
        path: Optional[str] = select_local_path("file", "Select preprocessing YAML")
        return path if path else gr.update()

    def browse_folder() -> Any:
        path: Optional[str] = select_local_path("folder", "Select directory")
        return path if path else gr.update()

    browse_prep_config_btn.click(browse_file, outputs=existing_yaml)
    prep_data_btn.click(browse_folder, outputs=data_dir)
    prep_out_btn.click(browse_folder, outputs=out_dir)

    def load_config(yaml_path: str) -> List[Any]:
        """Hydrate form widgets from an existing preprocessing YAML file."""
        noop = gr.update()
        if not yaml_path or not os.path.exists(yaml_path):
            return [noop] * 40

        loaded: Optional[Dict[str, Any]] = load_config_yaml(yaml_path)
        if not loaded:
            return [noop] * 40

        prep: Dict[str, Any] = loaded.get("Preprocessing", {}) or {}
        order_keys: List[str] = [k for k in prep.keys() if k in KNOWN_STEPS]
        for step in KNOWN_STEPS:
            if step not in order_keys:
                order_keys.append(step)

        n4 = prep.get("n4_correction", {})
        res = prep.get("resample", {})
        zs = prep.get("zscore_normalization", {})
        reg = prep.get("registration", {})
        save_opt: Dict[str, Any] = loaded.get("save_options", {}) or {}

        extra_blocks: Dict[str, Any] = {
            k: v for k, v in prep.items()
            if k not in {"n4_correction", "resample", "zscore_normalization", "registration"}
        }

        sitk_shrink_val = reg.get("shrink_factors_per_level", [4, 2, 1])
        sitk_sigmas_val = reg.get("smoothing_sigmas_per_level", [2.1, 1.0, 0.0])
        elastix_files = reg.get("elastix_parameter_files", "")
        if isinstance(elastix_files, list):
            elastix_files_str = ", ".join(str(x) for x in elastix_files)
        else:
            elastix_files_str = str(elastix_files or "")

        return [
            loaded.get("data_dir", ""),
            loaded.get("out_dir", ""),
            int(loaded.get("processes", 1)),
            loaded.get("auto_select_first_file", True),
            int(loaded.get("random_state", 42)),
            ", ".join(order_keys),
            "n4_correction" in prep,
            ", ".join(n4.get("images", ["T1", "T2", "FLAIR"])),
            int(n4.get("num_fitting_levels", 4)),
            "resample" in prep,
            ", ".join(res.get("images", ["T1", "T2", "DWI", "ADC"])),
            ", ".join(str(s) for s in res.get("target_spacing", [1.0, 1.0, 1.0])),
            "zscore_normalization" in prep,
            ", ".join(zs.get("images", ["T1", "T2"])),
            zs.get("only_inmask", False),
            zs.get("mask_key", "") or "",
            "registration" in prep,
            ", ".join(reg.get("images", ["T2WI", "ADC"])),
            reg.get("fixed_image", "T2WI"),
            reg.get("backend", "ants"),
            reg.get("type_of_transform", "SyNRA"),
            reg.get("metric", "MI"),
            reg.get("optimizer", "") or "",
            elastix_files_str,
            reg.get("elastix_path", "") or "",
            reg.get("transformix_path", "") or "",
            int(reg.get("elastix_threads", 0) or 0),
            int(reg.get("number_of_histogram_bins", 50)),
            float(reg.get("metric_sampling_percentage", 0.01)),
            ", ".join(str(x) for x in sitk_shrink_val),
            ", ".join(str(x) for x in sitk_sigmas_val),
            float(reg.get("learning_rate", 1.0)),
            int(reg.get("number_of_iterations", 100)),
            int(reg.get("bspline_mesh_size", 8)),
            int(reg.get("bspline_order", 3)),
            reg.get("use_mask", False),
            reg.get("replace_by_fixed_image_mask", True),
            dict_to_yaml_block(extra_blocks),
            save_opt.get("save_intermediate", False),
            save_opt.get("intermediate_steps", []) or [],
        ]

    load_outputs = [
        data_dir, out_dir, processes, auto_select, random_state, step_order_text,
        en_n4, n4_images, n4_levels,
        en_res, res_images, res_spacing,
        en_zs, zs_images, zs_mask, zs_mask_key,
        en_reg, reg_images, reg_fixed, reg_backend, reg_transform, reg_metric, reg_optimizer,
        elastix_parameter_files, elastix_path, transformix_path, elastix_threads,
        sitk_bins, sitk_sampling, sitk_shrink, sitk_sigmas, sitk_lr, sitk_iters,
        sitk_bspline_mesh, sitk_bspline_order,
        reg_mask, reg_replace_mask,
        extra_prep_yaml, save_intermediate, intermediate_steps,
    ]
    existing_yaml.change(load_config, inputs=existing_yaml, outputs=load_outputs)

    def run_pipeline(*args: Any):
        """Validate config, save YAML, and run preprocessing with live logs."""
        (
            data_dir_val, out_dir_val, processes_val, auto_select_val, random_state_val,
            step_order_val,
            en_n4_val, n4_images_val, n4_levels_val,
            en_res_val, res_images_val, res_spacing_val,
            en_zs_val, zs_images_val, zs_mask_val, zs_mask_key_val,
            en_reg_val, reg_images_val, reg_fixed_val, reg_backend_val,
            reg_transform_val, reg_metric_val, reg_optimizer_val,
            elastix_files_val, elastix_path_val, transformix_path_val, elastix_threads_val,
            sitk_bins_val, sitk_sampling_val, sitk_shrink_val, sitk_sigmas_val,
            sitk_lr_val, sitk_iters_val, sitk_bspline_mesh_val, sitk_bspline_order_val,
            reg_mask_val, reg_replace_mask_val,
            extra_prep_yaml_val, save_intermediate_val, intermediate_steps_val,
        ) = args

        if not data_dir_val or not out_dir_val:
            yield "❌ data_dir and out_dir are required.", gr.update(visible=False)
            return

        try:
            step_order: List[str] = _parse_step_order(step_order_val)
        except ValueError as exc:
            yield f"❌ {exc}", gr.update(visible=False)
            return

        built_steps: Dict[str, Dict[str, Any]] = {}

        if en_n4_val:
            built_steps["n4_correction"] = {
                "images": parse_comma_list(n4_images_val),
                "num_fitting_levels": int(n4_levels_val),
            }
        if en_res_val:
            try:
                spacing = [float(s) for s in parse_comma_list(res_spacing_val)]
                if len(spacing) != 3:
                    raise ValueError
            except ValueError:
                yield "❌ target_spacing must be three comma-separated numbers.", gr.update(visible=False)
                return
            built_steps["resample"] = {
                "images": parse_comma_list(res_images_val),
                "target_spacing": spacing,
            }
        if en_zs_val:
            zs_cfg: Dict[str, Any] = {
                "images": parse_comma_list(zs_images_val),
                "only_inmask": zs_mask_val,
            }
            if zs_mask_key_val and str(zs_mask_key_val).strip():
                zs_cfg["mask_key"] = str(zs_mask_key_val).strip()
            built_steps["zscore_normalization"] = zs_cfg

        if en_reg_val:
            if not reg_fixed_val:
                yield "❌ fixed_image is required when registration is enabled.", gr.update(visible=False)
                return
            reg_cfg: Dict[str, Any] = {
                "images": parse_comma_list(reg_images_val),
                "fixed_image": str(reg_fixed_val).strip(),
                "backend": reg_backend_val,
                "use_mask": reg_mask_val,
                "replace_by_fixed_image_mask": reg_replace_mask_val,
            }
            if reg_backend_val == "elastix":
                files_raw = parse_comma_list(elastix_files_val)
                if not files_raw:
                    yield "❌ elastix_parameter_files is required when backend=elastix.", gr.update(visible=False)
                    return
                reg_cfg["elastix_parameter_files"] = files_raw if len(files_raw) > 1 else files_raw[0]
                if elastix_path_val.strip():
                    reg_cfg["elastix_path"] = elastix_path_val.strip()
                if transformix_path_val.strip():
                    reg_cfg["transformix_path"] = transformix_path_val.strip()
                if int(elastix_threads_val or 0) > 0:
                    reg_cfg["elastix_threads"] = int(elastix_threads_val)
            else:
                reg_cfg["type_of_transform"] = reg_transform_val
                reg_cfg["metric"] = reg_metric_val
                if reg_optimizer_val.strip():
                    reg_cfg["optimizer"] = reg_optimizer_val.strip()
                if reg_backend_val == "simpleitk":
                    reg_cfg["number_of_histogram_bins"] = int(sitk_bins_val)
                    reg_cfg["metric_sampling_percentage"] = float(sitk_sampling_val)
                    reg_cfg["shrink_factors_per_level"] = [int(x) for x in parse_comma_list(sitk_shrink_val)]
                    reg_cfg["smoothing_sigmas_per_level"] = [
                        float(x) for x in parse_comma_list(sitk_sigmas_val)
                    ]
                    reg_cfg["learning_rate"] = float(sitk_lr_val)
                    reg_cfg["number_of_iterations"] = int(sitk_iters_val)
                    reg_cfg["bspline_mesh_size"] = int(sitk_bspline_mesh_val)
                    reg_cfg["bspline_order"] = int(sitk_bspline_order_val)
            built_steps["registration"] = reg_cfg

        if extra_prep_yaml_val and str(extra_prep_yaml_val).strip():
            try:
                extra_block = yaml_block_to_dict(str(extra_prep_yaml_val))
                if extra_block:
                    built_steps.update(extra_block)
            except ValueError as exc:
                yield f"❌ {exc}", gr.update(visible=False)
                return

        prep_steps_config: Dict[str, Dict[str, Any]] = {}
        for step_key in step_order:
            if step_key in built_steps:
                prep_steps_config[step_key] = built_steps[step_key]

        config_data: Dict[str, Any] = {
            "data_dir": data_dir_val,
            "out_dir": out_dir_val,
            "Preprocessing": prep_steps_config,
            "save_options": {
                "save_intermediate": save_intermediate_val,
                "intermediate_steps": list(intermediate_steps_val or []),
            },
            "processes": int(processes_val),
            "auto_select_first_file": auto_select_val,
            "random_state": int(random_state_val),
            "preprocessing_input_layout": "habit_default",
        }

        try:
            PreprocessingConfig(**config_data)
            os.makedirs(out_dir_val, exist_ok=True)
            gui_config_path = str(Path(out_dir_val) / "config_preprocess_gui.yaml")
            save_config_yaml(config_data, gui_config_path)
            yield (
                f"💾 Config validated and saved to {gui_config_path}\n🚀 Running preprocessing...",
                gr.update(visible=False),
            )
            for log_text in run_background_job(run_preprocess, args=(gui_config_path,)):
                yield log_text, gr.update(visible=True)
        except ValidationError as val_err:
            friendly = translate_pydantic_error(val_err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {e}" for e in friendly), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            yield f"❌ Failed to start: {exc}", gr.update(visible=False)

    submit_btn.click(
        run_pipeline,
        inputs=[
            data_dir, out_dir, processes, auto_select, random_state, step_order_text,
            en_n4, n4_images, n4_levels,
            en_res, res_images, res_spacing,
            en_zs, zs_images, zs_mask, zs_mask_key,
            en_reg, reg_images, reg_fixed, reg_backend, reg_transform, reg_metric, reg_optimizer,
            elastix_parameter_files, elastix_path, transformix_path, elastix_threads,
            sitk_bins, sitk_sampling, sitk_shrink, sitk_sigmas, sitk_lr, sitk_iters,
            sitk_bspline_mesh, sitk_bspline_order,
            reg_mask, reg_replace_mask,
            extra_prep_yaml, save_intermediate, intermediate_steps,
        ],
        outputs=[log_output, open_dir_btn],
    )
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)
