# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Preprocessing pipeline step helpers for the Gradio GUI.

Constants, order management, and config dict assembly for ``PreprocessingConfig``.
Step-specific parameter widgets remain in ``tab_preprocess.py`` until each step
has a dedicated Pydantic sub-schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr

from habit.gui.utils import abs_path, parse_comma_list, yaml_block_to_dict

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

DEFAULT_HIST_PERCENTILES: str = "1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99"

BUILTIN_FORM_STEPS: frozenset[str] = frozenset(KNOWN_STEPS)

PREPROCESS_QUICK_PRESETS: Dict[str, List[str]] = {
    "Standard (resample + registration + z-score)": [
        "resample", "registration", "zscore_normalization",
    ],
    "Minimal (resample only)": ["resample"],
    "With N4 bias correction": [
        "n4_correction", "resample", "registration", "zscore_normalization",
    ],
    "DICOM to NIfTI only": ["dcm2nii"],
}


def dcm2_single_file_from_yaml(value: Any) -> str:
    """Map YAML single_file_mode to GUI dropdown value."""
    if value is None:
        return "auto"
    return "true" if bool(value) else "false"


def dcm2_single_file_to_yaml(value: str) -> Optional[bool]:
    """Map GUI dropdown value to YAML single_file_mode."""
    if value == "auto":
        return None
    return value == "true"


def parse_step_order(order: Union[str, List[str]]) -> List[str]:
    """
    Validate preprocessing step execution order.

    Args:
        order: List of step keys or legacy comma-separated string.

    Returns:
        List[str]: Validated ordered step keys.

    Raises:
        ValueError: When order is empty or contains unknown steps.
    """
    if isinstance(order, list):
        steps: List[str] = [str(s).strip() for s in order if s and str(s).strip()]
    else:
        steps = parse_comma_list(str(order))
    if not steps:
        raise ValueError("Step execution order cannot be empty.")
    unknown: List[str] = [s for s in steps if s not in KNOWN_STEPS]
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {', '.join(unknown)}")
    return steps


def enabled_from_check_values(check_values: List[bool]) -> List[str]:
    """Map parallel checkbox values to enabled step keys (``KNOWN_STEPS`` order)."""
    return [KNOWN_STEPS[idx] for idx, checked in enumerate(check_values) if checked]


def order_number_updates(order: List[str]) -> Tuple[Any, ...]:
    """Build gr.update payloads for per-step order number widgets."""
    return tuple(
        gr.update(value=(order.index(step_key) + 1) if step_key in order else None)
        for step_key in KNOWN_STEPS
    )


def apply_order_from_number(
    step_key: str,
    new_num: Optional[Union[int, float]],
    current_order: List[str],
    enabled: List[str],
) -> List[str]:
    """Reorder one enabled step by editing its sequence number."""
    order = [step for step in (current_order or []) if step in enabled]
    if step_key not in enabled:
        return order
    if new_num is None or (isinstance(new_num, str) and not str(new_num).strip()):
        return sync_order_with_enabled(enabled, order)
    if step_key in order:
        order.remove(step_key)
    target = max(1, min(int(new_num), len(enabled)))
    order.insert(target - 1, step_key)
    return order


def move_step_in_order(order: List[str], step_key: str, direction: int) -> List[str]:
    """Move a step up (-1) or down (+1) within the current order list."""
    if step_key not in order:
        return list(order)
    order = list(order)
    idx = order.index(step_key)
    new_idx = idx + direction
    if 0 <= new_idx < len(order):
        order[idx], order[new_idx] = order[new_idx], order[idx]
    return order


def sync_order_with_enabled(enabled: List[str], current_order: List[str]) -> List[str]:
    """Reconcile execution order when enabled steps change."""
    enabled_set = set(enabled or [])
    preserved = [step for step in (current_order or []) if step in enabled_set]
    for step in KNOWN_STEPS:
        if step in enabled_set and step not in preserved:
            preserved.append(step)
    return preserved


def panel_visible(step_key: str, enabled: List[str]) -> Dict[str, Any]:
    """Return gr.update toggling a parameter panel based on enabled steps."""
    return gr.update(visible=step_key in set(enabled or []))


def modalities_display_text(modalities: List[str]) -> str:
    """Format modality keys for read-only GUI text fields."""
    return ", ".join(modalities)


def resolve_modalities_for_run(
    data_dir: str,
    auto_select: bool,
    modalities_text: str,
) -> Tuple[List[str], Optional[str]]:
    """
    Resolve modality keys at run time from display text or rescan ``data_dir``.

    Returns:
        Tuple[List[str], Optional[str]]: Modality keys and error message when empty.
    """
    from habit.gui.utils import discover_modalities_from_data_dir

    parsed = parse_comma_list(modalities_text)
    if parsed:
        return parsed, None
    modalities, status = discover_modalities_from_data_dir(data_dir, auto_select)
    if modalities:
        return modalities, None
    return [], status or "No modalities detected. Check data_dir layout."


def toggle_backend_panels(backend: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Show ANTs/SimpleITK vs elastix-specific registration fields."""
    is_elastix: bool = backend == "elastix"
    return (
        gr.update(visible=not is_elastix),
        gr.update(visible=is_elastix),
        gr.update(visible=backend == "simpleitk"),
    )


@dataclass
class StepParamValues:
    """Widget values for all built-in preprocessing step parameter panels."""

    n4_levels: Any
    res_spacing: Any
    zs_mask: Any
    zs_mask_key: Any
    reg_fixed: Any
    reg_backend: Any
    reg_transform: Any
    reg_metric: Any
    reg_optimizer: Any
    elastix_files: Any
    elastix_path: Any
    transformix_path: Any
    elastix_threads: Any
    elastix_overrides: Any
    sitk_bins: Any
    sitk_sampling: Any
    sitk_shrink: Any
    sitk_sigmas: Any
    sitk_lr: Any
    sitk_iters: Any
    sitk_bspline_mesh: Any
    sitk_bspline_order: Any
    reg_mask: Any
    reg_replace_mask: Any
    reg_mask_key: Any
    hist_percentiles: Any
    hist_target_min: Any
    hist_target_max: Any
    hist_mask_key: Any
    ahe_alpha: Any
    ahe_beta: Any
    ahe_radius: Any
    reorient_target: Any
    reorient_mode: Any
    dcm2_path: Any
    dcm2_format: Any
    dcm2_compress: Any
    dcm2_anonymize: Any
    dcm2_ignore_derived: Any
    dcm2_crop: Any
    dcm2_json: Any
    dcm2_verbose: Any
    dcm2_batch: Any
    dcm2_adjacent: Any
    dcm2_merge: Any
    dcm2_single_file: Any
    extra_prep_yaml: Any


def build_preprocessing_steps(
    enabled_set: set[str],
    step_order: List[str],
    modalities: List[str],
    modality_err: Optional[str],
    dcm2_modalities: List[str],
    dcm2_err: Optional[str],
    params: StepParamValues,
) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    """
    Assemble the ``Preprocessing`` block for ``PreprocessingConfig``.

    Args:
        enabled_set: Set of enabled step keys.
        step_order: Full execution order (enabled steps only).
        modalities: Resolved modality keys for most steps.
        modality_err: Error when modalities are empty.
        dcm2_modalities: Modality keys for dcm2nii.
        dcm2_err: Error when dcm2 modalities are empty.
        params: Step parameter widget values.

    Returns:
        Tuple[Optional[str], Dict]: Error message (if any) and step config dict.
    """
    built_steps: Dict[str, Dict[str, Any]] = {}

    if "n4_correction" in enabled_set:
        if not modalities:
            return f"❌ n4_correction: {modality_err}", {}
        built_steps["n4_correction"] = {
            "images": modalities,
            "num_fitting_levels": int(params.n4_levels),
        }

    if "resample" in enabled_set:
        if not modalities:
            return f"❌ resample: {modality_err}", {}
        try:
            spacing = [float(s) for s in parse_comma_list(str(params.res_spacing))]
            if len(spacing) != 3:
                raise ValueError
        except ValueError:
            return "❌ target_spacing must be three comma-separated numbers.", {}
        built_steps["resample"] = {"images": modalities, "target_spacing": spacing}

    if "zscore_normalization" in enabled_set:
        if not modalities:
            return f"❌ zscore_normalization: {modality_err}", {}
        zs_cfg: Dict[str, Any] = {
            "images": modalities,
            "only_inmask": params.zs_mask,
        }
        if params.zs_mask_key and str(params.zs_mask_key).strip():
            zs_cfg["mask_key"] = str(params.zs_mask_key).strip()
        built_steps["zscore_normalization"] = zs_cfg

    if "registration" in enabled_set:
        if not modalities:
            return f"❌ registration: {modality_err}", {}
        fixed_image = str(params.reg_fixed).strip() or modalities[0]
        if fixed_image not in modalities:
            return (
                f'❌ fixed_image "{fixed_image}" must be one of the detected modalities: '
                f"{', '.join(modalities)}."
            ), {}
        reg_cfg: Dict[str, Any] = {
            "images": modalities,
            "fixed_image": fixed_image,
            "backend": params.reg_backend,
            "use_mask": params.reg_mask,
            "replace_by_fixed_image_mask": params.reg_replace_mask,
        }
        if params.reg_mask_key and str(params.reg_mask_key).strip():
            reg_cfg["mask_key"] = str(params.reg_mask_key).strip()
        if params.reg_backend == "elastix":
            files_raw = parse_comma_list(str(params.elastix_files))
            if not files_raw:
                return "❌ elastix_parameter_files is required when backend=elastix.", {}
            abs_files = [abs_path(f) for f in files_raw]
            reg_cfg["elastix_parameter_files"] = abs_files if len(abs_files) > 1 else abs_files[0]
            if str(params.elastix_path).strip():
                reg_cfg["elastix_path"] = abs_path(str(params.elastix_path).strip())
            if str(params.transformix_path).strip():
                reg_cfg["transformix_path"] = abs_path(str(params.transformix_path).strip())
            if int(params.elastix_threads or 0) > 0:
                reg_cfg["elastix_threads"] = int(params.elastix_threads)
            if params.elastix_overrides and str(params.elastix_overrides).strip():
                try:
                    overrides = yaml_block_to_dict(str(params.elastix_overrides))
                    if overrides:
                        reg_cfg["elastix_parameter_overrides"] = overrides
                except ValueError as exc:
                    return f"❌ elastix_parameter_overrides: {exc}", {}
        else:
            reg_cfg["type_of_transform"] = params.reg_transform
            reg_cfg["metric"] = params.reg_metric
            if str(params.reg_optimizer).strip():
                reg_cfg["optimizer"] = str(params.reg_optimizer).strip()
            if params.reg_backend == "simpleitk":
                reg_cfg["number_of_histogram_bins"] = int(params.sitk_bins)
                reg_cfg["metric_sampling_percentage"] = float(params.sitk_sampling)
                reg_cfg["shrink_factors_per_level"] = [
                    int(x) for x in parse_comma_list(str(params.sitk_shrink))
                ]
                reg_cfg["smoothing_sigmas_per_level"] = [
                    float(x) for x in parse_comma_list(str(params.sitk_sigmas))
                ]
                reg_cfg["learning_rate"] = float(params.sitk_lr)
                reg_cfg["number_of_iterations"] = int(params.sitk_iters)
                reg_cfg["bspline_mesh_size"] = int(params.sitk_bspline_mesh)
                reg_cfg["bspline_order"] = int(params.sitk_bspline_order)
        built_steps["registration"] = reg_cfg

    if "histogram_standardization" in enabled_set:
        if not modalities:
            return f"❌ histogram_standardization: {modality_err}", {}
        try:
            percentiles = [float(x) for x in parse_comma_list(str(params.hist_percentiles))]
        except ValueError:
            return "❌ histogram percentiles must be comma-separated numbers.", {}
        hist_cfg: Dict[str, Any] = {
            "images": modalities,
            "percentiles": percentiles,
            "target_min": float(params.hist_target_min),
            "target_max": float(params.hist_target_max),
        }
        if params.hist_mask_key and str(params.hist_mask_key).strip():
            hist_cfg["mask_key"] = str(params.hist_mask_key).strip()
        built_steps["histogram_standardization"] = hist_cfg

    if "adaptive_histogram_equalization" in enabled_set:
        if not modalities:
            return f"❌ adaptive_histogram_equalization: {modality_err}", {}
        built_steps["adaptive_histogram_equalization"] = {
            "images": modalities,
            "alpha": float(params.ahe_alpha),
            "beta": float(params.ahe_beta),
            "radius": int(params.ahe_radius),
        }

    if "reorientation" in enabled_set:
        if not modalities:
            return f"❌ reorientation: {modality_err}", {}
        built_steps["reorientation"] = {
            "images": modalities,
            "target_orientation": str(params.reorient_target).strip(),
            "mode": str(params.reorient_mode).strip(),
        }

    if "dcm2nii" in enabled_set:
        if not dcm2_modalities:
            return f"❌ dcm2nii: {dcm2_err or 'No modality key available.'}", {}
        dcm2_cfg: Dict[str, Any] = {
            "images": dcm2_modalities,
            "compress": params.dcm2_compress,
            "anonymize": params.dcm2_anonymize,
            "ignore_derived": params.dcm2_ignore_derived,
            "crop_images": params.dcm2_crop,
            "generate_json": params.dcm2_json,
            "verbose": params.dcm2_verbose,
            "batch_mode": params.dcm2_batch,
            "adjacent_dicoms": params.dcm2_adjacent,
            "merge_slices": str(params.dcm2_merge),
        }
        if str(params.dcm2_path).strip():
            dcm2_cfg["dcm2niix_path"] = str(params.dcm2_path).strip()
        if str(params.dcm2_format).strip():
            dcm2_cfg["filename_format"] = str(params.dcm2_format).strip()
        single_file_mode = dcm2_single_file_to_yaml(str(params.dcm2_single_file))
        if single_file_mode is not None:
            dcm2_cfg["single_file_mode"] = single_file_mode
        built_steps["dcm2nii"] = dcm2_cfg

    if params.extra_prep_yaml and str(params.extra_prep_yaml).strip():
        try:
            extra_block = yaml_block_to_dict(str(params.extra_prep_yaml))
            if extra_block:
                built_steps.update(extra_block)
        except ValueError as exc:
            return f"❌ {exc}", {}

    prep_steps_config: Dict[str, Dict[str, Any]] = {}
    for step_key in step_order:
        if step_key in built_steps:
            prep_steps_config[step_key] = built_steps[step_key]

    return None, prep_steps_config
