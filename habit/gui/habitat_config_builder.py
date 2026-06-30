# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Build habitat YAML dictionaries from habitat wizard form values."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def _nullable_int(value: Any) -> Optional[int]:
    """
    Convert GUI numeric values to optional integers.

    Args:
        value: Raw widget value.

    Returns:
        Optional[int]: Parsed integer or None when unset / inherit sentinel.
    """
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _parse_connectivity(value: Any) -> int:
    """
    Parse connectivity dropdown values such as ``1 (6-neighbor)``.

    Args:
        value: Raw dropdown value.

    Returns:
        int: Connectivity integer in ``{1, 2, 3}``.
    """
    text = str(value).strip()
    if text and text[0].isdigit():
        return int(text.split()[0])
    try:
        parsed = int(text)
    except ValueError:
        return 1
    return max(1, min(3, parsed))


def _parse_subject_list(raw: str) -> List[str]:
    """
    Parse comma/newline separated subject ids.

    Args:
        raw: Raw text from the GUI.

    Returns:
        List[str]: Non-empty subject identifiers.
    """
    if not raw:
        return []
    tokens = raw.replace("\n", ",").split(",")
    return [token.strip() for token in tokens if token.strip()]


def _parse_torch_gpus(raw: str) -> List[int]:
    """
    Parse comma-separated GPU ids for Torch radiomics.

    Args:
        raw: Raw text such as ``0,1`` or empty string.

    Returns:
        List[int]: GPU index list; empty means inherit/all.
    """
    if not str(raw).strip():
        return []
    result: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _yaml_bool_or_str(value: str) -> Union[bool, str]:
    """
    Preserve ``auto`` while converting GUI dropdown strings to bools.

    Args:
        value: ``auto``, ``true``, or ``false``.

    Returns:
        Union[bool, str]: YAML-ready value.
    """
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return value


def _build_voxel_method(
    feature_type: str,
    modalities: Sequence[str],
    use_custom_dsl: bool,
    custom_method: str,
) -> str:
    """
    Build the ``FeatureConstruction.voxel_level.method`` DSL expression.

    Args:
        feature_type: Selected voxel feature type key.
        modalities: Selected modality folder names.
        use_custom_dsl: When True, return ``custom_method`` unchanged.
        custom_method: Manual DSL from the expert panel.

    Returns:
        str: Feature construction method expression.
    """
    if use_custom_dsl and str(custom_method).strip():
        return str(custom_method).strip()
    mods = [str(m).strip() for m in modalities if str(m).strip()] or ["T2"]
    feature = str(feature_type or "raw")
    if feature == "kinetic":
        raw_parts = ", ".join(f"raw({modality})" for modality in mods)
        return f"kinetic({raw_parts}, timestamps)"
    if feature == "local_entropy":
        parts = [f"local_entropy({modality})" for modality in mods]
    elif feature == "voxel_radiomics":
        parts = [f"voxel_radiomics({modality}, params_file, kernelRadius)" for modality in mods]
    else:
        parts = [f"raw({modality})" for modality in mods]
    if len(parts) == 1:
        return parts[0]
    return f"concat({', '.join(parts)})"


def _build_supervoxel_method(
    sv_agg: str,
    sv_level_method: str,
    sv_rad_mod: str,
    modalities: Sequence[str],
) -> str:
    """
    Build ``FeatureConstruction.supervoxel_level.method``.

    Args:
        sv_agg: Aggregation dropdown value.
        sv_level_method: Manual supervoxel DSL override.
        sv_rad_mod: Modality used for supervoxel radiomics.
        modalities: Selected image modalities.

    Returns:
        str: Supervoxel-level method expression.
    """
    manual = str(sv_level_method or "").strip()
    if manual:
        return manual
    if str(sv_agg) == "supervoxel_radiomics":
        modality = str(sv_rad_mod).strip()
        if not modality:
            modality = next((str(m) for m in modalities if str(m).strip()), "T2")
        return f"concat(supervoxel_radiomics({modality}, params_file))"
    return "mean_voxel_features()"


def build_habitat_config_data(**kwargs: Any) -> Dict[str, Any]:
    """
    Convert habitat wizard widget values into a habitat YAML dictionary.

    Args:
        **kwargs: Flat keyword arguments produced by ``habitat_wizard._collect_config_kwargs``.

    Returns:
        Dict[str, Any]: Habitat analysis configuration mapping.
    """
    run_mode: str = str(kwargs.get("run_mode", "train"))
    data_dir: str = str(kwargs.get("data_dir", "") or "")
    out_dir: str = str(kwargs.get("out_dir", "") or "")
    pipeline_path: str = str(kwargs.get("pipeline_path", "") or "")
    clustering_mode: str = str(kwargs.get("clustering_mode", "two_step"))

    selected_modalities: List[str] = list(kwargs.get("selected_modalities") or [])
    subj_methods: List[Dict[str, Any]] = list(kwargs.get("subj_methods") or [])
    group_methods: List[Dict[str, Any]] = list(kwargs.get("group_methods") or [])

    voxel_method = _build_voxel_method(
        feature_type=str(kwargs.get("voxel_feature_type", "raw")),
        modalities=selected_modalities,
        use_custom_dsl=bool(kwargs.get("voxel_use_custom_dsl")),
        custom_method=str(kwargs.get("voxel_method", "")),
    )
    voxel_params: Dict[str, Any] = {}
    if str(kwargs.get("voxel_feature_type")) == "voxel_radiomics" or "params_file" in voxel_method:
        voxel_params = {
            "params_file": str(kwargs.get("voxel_pf", "")),
            "kernelRadius": int(kwargs.get("voxel_kr", 3) or 3),
            "voxelBatch": int(kwargs.get("voxel_batch", 1000) or 1000),
            "useTorchRadiomics": _yaml_bool_or_str(str(kwargs.get("use_torch", "auto"))),
        }
        gpu_ids = _parse_torch_gpus(str(kwargs.get("torch_gpus", "")))
        if gpu_ids:
            voxel_params["torchGpus"] = gpu_ids
        torch_device = str(kwargs.get("torch_device", "")).strip()
        if torch_device:
            voxel_params["torchDevice"] = torch_device

    feature_construction: Dict[str, Any] = {
        "voxel_level": {
            "method": voxel_method,
            "params": voxel_params,
        },
    }
    if clustering_mode == "two_step":
        sv_params: Dict[str, Any] = {}
        if str(kwargs.get("sv_agg")) == "supervoxel_radiomics":
            sv_params["params_file"] = str(kwargs.get("sv_lpf", ""))
            sv_params["supervoxelUnionBboxCrop"] = bool(kwargs.get("sv_crop", True))
            sv_params["useSupervoxelCext"] = _yaml_bool_or_str(str(kwargs.get("use_sv_cext", "auto")))
            sv_params["supervoxelBatch"] = int(kwargs.get("sv_batch", 64) or 64)
            pad_distance = int(kwargs.get("sv_pad", 0) or 0)
            if pad_distance:
                sv_params["supervoxelPadDistance"] = pad_distance
        feature_construction["supervoxel_level"] = {
            "supervoxel_file_keyword": str(kwargs.get("sv_kw", "*_supervoxel.nrrd")),
            "method": _build_supervoxel_method(
                sv_agg=str(kwargs.get("sv_agg", "mean_voxel_features")),
                sv_level_method=str(kwargs.get("sv_lm", "")),
                sv_rad_mod=str(kwargs.get("sv_rad_mod", "")),
                modalities=selected_modalities,
            ),
            "params": sv_params,
        }
    if subj_methods:
        feature_construction["preprocessing_for_subject_level"] = {"methods": subj_methods}
    if group_methods:
        feature_construction["preprocessing_for_group_level"] = {"methods": group_methods}

    hb_fixed_en = bool(kwargs.get("hb_fixed_en"))
    hb_fixed_n = _nullable_int(kwargs.get("hb_fixed_n"))
    if bool(kwargs.get("simple_mode")) and not hb_fixed_en:
        hb_fixed_en = not bool(kwargs.get("hb_auto", True))
        if hb_fixed_en:
            hb_fixed_n = _nullable_int(kwargs.get("hb_simple_n"))

    habitat_block: Dict[str, Any] = {
        "algorithm": str(kwargs.get("hb_algo", "kmeans")),
        "max_clusters": int(kwargs.get("hb_max", 10) or 10),
        "habitat_cluster_selection_method": [str(kwargs.get("hb_sel", "elbow"))],
        "fixed_n_clusters": hb_fixed_n if hb_fixed_en else None,
        "max_iter": int(kwargs.get("hb_max_iter", 300) or 300),
        "n_init": int(kwargs.get("hb_n_init", 10) or 10),
        "parallel_cluster_search": bool(kwargs.get("hb_par", True)),
        "cluster_search_workers": _nullable_int(kwargs.get("hb_workers")) or None,
    }
    hb_rs = _nullable_int(kwargs.get("hb_rs"))
    if hb_rs is not None:
        habitat_block["random_state"] = hb_rs

    supervoxel_block: Dict[str, Any] = {
        "algorithm": str(kwargs.get("sv_algo", "kmeans")),
        "n_clusters": int(kwargs.get("sv_n_clusters", 50) or 50),
        "max_iter": int(kwargs.get("sv_max_iter", 300) or 300),
        "n_init": int(kwargs.get("sv_n_init", 10) or 10),
        "compactness": float(kwargs.get("sv_compactness", 0.1) or 0.1),
        "sigma": float(kwargs.get("sv_sigma", 0.0) or 0.0),
        "enforce_connectivity": bool(kwargs.get("sv_enforce_conn", True)),
    }
    sv_rs = _nullable_int(kwargs.get("sv_rs"))
    if sv_rs is not None:
        supervoxel_block["random_state"] = sv_rs

    os_fixed_n = _nullable_int(kwargs.get("os_fixed_n"))
    if os_fixed_n == 0:
        os_fixed_n = None
    supervoxel_block["one_step_settings"] = {
        "min_clusters": int(kwargs.get("os_min", 2) or 2),
        "max_clusters": int(kwargs.get("os_max", 10) or 10),
        "fixed_n_clusters": os_fixed_n,
        "selection_method": str(kwargs.get("os_sel", "elbow")),
        "plot_validation_curves": bool(kwargs.get("os_plot", True)),
    }

    segmentation: Dict[str, Any] = {
        "clustering_mode": clustering_mode,
        "supervoxel": supervoxel_block,
        "habitat": habitat_block,
        "postprocess_supervoxel": {
            "enabled": bool(kwargs.get("pp_sv_en")),
            "min_component_size": int(kwargs.get("pp_sv_min", 30) or 30),
            "connectivity": _parse_connectivity(kwargs.get("pp_sv_conn")),
            "reassign_method": "neighbor_vote",
            "max_iterations": int(kwargs.get("pp_sv_maxiter", 3) or 3),
        },
        "postprocess_habitat": {
            "enabled": bool(kwargs.get("pp_hab_en")),
            "min_component_size": int(kwargs.get("pp_hab_min", 30) or 30),
            "connectivity": _parse_connectivity(kwargs.get("pp_hab_conn")),
            "reassign_method": "neighbor_vote",
            "max_iterations": int(kwargs.get("pp_hab_maxiter", 3) or 3),
        },
    }

    config: Dict[str, Any] = {
        "run_mode": run_mode,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "FeatureConstruction": feature_construction,
        "HabitatSegmentation": segmentation,
        "processes": int(kwargs.get("processes", 2) or 2),
        "cap_processes_to_gpu_pool": bool(kwargs.get("cap_gpu", False)),
        "individual_subject_timeout_sec": kwargs.get("subj_to"),
        "individual_subject_spawn_timeout_sec": kwargs.get("spawn_to"),
        "individual_subject_graceful_shutdown_sec": float(kwargs.get("graceful_sd", 15) or 15),
        "on_subject_failure": str(kwargs.get("on_fail", "continue")),
        "oom_backoff": bool(kwargs.get("oom_back", False)),
        "oom_backoff_reduce_processes_by": int(kwargs.get("oom_red", 1) or 1),
        "resume": bool(kwargs.get("resume", True)),
        "strict_checkpoint_hash": bool(kwargs.get("strict_ck", False)),
        "checkpoint_dir": kwargs.get("ck_dir") or None,
        "force_rerun_subjects": _parse_subject_list(str(kwargs.get("force_rerun", ""))),
        "retry_failed_subjects": bool(kwargs.get("retry_fail", False)),
        "individual_subject_auto_retry_rounds": int(kwargs.get("retry_rounds", 2) or 2),
        "individual_subject_parallel_mode": str(kwargs.get("par_mode", "persistent")),
        "persistent_worker_max_consecutive_failures": int(kwargs.get("pers_max_fail", 1) or 1),
        "persistent_worker_recycle_after_tasks": int(kwargs.get("pers_recycle", 0) or 0),
        "clear_checkpoint_on_success": bool(kwargs.get("clear_ck", False)),
        "plot_curves": bool(kwargs.get("plot_curves", True)),
        "save_images": bool(kwargs.get("save_images", True)),
        "save_results_csv": bool(kwargs.get("save_results", True)),
        "random_state": int(kwargs.get("rs", 42) or 42),
        "debug": bool(kwargs.get("debug", False)),
        "verbose": bool(kwargs.get("verbose", True)),
    }
    if pipeline_path:
        config["pipeline_path"] = pipeline_path
    return config


__all__ = ["build_habitat_config_data"]
