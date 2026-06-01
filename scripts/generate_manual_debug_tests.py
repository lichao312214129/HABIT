"""Generate demo YAML under config/ and one intuitive test script per scenario under tests/."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "#%%================================================================================"

DATA_DIR = "../../.cursor/test/resample_02"

RUN_TAIL = """processes: 2
cap_processes_to_gpu_pool: false
individual_subject_timeout_sec: 900
oom_backoff: false

resume: true
strict_checkpoint_hash: false
checkpoint_dir: null
force_rerun_subjects: []
retry_failed_subjects: false
individual_subject_auto_retry_rounds: 2
individual_subject_parallel_mode: persistent
persistent_worker_max_consecutive_failures: 1
persistent_worker_recycle_after_tasks: 0
clear_checkpoint_on_success: false

plot_curves: true
save_results_csv: true
random_state: 42
debug: false
"""

PREPROCESS_SUBJECT = """  preprocessing_for_subject_level:
    # Stage 1: per-subject, before checkpoint (included in config_hash).
    methods:
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: false
      - method: minmax
        global_normalize: false
"""

PREPROCESS_GROUP_BIN = """  preprocessing_for_group_level:
    # Stage 2 only — NOT in config_hash.
    methods:
      - method: binning
        n_bins: 10
        bin_strategy: uniform
        global_normalize: false
"""

PREPROCESS_GROUP_VAR = """  preprocessing_for_group_level:
    # Stage 2 only — NOT in config_hash. Prefer for radiomics-heavy features.
    methods:
      - method: variance_filter
        variance_threshold: 0.01
        global_normalize: false
      - method: zscore
        global_normalize: false
"""

HABITAT_SUPER = """  habitat:
    algorithm: kmeans
    max_clusters: 10
    habitat_cluster_selection_method:
      - elbow
    fixed_n_clusters:
    max_iter: 300
    n_init: 10
"""

SUPERVOXEL_BLOCK = """  supervoxel:
    algorithm: kmeans
    n_clusters: 50
    max_iter: 300
    n_init: 10
    one_step_settings:
      min_clusters: 2
      max_clusters: 10
      fixed_n_clusters:
      selection_method: elbow
      plot_validation_curves: true
"""

SUPERVOXEL_PARAMS = """    params:
      params_file: ../radiomics/params_supervoxel_radiomics.yaml
      supervoxelUnionBboxCrop: true
      useSupervoxelCext: auto
"""

FEATURE: dict[str, dict[str, str]] = {
    "raw_concat": {
        "title": "raw intensity concat — concat(raw(T1), raw(T2))",
        "title_cn": "原始强度拼接特征",
        "voxel": "    method: concat(raw(T1), raw(T2))\n    params: {}",
        "super": "    method: mean_voxel_features()",
        "group_pre": PREPROCESS_GROUP_BIN,
    },
    "voxel_radiomics": {
        "title": "voxel texture — voxel_radiomics(T2, params_file, kernelRadius)",
        "title_cn": "体素纹理（影像组学）特征",
        "voxel": """    method: concat(voxel_radiomics(T2, params_file, kernelRadius))
    params:
      params_file: ../radiomics/params_voxel_radiomics.yaml
      kernelRadius: 1
      voxelBatch: 1000
      useTorchRadiomics: auto""",
        "super": "    method: mean_voxel_features()",
        "group_pre": PREPROCESS_GROUP_VAR,
    },
    "supervoxel_radiomics": {
        "title": "supervoxel texture — supervoxel_radiomics(T2, params_file)",
        "title_cn": "超体素纹理（影像组学）特征",
        "voxel": "    method: concat(raw(T1), raw(T2))\n    params: {}",
        "super": """    method: concat(supervoxel_radiomics(T2, params_file))
    params:
      params_file: ../radiomics/params_supervoxel_radiomics.yaml
      supervoxelUnionBboxCrop: true
      useSupervoxelCext: auto
      useTorchRadiomics: auto""",
        "group_pre": PREPROCESS_GROUP_VAR,
    },
}

MODES = {
    "two_step": ("two_step", "Two-step clustering (supervoxel then population habitat)"),
    "one_step": ("one_step", "One-step clustering (voxel to habitat per subject)"),
    "pooling": ("direct_pooling", "Direct pooling (pool all voxels, cluster once)"),
}

MODE_CN = {
    "two_step": "二步法生境",
    "one_step": "一步法生境",
    "pooling": "直接池化生境",
}

MODE_FILE_PREFIX = {
    "two_step": "two_step",
    "one_step": "one_step",
    "pooling": "direct_pooling",
}

RUN_CN = {"train": "训练", "predict": "预测"}

FEATURES_BY_MODE = {
    "two_step": ["raw_concat", "voxel_radiomics", "supervoxel_radiomics"],
    "one_step": ["raw_concat", "voxel_radiomics"],
    "pooling": ["raw_concat", "voxel_radiomics"],
}

HABITAT_REUSE: dict[str, str] = {
    "two_step_raw_concat_train": "config/habitat/config_habitat_two_step.yaml",
    "two_step_raw_concat_predict": "config/habitat/config_habitat_two_step_predict.yaml",
    "pooling_raw_concat_train": "config/habitat/config_habitat_direct_pooling.yaml",
    "pooling_raw_concat_predict": "config/habitat/config_habitat_direct_pooling_predict.yaml",
}

ML_SCRIPTS: dict[str, tuple[str, list[str], str, str]] = {
    "ml_train_clinical": (
        "model",
        ["-m", "train"],
        "config/machine_learning/config_machine_learning_clinical.yaml",
        "临床特征机器学习 — 训练（单表）",
    ),
    "ml_train_radiomics": (
        "model",
        ["-m", "train"],
        "config/machine_learning/config_machine_learning_radiomics.yaml",
        "影像组学机器学习 — 训练（单表）",
    ),
    "ml_train_multi_tables": (
        "model",
        ["-m", "train"],
        "config/machine_learning/config_machine_learning_multi_tables_demo.yaml",
        "多表特征机器学习 — 训练（临床 + 影像组学）",
    ),
    "ml_predict": (
        "model",
        ["-m", "predict"],
        "config/machine_learning/config_machine_learning_predict.yaml",
        "机器学习 — 预测",
    ),
    "ml_kfold_cross_validation": (
        "cv",
        [],
        "config/machine_learning/config_machine_learning_kfold_demo.yaml",
        "K 折交叉验证",
    ),
    "ml_radiomics_standalone": (
        "radiomics",
        [],
        "config/radiomics/config_traditional_radiomics.yaml",
        "传统影像组学（独立流程）",
    ),
}

COMPARE_SCRIPTS: dict[str, tuple[str, str]] = {
    "compare_models_multi": (
        "config/model_comparison/config_model_comparison_demo.yaml",
        "多模型对比（ROC、校准曲线等）",
    ),
    "compare_models_single_radiomics": (
        "config/model_comparison/config_model_comparison_single_radiomics.yaml",
        "单模型（影像组学）对比",
    ),
}


def _habitat_script_name(mode_key: str, feat_key: str, run_mode: str) -> str:
    prefix = MODE_FILE_PREFIX[mode_key]
    return f"habitat_{prefix}_{feat_key}_{run_mode}.py"


def _cli_py(title_cn: str, test_rel: str, subcommand: str, cfg_rel: str, extra_args: list[str] | None = None) -> str:
    parts = ["habit", subcommand, "-c", cfg_rel, *(extra_args or []), "*sys.argv[1:]"]
    argv_inner = ", ".join(f'"{p}"' if p != "*sys.argv[1:]" else p for p in parts)
    return f'''"""{title_cn}

Config: {cfg_rel}
Run:    python tests/{test_rel}

Edit the YAML above (#%% path blocks) for your own data. Optional: pass --debug
"""

import os
import sys
from pathlib import Path


def main() -> None:
    """Invoke habit CLI from repository root (Windows spawn-safe)."""
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [{argv_inner}]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
'''


def _workflow_py(title_cn: str, test_rel: str, steps: list[list[str]]) -> str:
    steps_repr = repr(steps)
    return f'''"""{title_cn}

Run: python tests/{test_rel}

Each step uses its own config under config/ — edit paths in those YAML files.
"""

import os
import sys
from pathlib import Path

STEPS = {steps_repr}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from habit.cli import cli

    extra = sys.argv[1:]
    for i, step in enumerate(STEPS, 1):
        print(f"\\n--- step {{i}}/{{len(STEPS)}}: habit {{' '.join(step)}} ---\\n")
        sys.argv = ["habit", *step, *extra]
        cli()


if __name__ == "__main__":
    main()
'''


def _util_py(title_cn: str, test_rel: str, argv_after_habit: list[str]) -> str:
    tail = ", ".join(f'"{p}"' for p in argv_after_habit)
    return f'''"""{title_cn}

Run: python tests/{test_rel}
"""

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = ["habit", {tail}, *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
'''


def _habitat_banner(
    mode_key: str,
    mode_label: str,
    feat_key: str,
    feat_title: str,
    run_mode: str,
    yaml_name: str,
    script_name: str,
) -> str:
    title_cn = f"{MODE_CN[mode_key]} — {FEATURE[feat_key]['title_cn']} — {RUN_CN[run_mode]}"
    return f"""# =============================================================================
# Habitat — {title_cn}
# =============================================================================
# CLI:  habit get-habitat --config config/habitat/{yaml_name}
#       python tests/habitat/{script_name}
#
# Purpose:
#   {mode_label}
#   Feature: {feat_title}
#   Demo data: .cursor/test/resample_02 (T1, T2; images/ + masks/ under data_dir root)
#
# Must-edit keys: wrapped in {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================
"""


def _habitat_yaml(mode_key: str, feat_key: str, run_mode: str) -> tuple[str, str, str]:
    clustering, mode_label = MODES[mode_key]
    feat_cfg = FEATURE[feat_key]
    yaml_name = f"config_habitat_{mode_key}_{feat_key}_{run_mode}.yaml"
    script_name = _habitat_script_name(mode_key, feat_key, run_mode)
    out_suffix = f"{mode_key}_{feat_key}" if run_mode == "train" else f"{mode_key}_{feat_key}/predict"
    out_dir = f"../../demo_data/results/habitat_{out_suffix}"

    lines = [
        _habitat_banner(
            mode_key, mode_label, feat_key, feat_cfg["title"], run_mode, yaml_name, script_name
        ).rstrip(),
        "",
        MARKER,
    ]
    if run_mode == "predict":
        train_out = f"../../demo_data/results/habitat_{mode_key}_{feat_key}"
        lines.append("run_mode: predict")
        lines.append(f"pipeline_path: {train_out}/habitat_pipeline.pkl")
    else:
        lines.append("run_mode: train")
    lines.extend(
        [
            f"data_dir: {DATA_DIR}",
            f"out_dir: {out_dir}",
            MARKER,
            "",
            "FeatureConstruction:",
            "  voxel_level:",
            feat_cfg["voxel"],
        ]
    )
    if clustering != "one_step":
        lines.extend(
            [
                "",
                "  supervoxel_level:",
                "    supervoxel_file_keyword: '*_supervoxel.nrrd'",
                feat_cfg["super"],
            ]
        )
        if feat_key != "supervoxel_radiomics":
            lines.append(SUPERVOXEL_PARAMS)
    lines.extend(
        [
            "",
            PREPROCESS_SUBJECT,
            feat_cfg["group_pre"],
            "",
            "HabitatSegmentation:",
            f"  clustering_mode: {clustering}",
            "",
            SUPERVOXEL_BLOCK,
            "",
            HABITAT_SUPER,
            "",
        ]
    )
    if clustering == "direct_pooling":
        lines.append("save_images: true")
    lines.append(RUN_TAIL)
    return yaml_name, script_name, "\n".join(lines) + "\n"


def _write_script(folder: Path, filename: str, content: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding="utf-8")


def generate_habitat() -> None:
    hab_dir = ROOT / "config" / "habitat"
    hab_test = ROOT / "tests" / "habitat"

    for mode_key in MODES:
        for feat_key in FEATURES_BY_MODE[mode_key]:
            for run_mode in ("train", "predict"):
                internal = f"{mode_key}_{feat_key}_{run_mode}"
                script_name = _habitat_script_name(mode_key, feat_key, run_mode)
                reuse = HABITAT_REUSE.get(internal)
                if reuse:
                    cfg_rel = reuse
                    title_cn = (
                        f"{MODE_CN[mode_key]} — {FEATURE[feat_key]['title_cn']} — {RUN_CN[run_mode]}"
                    )
                else:
                    yaml_name, script_name, body = _habitat_yaml(mode_key, feat_key, run_mode)
                    (hab_dir / yaml_name).write_text(body, encoding="utf-8")
                    cfg_rel = f"config/habitat/{yaml_name}"
                    title_cn = (
                        f"{MODE_CN[mode_key]} — {FEATURE[feat_key]['title_cn']} — {RUN_CN[run_mode]}"
                    )
                _write_script(
                    hab_test,
                    script_name,
                    _cli_py(title_cn, f"habitat/{script_name}", "get-habitat", cfg_rel),
                )


def generate_preprocessing() -> None:
    pre_dir = ROOT / "config" / "preprocessing"
    pre_test = ROOT / "tests" / "preprocessing"

    cases: dict[str, tuple[str, str | None, str, str]] = {
        "preprocess_registration_elastix": (
            "config_preprocessing_demo.yaml",
            None,
            "preprocess",
            "预处理 — Elastix 配准（resample_02, T1/T2）",
        ),
        "preprocess_resample_only": (
            "config_preprocessing_resample_only.yaml",
            f"""# =============================================================================
# Preprocessing — resample only
# =============================================================================
# CLI:  habit preprocess --config config/preprocessing/config_preprocessing_resample_only.yaml
#       python tests/preprocessing/preprocess_resample_only.py
# Must-edit keys: {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================

{MARKER}
data_dir: {DATA_DIR}
out_dir: ../../demo_data/results/preprocessing_resample_only
{MARKER}

auto_select_first_file: true

Preprocessing:
  resample:
    images: [T1, T2]
    target_spacing: [2.0, 2.0, 2.0]

save_options:
  save_intermediate: false

processes: 1
random_state: 42
""",
            "preprocess",
            "预处理 — 仅重采样",
        ),
        "preprocess_n4_resample_registration": (
            "config_preprocessing_n4_resample_registration.yaml",
            f"""# =============================================================================
# Preprocessing — N4 + resample + elastix registration
# =============================================================================
# CLI:  habit preprocess --config config/preprocessing/config_preprocessing_n4_resample_registration.yaml
#       python tests/preprocessing/preprocess_n4_resample_registration.py
# Must-edit keys: {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================

{MARKER}
data_dir: {DATA_DIR}
out_dir: ../../demo_data/results/preprocessing_n4_resample_registration
{MARKER}

auto_select_first_file: true

Preprocessing:
  n4_correction:
    images: [T1, T2]
    num_fitting_levels: 2
  resample:
    images: [T1, T2]
    target_spacing: [2.0, 2.0, 2.0]
  registration:
    images: [T1, T2]
    fixed_image: T1
    use_mask: true
    backend: elastix
    elastix_parameter_files: ../../demo_data/Par0040affine.txt

save_options:
  save_intermediate: true
  intermediate_steps: [n4_correction, resample, registration]

processes: 1
random_state: 42
""",
            "preprocess",
            "预处理 — N4偏场校正 + 重采样 + 配准",
        ),
        "preprocess_resample02": (
            "config_preprocessing_dcm2nii_demo.yaml",
            f"""# =============================================================================
# Preprocessing — resample_02 NIfTI resample (T1, T2)
# =============================================================================
# CLI:  habit preprocess --config config/preprocessing/config_preprocessing_dcm2nii_demo.yaml
#       python tests/preprocessing/preprocess_resample02.py
# Data: .cursor/test/resample_02 (already NIfTI; resample step only)
# Must-edit keys: {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================

{MARKER}
data_dir: {DATA_DIR}
out_dir: ../../demo_data/results/preprocessing_resample_02
{MARKER}

auto_select_first_file: true

Preprocessing:
  resample:
    images: [T1, T2]
    target_spacing: [2.0, 2.0, 2.0]

save_options:
  save_intermediate: true
  intermediate_steps: [resample]

processes: 1
random_state: 42
""",
            "preprocess",
            "预处理 — resample_02 重采样（T1/T2）",
        ),
    }

    for script_name, (yaml_name, body, subcommand, title_cn) in cases.items():
        cfg_rel = f"config/preprocessing/{yaml_name}"
        if body is not None:
            (pre_dir / yaml_name).write_text(body, encoding="utf-8")
        _write_script(
            pre_test,
            f"{script_name}.py",
            _cli_py(title_cn, f"preprocessing/{script_name}.py", subcommand, cfg_rel),
        )


def generate_feature_extraction() -> None:
    ext_dir = ROOT / "config" / "feature_extraction"
    ext_test = ROOT / "tests" / "feature_extraction"

    yaml_name = "config_extract_features_demo.yaml"
    script_name = "extract_features.py"
    title_cn = "特征提取 — 全部类型"
    body = f"""# =============================================================================
# Feature extraction — {title_cn}
# =============================================================================
# CLI:  habit extract --config config/feature_extraction/{yaml_name}
#       python tests/feature_extraction/{script_name}
# Must-edit keys: {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================

{MARKER}
raw_img_folder: {DATA_DIR}
habitats_map_folder: ../../demo_data/results/habitat_two_step
out_dir: ../../demo_data/results/features_demo
{MARKER}

params_file_of_non_habitat: ../radiomics/parameter.yaml
params_file_of_habitat: ../radiomics/parameter_habitat.yaml

n_processes: 2
habitat_pattern: '*_habitats.nrrd'
feature_types:
  - traditional
  - non_radiomics
  - whole_habitat
  - msi
  - ith_score

n_habitats:
debug: false
"""
    (ext_dir / yaml_name).write_text(body, encoding="utf-8")
    cfg_rel = f"config/feature_extraction/{yaml_name}"
    _write_script(
        ext_test,
        script_name,
        _cli_py(title_cn, f"feature_extraction/{script_name}", "extract", cfg_rel),
    )


def generate_machine_learning() -> None:
    ml_test = ROOT / "tests" / "machine_learning"
    for script_name, (cmd, extra, cfg_rel, title_cn) in ML_SCRIPTS.items():
        _write_script(
            ml_test,
            f"{script_name}.py",
            _cli_py(title_cn, f"machine_learning/{script_name}.py", cmd, cfg_rel, extra),
        )


def generate_model_comparison() -> None:
    mc_dir = ROOT / "config" / "model_comparison"
    mc_test = ROOT / "tests" / "model_comparison"

    for script_name, (cfg_rel, title_cn) in COMPARE_SCRIPTS.items():
        _write_script(
            mc_test,
            f"{script_name}.py",
            _cli_py(title_cn, f"model_comparison/{script_name}.py", "compare", cfg_rel),
        )

    single_yaml = "config_model_comparison_single_radiomics.yaml"
    if not (mc_dir / single_yaml).is_file():
        body = f"""# =============================================================================
# Model comparison — single radiomics model
# =============================================================================
# CLI:  habit compare --config config/model_comparison/{single_yaml}
#       python tests/model_comparison/compare_models_single_radiomics.py
# Must-edit keys: {MARKER} pairs (config/README_CONFIG.md)
# =============================================================================

{MARKER}
output_dir: ../../demo_data/results/model_comparison_single_radiomics
{MARKER}

files_config:
  - path: ../../demo_data/results/ml/radiomics/all_prediction_results.csv
    model_name: radiomics
    subject_id_col: subject_id
    label_col: label
    prob_col: LogisticRegression_prob
    pred_col: LogisticRegression_pred
    split_col: dataset

merged_data:
  enabled: false

split:
  enabled: true

visualization:
  roc:
    enabled: true
  calibration:
    enabled: true
    n_bins: 10

delong_test:
  enabled: false

metrics:
  basic_metrics:
    enabled: true
"""
        (mc_dir / single_yaml).write_text(body, encoding="utf-8")


def generate_integration() -> None:
    int_test = ROOT / "tests" / "integration"

    _write_script(
        int_test,
        "analysis_icc.py",
        _cli_py(
            "ICC 一致性分析",
            "integration/analysis_icc.py",
            "icc",
            "config/auxiliary/config_icc_demo.yaml",
        ),
    )
    _write_script(
        int_test,
        "analysis_test_retest.py",
        _cli_py(
            "Test-Retest 分析",
            "integration/analysis_test_retest.py",
            "retest",
            "config/auxiliary/config_test_retest.yaml",
        ),
    )

    workflow_full = [
        ["preprocess", "-c", "config/preprocessing/config_preprocessing_demo.yaml"],
        ["get-habitat", "-c", "config/habitat/config_habitat_two_step.yaml"],
        ["extract", "-c", "config/feature_extraction/config_extract_features_demo.yaml"],
        ["model", "-c", "config/machine_learning/config_machine_learning_clinical.yaml", "-m", "train"],
        ["compare", "-c", "config/model_comparison/config_model_comparison_demo.yaml"],
    ]
    workflow_resample = [
        ["preprocess", "-c", "config/preprocessing/config_preprocessing_resample_only.yaml"],
        *workflow_full[1:],
    ]

    _write_script(
        int_test,
        "workflow_preprocess_to_compare.py",
        _workflow_py("全流程 — 预处理 → 生境 → 特征 → 建模 → 对比", "integration/workflow_preprocess_to_compare.py", workflow_full),
    )
    _write_script(
        int_test,
        "workflow_resample_to_compare.py",
        _workflow_py(
            "全流程 — resample_02 重采样 → 生境 → 特征 → 建模 → 对比",
            "integration/workflow_resample_to_compare.py",
            workflow_resample,
        ),
    )


def generate_utils() -> None:
    util_test = ROOT / "tests" / "utils"
    _write_script(
        util_test,
        "util_dice_overlap.py",
        _util_py(
            "工具 — Dice 重叠度",
            "utils/util_dice_overlap.py",
            [
                "dice",
                "--input1",
                ".cursor/test/resample_02",
                "--input2",
                ".cursor/test/resample_02",
                "--output",
                "demo_data/results/dice_resample02_demo.csv",
                "--mask-keyword",
                "masks",
            ],
        ),
    )
    _write_script(
        util_test,
        "util_merge_csv.py",
        '''"""工具 — 合并 CSV

Run: python tests/utils/util_merge_csv.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))

    from habit.cli import cli
    from habit.utils.log_utils import stop_queue_listener

    out_path = root / "demo_data" / "results" / "merge_csv_demo.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        file1 = tmp / "part_a.csv"
        file2 = tmp / "part_b.csv"
        pd.DataFrame({"subjID": ["sub1", "sub2"], "feature_a": [1.0, 2.0]}).to_csv(
            file1, index=False
        )
        pd.DataFrame({"subjID": ["sub1", "sub2"], "feature_b": [10.0, 20.0]}).to_csv(
            file2, index=False
        )
        try:
            sys.argv = [
                "habit",
                "merge-csv",
                str(file1),
                str(file2),
                "-o",
                str(out_path),
                "--index-col",
                "subjID",
                *sys.argv[1:],
            ]
            cli()
        finally:
            stop_queue_listener()


if __name__ == "__main__":
    main()
''',
    )


def remove_manual_folder() -> None:
    manual = ROOT / "config" / "manual"
    if manual.is_dir():
        shutil.rmtree(manual)


def main() -> None:
    generate_habitat()
    generate_preprocessing()
    generate_feature_extraction()
    generate_machine_learning()
    generate_model_comparison()
    generate_integration()
    generate_utils()
    remove_manual_folder()
    print("Done.")


if __name__ == "__main__":
    main()
