---
name: habit-quickstart
description: Entry-point skill for the HABIT (Habitat Analysis Biomedical Imaging Toolkit) package. Use this skill when the user wants to perform tumor habitat / sub-region analysis on medical images (CT/MRI/DCE), mentions "habitat", "tumor heterogeneity", "sub-region clustering", "影像组学生境分析", "肿瘤异质性", "亚区聚类", or asks how to start with HABIT. This skill orchestrates the full workflow (preprocess → habitat → extract → model → compare) and routes the user to the right specialized HABIT skill.
---

# HABIT Quickstart — Workflow Router

This is the **entry skill** for the HABIT package (Habitat Analysis Biomedical Imaging Toolkit). It does NOT execute analyses on its own — its job is to:

1. Verify the environment is ready.
2. Understand what the user wants.
3. Hand off to the correct specialized skill.

> 中文用户提示：本 skill 是 HABIT 工具包的总入口/向导。当用户说"做生境分析"、"做亚区聚类"、"分析肿瘤异质性"、"我有 MRI 想做 habitat" 等需求时，先读本文件搞清流程，再去调用对应子 skill。

---

## Step 0: Environment Check

Before doing anything, verify the user has HABIT installed:

```bash
habit --version
```

If the command is not found:
- Tell the user to install: `pip install -e .` from the HABIT repo root, inside a conda env with Python 3.8 (or 3.10 if they want AutoGluon).
- HABIT GitHub: https://github.com/lichao312214129/HABIT

Confirm the data layout is one of these:
```
data_dir/
├── subject_001/
│   ├── images/   (T1.nii.gz, T2.nii.gz, ...)
│   └── masks/    (mask_T1.nii.gz, ...)
└── subject_002/...
```
If the user's data is DICOM, route to **habit-preprocess** first (it includes dcm2nii).

---

## Step 1: Understand the Goal

Ask the user **one question** to figure out which workflow they need. Map their answer to the right skill:

| User says... | Hand off to skill |
|---|---|
| "I have raw DICOM, need to convert to NIfTI" | `habit-preprocess` (dcm2nii mode) |
| "Images are not aligned / need registration / normalization / N4" | `habit-preprocess` |
| "I want to find tumor sub-regions / habitats / heterogeneity zones" | `habit-habitat-analysis` |
| "I have habitat maps, extract features (MSI / ITH / whole_habitat)" | `habit-feature-extraction` |
| "Just classical radiomics features (no habitat)" | `habit-radiomics` |
| "Train a classifier / build a prediction model / k-fold CV" | `habit-machine-learning` |
| "Compare multiple models (ROC/DCA/DeLong/calibration)" | `habit-model-comparison` |
| "Test-retest reliability / ICC / DICOM info / merge CSVs" | `habit-dicom-tools` |
| "Run the full pipeline end-to-end" | This skill — see Step 2 below |

---

## Step 2: Full End-to-End Workflow (one-shot mode)

Only if the user explicitly wants the **whole pipeline**. The 5-step canonical chain:

```bash
# 1. Preprocess (resample / register / normalize / optional dcm2nii)
habit preprocess --config <path>/config_image_preprocessing.yaml

# 2. Habitat segmentation (tumor sub-region clustering)
habit get-habitat --config <path>/config_habitat_one_step.yaml

# 3. Feature extraction (radiomics + MSI + ITH from habitat maps)
habit extract --config <path>/config_extract_features.yaml

# 4. Machine learning (train + evaluate)
habit model --config <path>/config_machine_learning.yaml --mode train

# 5. Model comparison (multi-model ROC/DCA/DeLong)
habit compare --config <path>/config_model_comparison.yaml
```

For each step, **before running**, read the corresponding specialized skill (e.g. `habit-preprocess/SKILL.md`) and help the user fill the YAML config.

Annotated config templates (always link the user to these — never hand-write configs from scratch):
- `config_templates/config_image_preprocessing_annotated.yaml`
- `config_templates/config_getting_habitat_annotated.yaml`
- `config_templates/config_habitat_one_step_example.yaml`  (simpler, for beginners)
- `config_templates/config_extract_features_annotated.yaml`
- `config_templates/config_machine_learning_annotated.yaml`
- `config_templates/config_model_comparison_annotated.yaml`

---

## Step 3: Decision Tree — One-Step vs Two-Step Habitat

A common confusion. Help the user pick:

- **one_step** (recommended for beginners, single-cohort studies):
  - Each tumor independently determines optimal cluster number
  - Faster, simpler, no population-level model
  - Use template: `config_habitat_one_step_example.yaml`

- **two_step** (for multi-cohort / publication-grade studies):
  - Step 1: voxel → supervoxel (per subject)
  - Step 2: supervoxel → habitat (population-level model, applies to new test data)
  - Required if they want a model that generalizes to new patients
  - Use template: `config_getting_habitat_annotated.yaml`

If the user is unsure, start with **one_step**.

---

## Critical Rules (apply to all HABIT work)

1. **All plots must use English labels** — it's a project-wide rule. Never put Chinese characters on figures. (Documents/comments may be in Chinese.)
2. **Code edits**: utilities go in `habit/utils/`, progress bars must use `habit/utils/progress_utils.py`.
3. **Configs are YAML** — 2-space indent, no tabs, space after colon.
4. **Paths in configs** can be relative (resolved against config file location) or absolute.
5. **Demo data** is available in `demo_data/` — always offer to run it first if the user is new.

---

## When to Stop and Ask the User

Do NOT proceed silently if:
- User did not specify which `images:` modalities they have (T1/T2/DWI/ADC/DCE phases?)
- User did not specify the binary `label_col` for ML
- User wants `predict` mode but did not provide a trained pipeline path
- DICOM-to-NIfTI conversion needs `dcm2niix.exe` path

Always confirm these before generating configs.

---

## Resources

- Full CLI reference: `docs/source/cli_zh.rst`
- Configuration encyclopedia: `docs/source/configuration_zh.rst`
- Online docs: https://lichao312214129.github.io/HABIT
- Demo dataset: see `README.md` Step 2 for download link
