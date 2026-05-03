---
name: habit-quickstart
description: Entry-point router for the HABIT (Habitat Analysis Biomedical Imaging Toolkit) package. Use this FIRST whenever a user mentions HABIT, habitat analysis, tumor sub-region clustering, intra-tumor heterogeneity, "生境分析", "亚区聚类", "habitat segmentation", or wants a multi-step radiomics workflow. Verifies environment, validates data layout, then hands off to the right specialist skill.
---

# HABIT Quickstart — Workflow Router

This is the **entry point** for the HABIT package. It does NOT execute analyses
on its own — its job is to (1) verify the environment, (2) validate the
user's data layout, (3) ask the right clarifying questions, and (4) hand off
to the correct specialized skill or recipe.

> 中文用户提示：本 skill 是 HABIT 工具包的总入口/向导。当用户说"做生境分析"、
> "做亚区聚类"、"分析肿瘤异质性"、"我有 MRI 想做 habitat" 等需求时，先读本文件
> 走完路由流程，再去调用对应子 skill 或 recipe。

## Step 0 — Environment check (MANDATORY first action)

Before doing anything else, run:

```bash
python skills/habit-quickstart/scripts/check_environment.py
```

This verifies:
- Python ≥ 3.8 (≥ 3.10 needed for AutoGluon)
- `habit` CLI is installed and on PATH
- All critical packages (SimpleITK, pandas, numpy, sklearn, click, PyYAML) import
- Reports any missing optional packages (PyRadiomics, ANTsPy, XGBoost, ...)

If exit code is 1 (failure), tell the user to install HABIT:
```bash
conda create -n habit python=3.8 -y
conda activate habit
pip install -r requirements.txt
pip install -e .
```

If exit code is 2 (warnings only), proceed but warn the user about missing
optional packages they'll need later.

## Step 1 — Data layout check

If the user mentions a data directory:

```bash
python skills/habit-quickstart/scripts/check_data_layout.py <data_dir> --modalities <list>
```

This catches missing modalities and bad folder layouts BEFORE writing any
config. See `references/data_layout_spec.md` for the canonical layout.

## Step 2 — Required Information

Read `references/required_questions.md` for the per-skill checklist. Do NOT
proceed if any required field is unanswered — stop and ask the user. This
is per project rule 9 (analyze + propose + get approval first).

## Step 3 — Route to the right skill

Use the decision tree in `references/workflow_decision_tree.md` and this
quick table:

| User goal | Hand off to |
|---|---|
| Convert raw DICOM → NIfTI | `habit-preprocess` (dcm2nii mode) |
| Align / normalize / N4-correct images | `habit-preprocess` |
| Discover tumor sub-regions | `habit-habitat-analysis` |
| Extract MSI / ITH / habitat features | `habit-feature-extraction` |
| Classical PyRadiomics only (no habitat) | `habit-radiomics` |
| Train / predict / k-fold a classifier | `habit-machine-learning` |
| Compare ROC / DCA / DeLong of multiple models | `habit-model-comparison` |
| ICC / test-retest / Dice / merge CSV / DICOM info | `habit-dicom-tools` |
| **End-to-end pipeline** | `habit-recipes` |
| Got an error / something broke | `habit-troubleshoot` |

## Step 4 — Decision: one_step vs two_step habitat

A common confusion. Default to **one_step** unless the user is doing a
publication-grade multi-cohort study.

- **one_step** — each tumor finds its own optimal cluster number. Faster,
  simpler, no population-level model. Good for pilots.
- **two_step** — produces a population-level habitat model that generalizes
  to new patients via `--mode predict`. Required for predictive modeling
  studies that need external validation.

Detail: `habit-habitat-analysis/references/cluster_validity_guide.md`.

## Critical project-wide rules (apply to all HABIT work)

1. **All plot/figure outputs use English labels** — project-wide rule. Never
   put Chinese characters on figures. (Code comments / documentation may be
   Chinese.)
2. **YAML configs**: 2-space indent, no tabs, space after colon.
3. **Paths in configs** can be relative (resolved against config file
   location) or absolute.
4. **Always validate after each step** — use `validate_*.py` scripts in
   each skill's `scripts/` folder.
5. **Demo data** is available — if the user is new, suggest
   `habit-recipes/references/recipe_demo_walkthrough.md` first.

## When to stop and ask the user

Do NOT proceed silently if:
- User did not specify which `images:` modalities they have
- User did not specify the binary `label_col` for ML
- User wants `predict` mode but did not provide a trained pipeline path
- DICOM-to-NIfTI conversion needs `dcm2niix.exe` path
- User has not picked one_step vs two_step (and you can't safely default)

## Resources

- Full CLI reference: `docs/source/cli_zh.rst`
- Configuration encyclopedia: `docs/source/configuration_zh.rst`
- Online docs: https://lichao312214129.github.io/HABIT
- Demo dataset: see `README.md` Step 2
- Annotated config templates: `config_templates/config_*_annotated.yaml`
- Agent YAML scaffold index (paths for skills / OpenClaw): `skills/CONFIG_SOURCES.md`
