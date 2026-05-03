---
name: habit-recipes
description: End-to-end HABIT workflow recipes for common research scenarios (multi-modal MRI habitat, DCE-MRI kinetic, CT-only radiomics, test-retest reproducibility, demo dataset walkthrough). Use when the user wants to run the entire pipeline (preprocess -> habitat -> features -> ML -> comparison) instead of one isolated step. Triggers on phrases like "全流程", "端到端", "整个流程跑一遍", "demo 走一遍", "full pipeline", "end-to-end", "complete workflow".
---

# HABIT End-to-End Recipes

Pre-baked workflows for the most common research scenarios. Each recipe lists:
1. The user phrase that should trigger it
2. Required inputs the agent must collect
3. The exact CLI sequence to run
4. Where each output lands
5. A validation checkpoint after every step

## Available recipes

| Recipe file | When to use | Modality |
|---|---|---|
| `references/recipe_mri_habitat_full.md` | Multi-modal MRI habitat study | T1+T2(+DWI/ADC) |
| `references/recipe_dce_kinetic.md` | DCE dynamic habitat study | DCE-MRI / dynamic CT |
| `references/recipe_ct_radiomics_only.md` | Classical radiomics on CT | CT |
| `references/recipe_test_retest.md` | Reproducibility / ICC study | any modality, two scans |
| `references/recipe_demo_walkthrough.md` | First-time user demo | bundled demo data |

## How the agent should use this skill

1. Identify which recipe matches the user's request (use the table above).
2. Open the corresponding recipe file in `references/`.
3. Walk the user through the **Required Inputs** checklist BEFORE writing
   any config — do not invent paths or modality names.
4. Generate each YAML config in turn, starting from scaffolds under
   `config_templates/skill_scaffolds/` (see `skills/CONFIG_SOURCES.md` for the map to annotated templates).
5. Run each command, then immediately call the `scripts/validate_*.py` from
   the corresponding skill to verify outputs before moving on.
6. If a step fails, hand off to `habit-troubleshoot`.

## Recipe selection cheat sheet

```
User says...                                                Pick recipe...
---------------------------------------------------------- -------------------------------
"我有 T1+T2+DWI+ADC, 想做生境分析建模"                       recipe_mri_habitat_full
"DCE 动态对比增强分期生境 / kinetic"                         recipe_dce_kinetic
"我只有 CT, 想做传统影像组学"                                recipe_ct_radiomics_only
"两次扫描看重复性 / test-retest / ICC"                       recipe_test_retest
"我刚装好, 想跑一下 demo 看看"                               recipe_demo_walkthrough
"我想跑全流程"                                               ASK which scenario, then pick
```

## Universal best practices for any recipe

1. **Always run `check_environment.py` first** — verify HABIT is installed.
2. **Always run `check_data_layout.py` before preprocess** — catches missing
   modalities early.
3. **Always validate output after each major step** — `validate_preprocess_output.py`
   then `validate_habitat_output.py` then `inspect_feature_csv.py`.
4. **Save every config the agent generates** in a `./configs/` subfolder of
   the user's working directory; reproducibility matters.
5. **Use absolute paths in configs** when the user has files scattered across
   drives; use relative when everything is under one project root.
