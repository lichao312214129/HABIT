---
name: habit-troubleshoot
description: Diagnose and fix HABIT runtime errors — covers preprocess, habitat clustering, feature extraction, and ML/comparison failures. Use when the user pastes a Python traceback, when a `habit ...` CLI command fails, when an output file is missing or corrupt, or when results look biologically wrong (degenerate clusters, all-NaN features, AUC=0.5). Triggers on phrases like "报错", "出错", "不工作", "为什么", "error", "failed", "traceback", "crash", "AUC 太低", "没有生成结果".
---

# HABIT Troubleshooting

When something goes wrong, don't guess — match the symptom against the
playbook below. Each error file in `references/` covers one HABIT command.

## How the agent should use this skill

1. Identify which step failed (preprocess / habitat / extract / ml).
2. Open the corresponding error file:
   - Preprocess errors → `references/errors_preprocess.md`
   - Habitat errors → `references/errors_habitat.md`
   - Extraction errors → `references/errors_extraction.md`
   - ML / comparison errors → `references/errors_ml.md`
   - Cross-cutting recovery → `references/recovery_playbook.md`
3. Search the file for the symptom (error message text, file content).
4. Apply the suggested fix. If it works, log what was wrong (helps catch
   recurring issues).

## When to ask the user for more info

If the user just says "it failed" without details, ask for:
1. Which `habit ...` command they ran
2. The full traceback (last ~20 lines)
3. Whether the user has changed the config since the last successful run
4. Output of `python skills/habit-quickstart/scripts/check_environment.py`

Do NOT recommend solutions blindly. The error message is almost always
enough to localize the cause.

## Universal first response to any failure

Run the appropriate validator script — it catches 80% of issues without
needing the traceback:

| If failure was during... | Run this |
|---|---|
| `habit preprocess` | `validate_preprocess_output.py <out_dir>` |
| `habit get-habitat` | `validate_habitat_output.py <out_dir>` |
| `habit extract` | `inspect_feature_csv.py` on each output CSV |
| `habit model` | `inspect_feature_csv.py` on the input CSV |
| `habit compare` | `inspect_feature_csv.py` on each prediction CSV |

## Hand-off to other skills

- If error reveals a **wrong config**: regenerate from the corresponding
  specialist skill's `references/` template.
- If error reveals **missing data**: hand off to `habit-quickstart`'s
  `check_data_layout.py`.
- If error is **environment-level** (`ImportError`, `command not found`):
  ask user to rerun `check_environment.py` and report back.

## Known unfixable cases

These cases are NOT software bugs — explain to the user and stop:
1. **Image registration failed because images are anatomically incompatible**
   (e.g. trying to register brain MRI to abdominal CT). Solution: re-acquire
   or use a different fixed_image.
2. **Tumor too small for habitat clustering** (< 100 voxels). Solution:
   the user must either redraw a larger ROI or accept that habitat
   analysis is inappropriate here.
3. **All clinical labels are 0 (or all 1)**. Solution: there is no model
   to train; this is a data problem.
