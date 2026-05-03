# HABIT Agent Skills

A self-contained collection of **Agent Skills** that lets AI assistants
(Claude Code, Cursor, Claude Desktop, the Anthropic API, OpenCode, etc.)
drive the entire HABIT (Habitat Analysis Biomedical Imaging Toolkit) workflow
on behalf of clinicians and researchers — **with no Python coding required**.

> 这个文件夹包含一组 **Agent Skills**，可以让 AI 助手（Claude、Cursor、OpenCode 等）
> 代替临床医生和研究人员驱动整个 HABIT 工作流。用户只需要用自然语言描述需求
> （"我有一组肝癌 MRI，想做生境分析"），AI 就会自动调用合适的 skill、
> 生成配置文件、运行命令、解读结果。

## What is an Agent Skill?

An Agent Skill is a folder containing a `SKILL.md` file with YAML frontmatter
(`name` and `description`), optional reference files (templates, guides),
and optional executable scripts. The AI reads the `description` of every
available skill at startup, then **progressively loads** the full `SKILL.md`
(and any referenced files) only for the skill matching the current task.

Reference: [Anthropic Agent Skills documentation](https://docs.anthropic.com/en/docs/agents/skills)

## What you get

```
skills/
├── README.md                       # this file
├── CONFIG_SOURCES.md               # index: YAML scaffolds → annotated → config/ (single source map)
├── INSTALL.md                      # how to plug skills into different AI tools
│
├── habit-quickstart/               # ★ entry router — read this first
│   ├── SKILL.md
│   ├── references/                 # decision tree, required questions, data layout spec
│   └── scripts/                    # check_environment.py, check_data_layout.py
│
├── habit-preprocess/               # image preprocessing (resample/register/N4/normalize)
├── habit-habitat-analysis/         # tumor sub-region clustering (one_step / two_step)
├── habit-feature-extraction/       # MSI / ITH / habitat radiomics features
├── habit-machine-learning/         # train / predict / k-fold CV
├── habit-model-comparison/         # ROC / DCA / DeLong / calibration
├── habit-radiomics/                # classical PyRadiomics (no habitat)
├── habit-dicom-tools/              # DICOM info, CSV merge, ICC, test-retest, Dice
│
├── habit-recipes/                  # ★ end-to-end pipelines for common scenarios
│   └── references/                 # 5 recipes (MRI / DCE / CT / test-retest / demo)
│
└── habit-troubleshoot/             # ★ symptom→fix playbook for runtime errors
    └── references/                 # per-step error files + recovery playbook
```

Each skill is a self-contained directory:

```
<skill-name>/
├── SKILL.md                        # main instructions (loaded when triggered)
├── references/                     # Markdown guides only (YAML scaffolds live in config_templates/skill_scaffolds/)
└── scripts/                        # optional Python helpers (the AI runs these)
```

## Skill catalog

| Skill | Purpose | Trigger keywords (zh / en) |
|---|---|---|
| `habit-quickstart` | Entry router; environment + data check | "habitat", "生境分析", "start" |
| `habit-preprocess` | DICOM→NIfTI, resample, register, N4, normalize | "预处理", "配准", "preprocess", "register" |
| `habit-habitat-analysis` | Core voxel clustering (one_step / two_step) | "生境", "亚区", "habitat", "supervoxel" |
| `habit-feature-extraction` | MSI / ITH / habitat radiomics | "MSI", "ITH", "habitat features" |
| `habit-machine-learning` | Train / predict / k-fold | "建模", "训练", "k-fold", "LASSO" |
| `habit-model-comparison` | Multi-model ROC / DCA / DeLong | "模型比较", "DeLong", "ROC 对比" |
| `habit-radiomics` | Classical PyRadiomics (no habitat) | "传统影像组学", "PyRadiomics" |
| `habit-dicom-tools` | DICOM info, CSV merge, ICC, test-retest, Dice | "DICOM", "ICC", "test-retest", "Dice" |
| `habit-recipes` | End-to-end pipelines | "全流程", "end-to-end", "demo" |
| `habit-troubleshoot` | Diagnose runtime errors | "报错", "error", "failed", "AUC 太低" |

## Why this layout works for AI agents

1. **Description-driven matching** — Anthropic's Skills system loads the
   matching skill automatically from the `description` field. Each
   description is single-sentence, mentions both Chinese and English
   triggers, and explicitly says when NOT to use the skill.

2. **Progressive disclosure** — The agent doesn't read the whole skill
   bundle. It reads `SKILL.md` for the skill that matched, then opens
   reference files only as needed. Keeps context efficient.

3. **Self-validating workflow** — Every step has a corresponding
   `validate_*.py` or `inspect_*.py` script. The agent runs validation
   after each step before continuing, so failures are caught early
   rather than cascading.

4. **No-config-from-scratch policy** — Every skill points at minimal
   YAML scaffolds in its `references/` folder. The agent fills in
   `<PLACEHOLDER>` values from user-confirmed inputs rather than
   inventing field names.

5. **Project-rule compliance** — Every skill enforces the project rules:
   English plot labels, YAML 2-space indent, output directory
   conventions. See `habit-quickstart/SKILL.md` for the full list.

## How to install / use

See [INSTALL.md](INSTALL.md) for step-by-step instructions for:
- Claude Code (CLI)
- Claude Desktop
- Cursor
- Anthropic API
- OpenCode and other Skills-compatible tools

## Prerequisites

These skills assume the HABIT package is installed:

```bash
conda create -n habit python=3.8 -y
conda activate habit
pip install -r requirements.txt
pip install -e .

# Verify
habit --version
```

If HABIT is not installed, the first thing `habit-quickstart` does is run
`scripts/check_environment.py` and tell the user how to install.

## Example user interactions

These prompts will trigger the right skill chain automatically once the
skills are installed:

- *"I have a folder of MRI DICOMs, help me run the full HABIT pipeline."*
  → `habit-quickstart` → `habit-recipes/recipe_mri_habitat_full.md`

- *"我有 200 个肝癌患者的 T1/T2/DWI/ADC 数据，想做生境分析然后建模预测复发。"*
  → `habit-quickstart` → `habit-recipes/recipe_mri_habitat_full.md`

- *"DCE-MRI 多期相，想做 kinetic 分期生境，然后看哪些生境与 PFS 相关。"*
  → `habit-recipes/recipe_dce_kinetic.md` → `habit-machine-learning`

- *"Compare the ROC curves of my clinical and radiomics models with DeLong test."*
  → `habit-model-comparison`

- *"Extract returned an empty CSV, what's wrong?"*
  → `habit-troubleshoot/references/errors_extraction.md`

- *"我刚装好，想跑一下 demo 看看效果。"*
  → `habit-recipes/references/recipe_demo_walkthrough.md`

## Adding a new skill

1. Create `skills/<new-skill-name>/SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: <new-skill-name>
   description: <single sentence: what it does + when to use + key triggers>
   ---
   ```
2. Add reference templates in `references/`.
3. Add helper Python in `scripts/` (optional but recommended).
4. Add a row to the catalog table above.
5. Update `habit-quickstart/SKILL.md` and `habit-quickstart/references/workflow_decision_tree.md` if it changes routing.

## Citation

If you use HABIT (and these skills) in research, please cite the project per
the main `README.md`. Issues and PRs welcome at
https://github.com/lichao312214129/HABIT.
