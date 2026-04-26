# HABIT Agent Skills

This folder contains a collection of **Agent Skills** that let AI assistants (Claude, Cursor, etc.) drive the HABIT (Habitat Analysis Biomedical Imaging Toolkit) workflow on behalf of clinicians and researchers.

> 这个文件夹包含一组 **Agent Skills**，可以让 AI 助手（Claude、Cursor 等）代替临床医生和研究人员驱动整个 HABIT 工作流。用户只需要用自然语言描述需求（"我有一组肝癌 MRI，想做生境分析"），AI 就会自动调用合适的 skill、生成配置文件、运行命令、解读结果。

---

## What is an Agent Skill?

An Agent Skill is a folder containing a `SKILL.md` file with YAML frontmatter (`name` and `description`) plus optional reference files. The AI assistant reads the `description` of every available skill at startup, then automatically loads the full `SKILL.md` when the user's request matches.

This is called **progressive disclosure** — the AI only loads the detailed instructions for the skill it actually needs, keeping context efficient.

Reference: [Anthropic Agent Skills documentation](https://docs.anthropic.com/en/docs/agents/skills)

---

## Available Skills

| Skill | Purpose | Trigger keywords (中英) |
|---|---|---|
| **`habit-quickstart`** | Workflow router / entry point | "做生境分析", "habitat", "start with HABIT", "5分钟快速开始" |
| **`habit-preprocess`** | Image preprocessing (DICOM→NIfTI, registration, N4, normalization) | "图像预处理", "重采样", "配准", "preprocess images" |
| **`habit-habitat-analysis`** | Core habitat clustering (one_step / two_step) | "生境聚类", "亚区分析", "habitat segmentation", "supervoxel" |
| **`habit-feature-extraction`** | Habitat-based features (MSI, ITH, whole/each habitat radiomics) | "提取生境特征", "MSI features", "ITH score", "habitat features" |
| **`habit-machine-learning`** | Train / predict / k-fold CV with feature selection | "训练模型", "机器学习建模", "K折交叉验证", "LASSO" |
| **`habit-model-comparison`** | Multi-model comparison (ROC/DCA/calibration/DeLong) | "模型比较", "ROC 对比", "DeLong 检验", "model comparison" |
| **`habit-radiomics`** | Classical PyRadiomics extraction (no habitat needed) | "传统影像组学", "PyRadiomics", "radiomics features" |
| **`habit-dicom-tools`** | DICOM info, CSV merge, ICC, test-retest, Dice | "DICOM 信息", "合并 CSV", "ICC 分析", "test-retest", "Dice" |

Each skill is self-contained:
```
<skill-name>/
├── SKILL.md                    # main instructions for the AI
└── references/                 # optional templates and examples
    ├── config_*_minimal.yaml   # ready-to-fill config scaffolds
    └── ...
```

---

## How to Use These Skills

### Option 1: Cursor

1. Place the `skills/` folder at the project root (already done in this repo).
2. Optionally copy or symlink to `.cursor/skills/` so Cursor auto-discovers them per workspace.
3. Open Cursor in this project — skills are detected automatically.
4. In chat, just describe what you want. Examples:
   - "I have a folder of MRI DICOMs, help me run the full HABIT pipeline."
   - "我有 200 个肝癌患者的 T1/T2/DWI/ADC 数据，想做生境分析然后建模预测复发，怎么开始？"
   - "Compare the ROC curves of my clinical and radiomics models with DeLong test."

### Option 2: Claude (Desktop / Code / Web)

1. Copy this entire `skills/` folder to your Claude skills directory:
   - **Claude Code (CLI)**: `~/.claude/skills/`
   - **Claude Desktop**: settings → Skills → import folder
   - **Claude.ai (Pro/Team)**: Skills tab in settings, upload as zip
2. Restart Claude or reload skills.
3. Ask Claude in natural language. It will auto-select the right skill.

### Option 3: Claude API

Use the Skills API endpoint with this folder uploaded as a skill bundle. See Anthropic API docs for the current endpoint.

---

## Prerequisites

These skills assume the HABIT package is already installed:

```bash
# In a conda env (Python 3.8 or 3.10)
conda create -n habit python=3.8 -y
conda activate habit
pip install -r requirements.txt
pip install -e .

# Verify
habit --version
```

If HABIT is not installed, the `habit-quickstart` skill will detect this and guide installation.

---

## Skill Design Notes

1. **Descriptions are bilingual-keyword friendly** — Chinese trigger phrases are embedded in English `description` fields so both Chinese and English users can activate skills naturally.
2. **All plot/figure outputs use English labels** — project-wide rule (see `README.md`). Documentation comments may be Chinese.
3. **Configs are referenced, not duplicated** — each skill points to `config_templates/config_*_annotated.yaml` for the full reference, and provides a minimal scaffold in its own `references/` for quick filling.
4. **No hidden state** — every skill is independent. Composition happens via the `habit-quickstart` router and explicit chained CLI calls.

---

## Updating / Extending

To add a new skill (e.g. for a new HABIT command):
1. Create `skills/<new-skill-name>/SKILL.md` with YAML frontmatter.
2. Write a single-sentence `description` that includes both English keywords and Chinese trigger phrases.
3. Add reference templates in `references/`.
4. Add a row to the table in this README.

To modify an existing skill, edit `SKILL.md` directly. No registration step needed — skills are discovered by folder structure.

---

## Citation

If you use HABIT (and these skills) in research, please cite the project per `README.md`. Issues and PRs welcome at https://github.com/lichao312214129/HABIT.
