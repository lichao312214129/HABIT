---
name: habit-dicom-tools
description: Auxiliary HABIT utilities — DICOM tag inspection, CSV merging, ICC reproducibility analysis, test-retest habitat re-mapping, and Dice coefficient computation. Use for data preparation, QC, and reproducibility studies. Triggers on "DICOM 信息", "查看 DICOM 标签", "合并 CSV", "ICC 分析", "test-retest", "重测一致性", "Dice 系数", "merge csv", "intraclass correlation".
---

# HABIT Auxiliary Tools

Five smaller HABIT CLI utilities. Each is standalone — pick the right one based
on user need.

## Required Information (depends on tool)

| Tool | Critical inputs |
|---|---|
| `dicom-info` | input directory, optional list of tags |
| `merge-csv` | list of input files, index column name(s) |
| `icc` | file groups (matched test/retest CSVs), output JSON path |
| `retest` | test/retest habitat tables, similarity method, scan-2 .nrrd dir |
| `dice` | two batch directories or YAML configs, output CSV |

## Tool decision table

| User goal | Command | Section |
|---|---|---|
| Inspect / extract DICOM tags | `habit dicom-info` | [§1](#1-dicom-info) |
| Merge multiple CSV/Excel files | `habit merge-csv` | [§2](#2-merge-csv) |
| ICC analysis (test-retest features) | `habit icc` | [§3](#3-icc-analysis) |
| Map habitats between test-retest scans | `habit retest` | [§4](#4-test-retest-habitat-mapping) |
| Dice coefficient between two ROI batches | `habit dice` | [§5](#5-dice-coefficient) |

---

## 1. DICOM info

Extract DICOM metadata (tags) from a directory tree of DICOM files. Useful
for cohort QC and metadata collection.

### CLI

```bash
# List available tags from a sample
habit dicom-info -i <dicom_dir> --list-tags --num-samples 3

# Extract specific tags to CSV
habit dicom-info -i <dicom_dir> \
  -t "PatientName,StudyDate,Modality,SeriesDescription,SliceThickness" \
  -o dicom_info.csv \
  --one-file-per-folder \
  --max-depth 3
```

### Key options

- `--one-file-per-folder` — much faster when each folder = one series
- `--max-depth 3` — typical DICOM layout: patient/study/series (depth 3)
- `--group-by-series` (default true) — read one file per SeriesInstanceUID
- `--include-no-extension` — for devices that produce DICOMs without `.dcm`
- `--num-workers 8` — parallel scanning

Common tags reference: `references/dicom_tags_cheatsheet.md`.

---

## 2. Merge CSV

Horizontally merge multiple CSV/Excel files on a common index column. Faster
and more error-tolerant than manual pandas merging.

### CLI

```bash
# Same index column for all files
habit merge-csv file1.csv file2.csv file3.csv -o merged.csv --index-col PatientID

# Different index columns per file
habit merge-csv a.csv b.csv -o merged.csv --index-col "PatientID,subject_id"

# Outer join (keep all rows)
habit merge-csv f1.csv f2.csv -o merged.csv --join outer
```

### Key options

- `--index-col` — single name (all files) or comma-separated (one per file)
- `--join inner` (default) | `outer`
- `--separator ";"` for European CSVs
- `--encoding gbk` for Chinese Windows CSVs

### Common use case

Merge clinical + radiomics + habitat features for ML:

```bash
habit merge-csv clinical.csv radiomics.csv habitat_features.csv \
  -o ml_input.csv --index-col PatientID --join inner
```

Full guide: `docs/source/app_merge_csv_zh.rst`.

---

## 3. ICC analysis

Compute Intraclass Correlation Coefficients to assess feature reproducibility
across two or more measurements (test-retest, inter-rater, multi-scanner).

### CLI

```bash
habit icc --config <config_icc_analysis.yaml>
```

### Config

Use `config_templates/skill_scaffolds/auxiliary_icc_minimal.yaml` as scaffold. Full template:
`config_templates/config_icc_analysis_annotated.yaml`.

```yaml
input:
  type: files
  file_groups:
    - [./scan1_radiomics.csv, ./scan2_radiomics.csv]
    - [./scan1_msi.csv, ./scan2_msi.csv]
output:
  path: ./results/icc_results.json
processes: 6
```

### Output

`icc_results.json` contains per-feature ICC values, 95% CIs, and reliability
classification:
- `< 0.40` → Poor
- `0.40-0.59` → Fair
- `0.60-0.74` → Good
- `0.75-1.00` → Excellent

ICC type: **ICC(3,1)** (two-way mixed effects, absolute agreement, single rater).

### Common use case

ICC results feed into ML feature selection via the `icc` selector — tell user to:
1. Run `habit icc` to get JSON
2. Reference it in `habit model` config:
   ```yaml
   feature_selection_methods:
     - method: icc
       params: {icc_results: ./icc_results.json, threshold: 0.8}
   ```

---

## 4. Test-retest habitat mapping

Habitat labels are arbitrary integers. Across two scans, "habitat 2" in scan1
may correspond to "habitat 4" in scan2. This tool finds the optimal mapping.

### CLI

```bash
habit retest --config <config_test_retest.yaml>
```

### Config

Use `config_templates/skill_scaffolds/auxiliary_test_retest_minimal.yaml` as scaffold.

```yaml
out_dir: ./results/test_retest
test_habitat_table: ./results/scan1/habitats.csv
retest_habitat_table: ./results/scan2/habitats.csv
similarity_method: pearson         # pearson | spearman | kendall | euclidean | cosine | manhattan
input_dir: ./results/scan2/habitat_maps     # directory of retest .nrrd files
output_dir: ./results/test_retest/remapped
processes: 4
```

### Method choice

- **Continuous feature values** → `pearson` or `spearman`
- **Probability maps** → `cosine` or `euclidean`

### Output

- `remapped_habitats.csv` — habitat table with consistent labels
- `mapping_quality.csv` — per-subject mapping reliability score
- Remapped `.nrrd` files in `output_dir`

Full guide: `docs/source/app_habitat_test_retest_zh.rst`.

---

## 5. Dice coefficient

Quantify mask agreement between two batches (e.g. two raters).

### CLI

```bash
habit dice \
  --input1 <path_to_batch1> \
  --input2 <path_to_batch2> \
  --output dice_results.csv \
  --mask-keyword masks \
  --label-id 1
```

### Inputs

- `--input1` / `--input2` — directories OR YAML config files listing mask paths
- `--mask-keyword` — folder keyword to find mask files
- `--label-id` — which label value to compute Dice for (default 1)

### Interpretation

- `> 0.90` — Excellent agreement
- `0.80-0.90` — Good
- `0.70-0.80` — Moderate
- `< 0.70` — Poor; ROI definitions may differ

---

## Cross-tool workflow example

A typical reproducibility study:

```bash
# 1. Check raw DICOM metadata
habit dicom-info -i ./raw_dicoms --one-file-per-folder -o cohort_qc.csv

# 2. After preprocessing + habitat on two scans, evaluate mask agreement
habit dice --input1 ./scan1/masks --input2 ./scan2/masks -o dice_masks.csv

# 3. Map habitat labels consistently across scans
habit retest --config config_test_retest.yaml

# 4. Compute ICC on extracted features
habit icc --config config_icc.yaml

# 5. Use ICC in ML feature selection (filter unstable features)
habit model --config config_ml.yaml

# 6. Merge final feature sets for modeling
habit merge-csv clinical.csv radiomics.csv habitat.csv -o ml_input.csv --index-col PatientID
```

For the full test-retest recipe, see `habit-recipes/references/recipe_test_retest.md`.

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/auxiliary_icc_minimal.yaml` | ICC scaffold |
| `config_templates/skill_scaffolds/auxiliary_test_retest_minimal.yaml` | habitat re-mapping scaffold |
| `references/dicom_tags_cheatsheet.md` | common DICOM tags |
