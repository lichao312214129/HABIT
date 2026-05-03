# HABIT Workflow Decision Tree

A flow chart the agent walks the user through to pick the right downstream skill.

```mermaid
flowchart TD
  Start([User wants to use HABIT]) --> Q0{Is `habit` CLI<br/>installed?}
  Q0 -- No --> Install[Install: pip install -e .<br/>Refer user to README Step 1]
  Q0 -- Yes --> Q1{Is the data<br/>DICOM or NIfTI?}

  Q1 -- DICOM --> Pre1[habit-preprocess: dcm2nii mode]
  Q1 -- NIfTI --> Q2{Are images<br/>aligned & normalized?}

  Q2 -- No --> Pre2[habit-preprocess: full pipeline]
  Q2 -- Yes --> Q3{What is the<br/>research goal?}

  Pre1 --> Q2
  Pre2 --> Q3

  Q3 -- "Tumor sub-regions" --> Habitat[habit-habitat-analysis]
  Q3 -- "Classical radiomics only" --> Radiomics[habit-radiomics]
  Q3 -- "Test-retest QC" --> DicomTools[habit-dicom-tools]

  Habitat --> Extract[habit-feature-extraction]
  Radiomics --> ML
  Extract --> ML[habit-machine-learning]
  ML --> Q4{Multiple models<br/>to compare?}
  Q4 -- Yes --> Compare[habit-model-comparison]
  Q4 -- No --> Done([Done])
  Compare --> Done
```

## Skill picker — by user phrase

| If the user says... | Hand off to skill |
|---|---|
| "I have raw DICOM" / "DICOM 转 NIfTI" | `habit-preprocess` (dcm2nii mode) |
| "Images not aligned" / "需要配准" / "N4" | `habit-preprocess` |
| "Find tumor sub-regions" / "亚区" / "habitat" | `habit-habitat-analysis` |
| "Habitat maps + want features" / "MSI" / "ITH" | `habit-feature-extraction` |
| "Just radiomics, no habitat" / "传统影像组学" | `habit-radiomics` |
| "Train a classifier" / "建模" / "K-fold" | `habit-machine-learning` |
| "Compare models" / "ROC 对比" / "DeLong" | `habit-model-comparison` |
| "ICC" / "test-retest" / "Dice" / "merge CSV" | `habit-dicom-tools` |
| "Run the whole pipeline" / "全流程" | `habit-recipes` |
| "I got an error" / "报错" / "It failed" | `habit-troubleshoot` |

## When to use one_step vs two_step

```mermaid
flowchart LR
  X{Single cohort or<br/>multi-cohort study?} -- "Single, exploratory" --> One[one_step<br/>per-tumor optimal cluster #]
  X -- "Multi-cohort or want<br/>train/test generalization" --> Two[two_step<br/>population-level model]
```

- **one_step** → simpler, faster, no `pipeline.pkl`, habitat labels are local per-tumor
- **two_step** → produces a population-level habitat model that can be applied to new patients via `--mode predict`

If the user is unsure, default to **one_step**.
