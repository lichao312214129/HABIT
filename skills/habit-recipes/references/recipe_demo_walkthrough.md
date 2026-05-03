# Recipe — Demo Dataset Walkthrough

For first-time users who just installed HABIT and want to see a real run
end-to-end before touching their own data. Uses the bundled `demo_data/`
dataset (download from the project README link).

## Prerequisite

User must have downloaded `demo_data.rar` from the link in `README.md` and
extracted it into the project root, so the layout looks like:

```
HABIT/
├── demo_data/
│   ├── preprocessed/      (already preprocessed, ready)
│   ├── ml_data/           (clinical CSV for ML)
│   ├── config_*.yaml      (pre-made configs for every step)
│   └── ...
```

If `demo_data/` is missing, redirect user to README Step 2 for the download
link.

## CLI sequence — verbatim

The demo configs are pre-baked; the agent doesn't need to generate any YAML.

```bash
# 0) Sanity check
python skills/habit-quickstart/scripts/check_environment.py

# 1) Preprocessing (~2-5 min)
habit preprocess --config demo_data/config_preprocessing.yaml
python skills/habit-preprocess/scripts/validate_preprocess_output.py demo_data/preprocessed

# 2) Habitat analysis (two-step, ~5-10 min)
habit get-habitat --config demo_data/config_habitat_two_step.yaml
python skills/habit-habitat-analysis/scripts/validate_habitat_output.py demo_data/results/habitat_two_step --two-step

# 3) Feature extraction (~3-8 min)
habit extract --config demo_data/config_extract_features.yaml

# 4) Train two ML models for comparison (radiomics vs clinical, ~2-5 min each)
habit model --config demo_data/config_machine_learning_radiomics.yaml --mode train
habit model --config demo_data/config_machine_learning_clinical.yaml --mode train

# 5) Multi-model comparison (~1-3 min)
habit compare --config demo_data/config_model_comparison.yaml
```

Total wall time: ~15–30 minutes on a modern laptop.

## Where to look at the results

After step 5, the user should open these to verify success:

| File | What it shows |
|---|---|
| `demo_data/results/habitat_two_step/subj001_habitats.nrrd` | open in ITK-SNAP, overlay on `demo_data/preprocessed/processed_images/images/subj001/delay2/delay2.nii.gz` |
| `demo_data/ml_data/radiomics/roc_curve.pdf` | per-model ROC |
| `demo_data/ml_data/model_comparison/roc_curves.pdf` | radiomics vs clinical comparison |
| `demo_data/ml_data/model_comparison/delong_results.json` | DeLong p-value |

## Optional: alternative habitat strategies

The demo also ships configs for one_step and direct_pooling — useful for
comparing habitat strategies on the same data:

```bash
habit get-habitat --config demo_data/config_habitat_one_step.yaml
habit get-habitat --config demo_data/config_habitat_direct_pooling.yaml
```

Outputs go to `demo_data/results/habitat_one_step/` and
`demo_data/results/habitat_direct_pooling/` respectively.

## When something goes wrong

The demo paths assume the user is running from the HABIT repo root. If the
user gets "file not found" errors, ask them to confirm:

```bash
pwd                                    # should end with .../HABIT
ls demo_data/config_preprocessing.yaml  # should exist
```

For other errors, hand off to `habit-troubleshoot`.

## Talking points for the user after the demo

After the demo runs successfully, tell the user:

1. The pipeline is now proven on their machine; their own data should work.
2. To adapt to their own data, the only changes are:
   - `data_dir` and `out_dir` in each config
   - modality folder names in `images:` lists
   - clinical CSV path / column names in the ML config
3. Recommend they read `recipe_mri_habitat_full.md` next for a from-scratch
   workflow on their own data.
