# Required Information Per Skill

The agent MUST collect the answers below before generating a config or running
the corresponding `habit ...` command. If any required field is missing, STOP
and ask the user explicitly. Do not invent paths or modality names.

## habit-preprocess

| Field | Why | Example |
|---|---|---|
| `data_dir` | input directory | `./data/raw_images` |
| `out_dir` | output directory | `./data/preprocessed_images` |
| Modality names | drives `images:` lists everywhere | `T1, T2, DWI, ADC` |
| Modality kind | MRI vs CT (decides whether to use N4) | "T1/T2/FLAIR are MRI" |
| Multi-modal? | decides whether `registration` is needed | "yes — 4 modalities" |
| Registration target | `fixed_image` | "T2" |
| (DICOM only) `dcm2niix_path` | absolute path to dcm2niix binary | `./software/dcm2niix.exe` |

## habit-habitat-analysis

| Field | Why |
|---|---|
| `data_dir` (preprocessed) and `out_dir` | I/O |
| Modalities to use for clustering | drives `voxel_level.method` |
| `clustering_mode`: one_step / two_step | core algorithmic choice |
| (kinetic only) timestamps Excel path | required for DCE kinetic features |
| (voxel_radiomics only) `params_file` | required PyRadiomics params |
| (two_step) `fixed_n_clusters` for habitat | typical: 3-5 |

## habit-feature-extraction

| Field | Why |
|---|---|
| `raw_img_folder` | location of preprocessed images + masks |
| `habitats_map_folder` | location of `*_habitats*.nrrd` files |
| `out_dir` | output directory |
| `feature_types` selection | what to extract |
| `habitat_pattern` | depends on whether one_step or two_step output |
| `params_file_of_*` | only if extracting radiomics-based features |

## habit-machine-learning

| Field | Why |
|---|---|
| Path(s) to feature CSV(s) | input |
| `subject_id_col` | how to merge multiple CSVs |
| `label_col` | target; must be binary 0/1 |
| `output` directory | where results land |
| Split strategy: stratified vs custom vs k-fold | decision |
| (custom only) `train_ids_file` and `test_ids_file` | required |
| (k-fold) `n_splits`, `stratified` | typical 5 or 10 |
| Models to train | at least one |

## habit-model-comparison

| Field | Why |
|---|---|
| `output_dir` | where comparison plots land |
| For EACH model: `path`, `name`, `subject_id_col`, `label_col`, `prob_col`, `pred_col`, `split_col` | comparison config requires explicit column mapping per file |

## habit-radiomics

| Field | Why |
|---|---|
| `paths.params_file` | PyRadiomics parameter YAML |
| `paths.images_folder` | preprocessed images + masks |
| `paths.out_dir` | output |
| `processing.process_image_types` | modality folder names |

## habit-dicom-tools

Depends on which sub-tool. Always confirm:
- For `dicom-info`: input directory + which tags
- For `merge-csv`: list of input files + index column
- For `icc`: file groups (matched test/retest CSVs)
- For `retest`: test/retest habitat tables + similarity method
- For `dice`: two batch directories + label ID

## Universal stop conditions

The agent MUST stop and ask if any of these are missing:
- Output directory not specified
- Modality names not confirmed
- ML label column not identified
- Custom split chosen but no ID files provided
- Predict mode chosen but no `.pkl` model path supplied
- DICOM conversion chosen but no `dcm2niix.exe` path supplied
