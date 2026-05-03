# HABIT Recovery Playbook

A symptom→cause→action map covering the most common cross-cutting failures.
Use this when the agent recognizes a symptom but can't immediately tell
which step caused it.

## Symptom-driven index

| Observation | Likely cause | Action |
|---|---|---|
| `command not found: habit` | HABIT not installed | `pip install -e .` from repo root |
| `ImportError: No module named 'X'` | Missing dependency | `pip install X` (or check `requirements.txt`) |
| `FileNotFoundError: <path>` | Wrong path in config | Check `ls <path>` and edit YAML |
| `KeyError: '<field>'` in YAML loading | Required field missing | Compare against the matching annotated template |
| Silent crash, empty output dir | Process killed (OOM) | Reduce `processes`, smaller `n_clusters`, or smaller PyRadiomics filter set |
| Result CSV has no rows | Subject ID mismatch upstream | Run `inspect_feature_csv.py` on every input CSV |
| Plot files all 0 bytes | matplotlib backend issue | Try `MPLBACKEND=Agg habit ...` |
| Long runs that never finish | One subject is hanging | Run with `--debug`, watch the log for the last subject mentioned, exclude it temporarily |

## When to start over vs patch in place

### Start over (re-run the offending step) if:
- Output directory is partially written and inconsistent
- You changed any preprocessing parameter (it cascades to everything below)
- Cluster numbers / habitat counts changed (extracted features depend on these)

### Patch in place (no re-run) if:
- Plot title is wrong → just regenerate plots
- ML config changes don't affect feature columns → re-run only `habit model`
- Comparison config changes → re-run only `habit compare`

## Resumability

HABIT does NOT have automatic resume from interrupted runs. To "resume":
1. Manually identify which subjects already have output
2. Move the unprocessed subjects to a new `data_dir`
3. Re-run with the new `data_dir`
4. Merge the outputs by simply copying files into the original out_dir

## Log files to inspect

Every command writes a log to its `out_dir`:

| Command | Log file |
|---|---|
| `habit preprocess` | `<out_dir>/preprocess.log` |
| `habit get-habitat` | `<out_dir>/habitat_analysis.log` |
| `habit extract` | `<out_dir>/extract_features.log` (or `processing.log`) |
| `habit model` | `<output>/ml.log` |
| `habit cv` | `<output>/cv.log` |
| `habit compare` | `<output_dir>/model_comparison.log` |

Read the **last 50 lines** first; HABIT logs the failing subject and step
explicitly before crashing.

## Reproducibility checklist

When the user reports an issue, the agent should collect:
1. HABIT version: `habit --version`
2. Python version: `python --version`
3. OS: `python -c "import platform; print(platform.platform())"`
4. The exact command line used
5. The full config YAML (paste verbatim)
6. The last 50 lines of the relevant log
7. Output of `python skills/habit-quickstart/scripts/check_environment.py --json`

This enables proper diagnosis instead of guessing.

## Escalation

If the symptom doesn't match any entry in any error file AND the validator
script returns 0, ask the user to file a GitHub issue with the
reproducibility checklist above:

https://github.com/lichao312214129/HABIT/issues

Do NOT invent fixes; do NOT modify HABIT source code from a skill.
