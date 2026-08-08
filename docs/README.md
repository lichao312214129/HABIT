# HABIT Documentation

Build: `cd docs && make html` → `docs/build/html/index.html`

Entry point: `source/index.rst` (English).

Demo pack download links / extract codes live in `source/conf.py`
(`NETDISK_SHARES`):

| Key | File | Extract to | Needed for |
|-----|------|------------|------------|
| `demo_data` | `preprocessed.zip` | `demo_data/preprocessed/{images,masks}/` | Habitat / preprocess / features |
| `ml_data` | `ml_data.zip` | `demo_data/ml_data/` | `habit model` / `habit cv` |

User-facing unpack steps: `tutorial/quickstart.rst`,
`how_to/before_you_start.rst`, `how_to/train_model.rst`.

## Structure

| Section | Path |
|---------|------|
| Tutorial | `tutorial/` |
| How-to | `how_to/` |
| Configuration | `configuration/` |
| Feature reference | `reference/features/` |
| CLI / auxiliary | `reference/cli.rst`, `reference/auxiliary.rst` |
| API (autodoc) | `api/` |
| Developer | `development/`, `customization/` |
