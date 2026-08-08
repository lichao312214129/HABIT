# HABIT Documentation

Build: `cd docs && make html` → `docs/build/html/index.html`

Entry point: `source/index.rst` (English).

## GitHub Pages

Published site: https://lichao312214129.github.io/HABIT/

Deploy only via a **temporary git worktree** on branch `gh-pages` (never
`git switch gh-pages` inside the HABIT working tree). Copy
`docs/build/html/*` into the worktree; copy `docs/gh-pages.gitignore` to the
worktree root as `.gitignore`; remove the worktree when done. Do not commit
source trees (`habit/`, `docs/`, `tests/`, …) to `gh-pages`. Full agent
checklist: `.cursor/rules/docs-gh-pages-deploy.mdc`.

Demo pack download links / extract codes live in `source/conf.py`
(`NETDISK_SHARES`):

| Key | File | Extract to | Needed for |
|-----|------|------------|------------|
| `demo_data` | `preprocessed.zip` | `demo_data/preprocessed/{images,masks}/` | Habitat / preprocess / features |
| `ml_data` | `ml_data.zip` | `demo_data/ml_data/` | `habit model` / `habit cv` |

User-facing unpack steps: `tutorial/quickstart.rst`,
`how_to/before_you_start.rst`, `how_to/train_model.rst`.

Demo **YAML** single source of truth: repository-root `config/` only.
Editable installs read that tree live. Wheels bake a copy under
`habit/resources/demo_config/` via `setup.py` `build_py` /
`scripts/sync_demo_config.py` (generated files are gitignored — do not
hand-edit them). Users materialize with `habit copy-demo-config` /
`habit.copy_demo_config`. Never package `demo_data/` into the wheel.

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
