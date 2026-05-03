# Installing HABIT Skills into AI Tools

This document explains how to plug the `skills/` directory into the four most
common AI agent platforms. Pick the one matching your tool.

> **Universal rule**: keep this `skills/` directory inside your HABIT project
> root. The skills reference paths like `skills/habit-quickstart/scripts/check_environment.py`,
> so the working directory matters when commands are run.

## Quick comparison

| Tool | Auto-discovery? | Best for |
|---|---|---|
| Claude Code (CLI) | Yes (drop into `~/.claude/skills/`) | Power users, terminal-first |
| Claude Desktop | Manual import via Settings → Skills | Non-technical users |
| Cursor | Yes (when placed under repo root or `.cursor/skills/`) | Developers |
| Anthropic API | Manual upload as Skill bundle | Programmatic / automated agents |
| OpenCode / other | Project-root scan | Custom agents that follow Skills spec |

---

## Option 1 — Claude Code (CLI)

The simplest setup. Claude Code auto-loads any skill directory placed under
`~/.claude/skills/`.

### Steps

```bash
# Linux / macOS
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills" ~/.claude/skills/habit
# OR copy if you don't want a symlink:
# cp -r skills ~/.claude/skills/habit

# Windows PowerShell
mkdir $HOME\.claude\skills -Force
New-Item -ItemType Junction -Path "$HOME\.claude\skills\habit" -Target "$(Get-Location)\skills"
# OR copy:
# Copy-Item -Recurse skills "$HOME\.claude\skills\habit"
```

### Verify

Restart Claude Code, then in the CLI:

```
/skills list
```

You should see `habit-quickstart`, `habit-preprocess`, etc.

### Use

Just chat naturally:
```
> 我有 200 例肝癌 MRI（T1/T2/DWI/ADC），想做生境分析然后建模预测复发。
```

Claude will activate `habit-quickstart`, ask the required questions, then
walk through the recipe.

---

## Option 2 — Claude Desktop

### Steps

1. Open Claude Desktop.
2. Go to **Settings → Skills**.
3. Click **Import Folder** (or **Upload Skill** depending on version).
4. Select this `skills/` directory (the parent, not an individual skill).
5. Each child skill folder will be registered automatically.

### Verify

Open a new chat. The skills indicator at the bottom of the chat window
should show "10 skills available" (or similar count).

### Use

Same as Claude Code — just describe your goal in natural language.

---

## Option 3 — Cursor

Cursor auto-discovers skills placed under your workspace.

### Steps

Choice A — keep them where they are (this repo):
- The `skills/` directory at your HABIT repo root is already detected when
  you open the repo in Cursor. No extra steps.

Choice B — make them available across workspaces:
```bash
mkdir -p ~/.cursor/skills
ln -s "$(pwd)/skills" ~/.cursor/skills/habit
# Or on Windows:
# mklink /J %USERPROFILE%\.cursor\skills\habit %CD%\skills
```

### Verify

Open the Cursor command palette (Ctrl/Cmd+Shift+P) → "Skills" → list. The
HABIT skills should appear.

### Use

In the Cursor chat panel, ask:
```
我刚下载好 demo_data，帮我跑一遍完整的 demo
```

Cursor will activate `habit-recipes/references/recipe_demo_walkthrough.md`.

---

## Option 4 — Anthropic API

For programmatic agents using the Skills API endpoint.

### Steps

1. Zip the `skills/` directory:
   ```bash
   cd skills/..
   zip -r habit-skills.zip skills/
   ```
2. Upload via the Anthropic API:
   ```python
   from anthropic import Anthropic
   client = Anthropic(api_key="...")
   skill = client.skills.create(
       name="habit",
       file=open("habit-skills.zip", "rb"),
   )
   print(skill.id)
   ```
3. Reference the skill ID in your agent runs:
   ```python
   message = client.messages.create(
       model="claude-opus-4",
       skills=[skill.id],
       messages=[{"role": "user", "content": "我想做生境分析..."}],
   )
   ```

(Exact API surface may evolve; see the latest Anthropic docs.)

---

## Option 5 — OpenCode and other Skills-compatible agents

OpenCode and similar tools follow the Anthropic Skills spec. The same
folder layout works:

1. Place this `skills/` folder somewhere the tool scans (often project root
   or `~/.<tool>/skills/`).
2. Restart the tool.
3. Use natural language to trigger.

If your tool uses a different config format, you may need to declare the
skills folder explicitly. Example for a tool with a `config.toml`:

```toml
[skills]
paths = ["./skills", "~/.opencode/skills"]
```

---

## Project-level installation note

Some teams keep their AI tool config inside the repo so every team member
gets the same skills automatically. Two patterns:

### Per-repo `.cursor/skills/`
```bash
mkdir -p .cursor/skills
cp -r skills/* .cursor/skills/
```
Commit `.cursor/` to git. Every Cursor user opening the repo sees them.

### Per-repo `.claude/skills/`
Same idea for Claude-aware tools that respect a per-repo config.

---

## Troubleshooting installation

### Skills don't show up after install
- Check the tool restarted after install
- Verify the `description` field in each `SKILL.md` is present and parseable
- Check the directory structure is `<tool_skills_dir>/<skill_name>/SKILL.md` (not nested deeper)

### Wrong skill activates
- Refine the user prompt to mention more specific keywords (e.g. "habitat
  preprocess" instead of just "preprocess")
- The `description` fields are tuned for HABIT-specific phrases — pure
  generic terms like "preprocess" might match other unrelated skills

### Scripts fail with `python: command not found`
- The skills run scripts as `python skills/.../...py`. The user must be in
  a shell with Python on PATH and the `habit` conda env activated:
  ```bash
  conda activate habit
  python skills/habit-quickstart/scripts/check_environment.py
  ```

### Working directory issues
- Always run from the HABIT repo root so relative paths in the skills
  resolve correctly. If the user is elsewhere:
  ```bash
  cd <path_to_habit_repo>
  ```

---

## Updating

When this `skills/` folder is updated (e.g. after a HABIT release that adds
new CLI commands), users need to:
- **Symlinked installs** (Cursor option B, Claude Code option A) → nothing,
  symlinks track live changes
- **Copy-based installs** (Claude Desktop, Anthropic API) → re-import / re-upload

---

## Uninstall

Remove the symlink or directory you created:
```bash
rm ~/.claude/skills/habit              # or
rm ~/.cursor/skills/habit
```

Or via the tool's UI (Claude Desktop → Settings → Skills → Remove).
