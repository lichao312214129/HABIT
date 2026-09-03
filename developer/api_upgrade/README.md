# HABIT v2 API upgrade

This directory contains only the active design record for the v2.0.0 breaking
migration from `habit.domain` to physical, capability-named public packages.

| File | Role |
|---|---|
| [06_v1_api_first_architecture.md](06_v1_api_first_architecture.md) | Retained API-first architecture principles that remain applicable to v2; not an execution plan. |
| [08_naming_decisions.md](08_naming_decisions.md) | Concise, authoritative naming and constructor-only parameter-contract decisions. |
| [09_capability_namespaces.md](09_capability_namespaces.md) | The sole detailed v2 execution plan: package moves, public imports, artifacts, plugins, documentation, and test definition of done. |

Read 08 for decisions, then execute 09. The v2 migration deliberately removes
`habit.domain`, flat component exports, component `*Params` Pydantic models,
and Registry `params_model` schemas. Components expose their parameter
contracts through typed, validated constructors; registries map names to
classes and create them; `inspect.signature`, annotations, and constructor
docstrings provide catalog and documentation metadata.

The historical v0/v1 facade, CI, roadmap, usage, rearchitecture, and cloud
status documents were removed after their still-applicable architecture and
acceptance constraints were consolidated into 06, 08, and 09. This directory
does not preserve compatibility requirements superseded by v2.
