# Registration module — backlog

Casual checklist for future work around `registration_preprocessor.py` and backends (`ants`, `simpleitk`, `elastix` CLI).

## Nice to have

- [ ] Evaluate **[SimpleElastix](https://simpleelastix.github.io/)** (`sitk.Elastix` / in-process elastix) as an optional backend: fewer temp files than CLI shell-out, uniform SimpleITK API; weigh install/binary story vs current `elastix`/`transformix` PATH workflow.
- [ ] Document elastix pitfalls when `use_mask=true` (ROI overlap / pyramid / initialization) next to CLI backend docs.
- [ ] Optional: retry or fallback path when Mattes reports “samples map outside moving image buffer” (e.g. diagnostic log, suggest no-mask or different init).

## Hygiene

- [ ] Keep `backend` naming consistent in configs and warnings (`elastic` deprecated alias, etc.).
- [ ] Align elastix parameter merging (`merge_elastix_parameter_file_text`) with versions users run (ITK 5.3 / elastix 5.x parameter renames).
