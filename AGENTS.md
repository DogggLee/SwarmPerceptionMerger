# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python prototype for multi-UAV perception fusion. Core fusion logic lives in `merger/`, with the main entry point in `merger/perception_merger.py`. Shared data models and enums are in `utils/data_utils.py`, and JSON transport helpers are in `utils/json_utils.py`. Situation state and simulation-side code live in `simulator/`, mainly `simulator/global_info.py`. Design notes and IO contracts are documented in `readme.md` and `IO.md`; keep those files aligned with code changes.

## Details of PerceptionMerger
The merge rules is based on `agent_docs/merge_rule.md`


## Build, Test, and Development Commands
There is no packaged build system yet, so use direct Python commands from the repository root:

- `python -m compileall merger simulator utils` checks for syntax errors.
- `python -m pytest` runs tests once a `tests/` directory is added.
- `python -i merger/perception_merger.py` is a quick way to inspect module imports during development.

If you add dependencies, introduce a standard manifest such as `pyproject.toml` rather than ad hoc setup steps.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and UTF-8 source files. Use `snake_case` for functions, methods, and module names; use `PascalCase` for classes such as `PerceptionMerger` and `GlobalInfo`; keep enum members in `UPPER_CASE`. Prefer explicit type hints on public methods and concise docstrings for fusion, alignment, and update logic. Keep comments focused on non-obvious behavior, especially around timestamps, coordinate frames, and matching rules.

## Testing Guidelines
Place tests under `tests/` with names like `test_perception_merger.py`. Mirror the source layout when practical, for example `tests/test_global_info.py` for `simulator/global_info.py`. Cover association, timestamp alignment, ID assignment, and stale-track handling. For new matching logic, include at least one deterministic fixture with small synthetic observations.

## Commit & Pull Request Guidelines
No Git history is available in this workspace, so use short imperative commit subjects such as `Add track history lookup` or `Implement object timestamp alignment`. Keep commits scoped to one concern. Pull requests should summarize the scenario, list changed modules, note any input/output contract updates, and include sample payloads or screenshots if a JSON format or simulation result changes.

## Documentation & Configuration
Update `readme.md` or `IO.md` whenever sensor enums, JSON fields, or fusion assumptions change. Do not hard-code environment-specific URLs or vehicle IDs in reusable modules.
