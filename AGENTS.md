# Repository Guidelines

## Project Structure & Module Organization
- `elliot/`: core package (notable submodules: `dataset/`, `recommender/`, `evaluation/`, `prefiltering/`, `splitter/`, `negative_sampling/`, `hyperoptimization/`, `utils/`, `result_handler/`, `namespace/`).
- `config_files/`: runnable YAML configs (e.g., `sample_hello_world.yml`, `basic_configuration.yml`).
- `data/`: datasets (gitignored). Place local files here.
- `results/`: outputs, logs, and model artifacts (gitignored; configured via YAML).
- `docs/`: Sphinx documentation.
- `external/`: optional external models referenced by config.
- `sample_*.py`, `start_experiments.py`: examples and CLI entry points.

## Build, Test, and Development Commands
- Create env + install: `python -m venv .venv && source .venv/bin/activate && pip install -e .` (or `pip install -r requirements.txt`).
- Run an experiment: `python start_experiments.py --config sample_hello_world`.
- Programmatic run: `python -c "from elliot.run import run_experiment as r; r('config_files/basic_configuration.yml')"`.
- Build docs: `make -C docs html` (open `docs/_build/html/index.html`).
- Package build (optional): `python setup.py sdist bdist_wheel`.

## Coding Style & Naming Conventions
- Python 3.6+; follow PEP 8, 4-space indentation, <= 100 cols where practical.
- Names: modules/files `snake_case`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_CASE`.
- Docstrings: Google or NumPy style (Sphinx autodoc enabled in `docs/`).
- Config: YAML keys `lower_snake_case`; files end with `.yml`.

## Testing Guidelines
- No bundled test suite. Use small configs for smoke tests (e.g., `config_files/sample_hello_world.yml`).
- `config_files/test_config.yml` contains a fuller example; adapt local paths before use.
- If adding tests, place them under `tests/` named `test_*.py` and target critical logic (parsers, metrics, splitters). Prefer `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "Fix dataset path", "Rename baseline config"). Group related changes.
- PRs: clear description, rationale, reproduction steps, and screenshots/log snippets when relevant. Link issues. Update docs/config examples when behavior changes. Do not commit data/results.

## Security & Configuration Tips
- Keep datasets in `data/` and outputs in `results/` (both gitignored).
- Avoid absolute, machine-specific paths in YAML; prefer relative paths within the repo.
- External models: place code under `external/` and reference via `external_models_path` in configs.

