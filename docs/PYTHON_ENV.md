# Python Environment Strategy

## Tool

This repo uses `uv` for Python environment and dependency management.

## Python Version

The project targets Python 3.12.

The expected version is recorded in:

```text
.python-version
```

The supported range is recorded in:

```text
pyproject.toml
```

## Environment Location

The local virtual environment is:

```text
.venv/
```

This folder is intentionally ignored by Git.

## Core Commands

Create or sync the environment:

```bash
uv sync
```

Run Python through the project environment:

```bash
uv run python --version
```

Run a Python command through the project environment:

```bash
uv run python -c "import sys; print(sys.executable)"
```

Run tests:

```bash
uv run python -m pytest -q
```

## Team Rule

Do not install project dependencies globally with pip.

Use project-level dependency management so the environment can be recreated on another machine.

## Local vs Cloud

Use this local WSL environment for:

- Python basics
- tests
- small scripts
- PyTorch smoke checks later
- tiny experiments

Use cloud GPU only when local CPU/GPU is insufficient.
