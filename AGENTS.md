# AGENTS.md

## Purpose

This repository is a Deep Learning Engineering / Applied Research Engineering onboarding repo.

It tracks a professional progression from environment setup to PyTorch engineering, computer vision, experiment tracking, checkpointing, and eventually modern deep learning systems.

## Source of Truth

This public GitHub repository is the source of truth for future assistant sessions.

Before assigning new work, inspect the repo files directly. Do not rely only on pasted handoffs.

## How to Understand This Repo

Start by reading:

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/TASKS.md`
4. `docs/DECISIONS.md`
5. `docs/TESTING.md`
6. `pyproject.toml`
7. `.gitignore`
8. `src/dl_onboarding/`
9. `tests/`
10. `scripts/quality_check.sh`

If docs and source disagree, report the mismatch and repair repo state before assigning new feature work.

## Current Workflow

Before changing code:

    ./scripts/quality_check.sh --torch

After changing code:

    ./scripts/quality_check.sh --torch
    git status
    git diff --stat

For docs-only tasks:

    ./scripts/quality_check.sh

## Current Project State

The current completed milestone is the PyTorch engineering foundation through:

- tensors / devices / autograd
- manual training loop
- Dataset / DataLoader
- `nn.Module`
- `torch.optim`
- train/validation metrics
- checkpointing
- TensorBoard logging
- FashionMNIST MLP classification baseline
- FashionMNIST CNN baseline

The next expected task should be read from `docs/TASKS.md`.

## Rules for AI Assistants

- Do not restart the curriculum.
- Do not assume missing file content.
- Inspect the repo files before proposing changes.
- Explain the design first.
- Define the test plan before implementation.
- Provide terminal-safe commands.
- Explain new imports, classes, functions, and PyTorch concepts.
- Run quality gates.
- Avoid committing generated artifacts.

Generated data/logs/checkpoints should stay ignored:

- `data/`
- `runs/`
- `outputs/`
- `.venv/`
- caches
