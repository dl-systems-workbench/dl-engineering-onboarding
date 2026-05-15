# AGENTS.md

## Purpose

This repository is a Deep Learning Engineering / Applied Research Engineering onboarding repo.

It tracks a professional progression from environment setup to PyTorch engineering, computer vision, experiment tracking, checkpointing, and eventually modern deep learning systems.

## How to Understand This Repo

Start by reading:

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/TASKS.md`
4. `docs/DECISIONS.md`
5. `docs/TESTING.md`
6. `src/dl_onboarding/`
7. `tests/`
8. `scripts/quality_check.sh`

## Current Workflow

Before changing code:

    ./scripts/quality_check.sh --torch

After changing code:

    ./scripts/quality_check.sh --torch
    git status
    git diff --stat

## Current Project State

The current completed milestone is the PyTorch engineering foundation through:

- tensors / devices / autograd
- manual training loop
- Dataset / DataLoader
- nn.Module
- torch.optim
- train/validation metrics
- checkpointing
- TensorBoard logging
- FashionMNIST MLP classification baseline

The next expected task is:

T1.10 — First CNN Baseline

## Rules for AI Assistants

Do not restart the curriculum.

Do not assume missing file content. Inspect the repo files.

When proposing changes:

- explain the design first
- define the test plan before implementation
- provide terminal-safe commands
- explain new imports/classes/functions
- run quality gates
- avoid committing generated artifacts

Generated data/logs/checkpoints should stay ignored:

- data/
- runs/
- outputs/
- .venv/
- caches
