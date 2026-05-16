# Deep Learning Engineering Onboarding

This repository tracks a serious long-term onboarding workflow into Deep Learning Engineering and Applied Research Engineering.

It is not a tutorial scratchpad. It is a cumulative engineering repo for building PyTorch systems, testing habits, experiment discipline, debugging skill, and eventually applied research artifacts.

## Current Status

Current phase:

    Phase 1 complete pending final recap acceptance.

Latest completed technical milestone:

    FashionMNIST supervised classification stack.

The repo now includes:

- tensor/device/autograd foundations
- manual training loops
- Dataset/DataLoader foundations
- `nn.Module` training
- optimizer-based training
- train/evaluation metrics
- checkpointing basics
- TensorBoard scalar logging
- FashionMNIST MLP baseline
- FashionMNIST CNN baseline
- controlled MLP vs CNN comparison
- classification experiment checkpointing
- train/validation/test split
- confusion matrix and per-class error analysis

## Main Commands

Run the standard quality gate:

    ./scripts/quality_check.sh

Run the quality gate with PyTorch CUDA verification:

    ./scripts/quality_check.sh --torch

Run the latest FashionMNIST error analysis lab:

    uv run python scripts/fashion_error_analysis_lab.py

Inspect TensorBoard logs after experiment scripts:

    uv run tensorboard --logdir runs/fashion_experiments

## Important Reports

Phase 1 recap:

    docs/reports/phase1_classification_stack.md

## Machine Assumption

This repo is developed on:

- Windows 10
- WSL2 Ubuntu
- Python 3.12 via `uv`
- NVIDIA GTX 1650, 4 GB VRAM

Local machine policy:

- good for development, tests, debugging, and small smoke runs
- not appropriate for large-scale training or modern foundation-model work

## Repo Rules

- Work inside the Linux filesystem, not `/mnt/c`.
- Use `uv` for Python commands.
- Use small, reviewable commits.
- Run quality gates before committing.
- Do not commit generated artifacts.
- Keep `data/`, `runs/`, `outputs/`, checkpoints, logs, caches, and `.venv/` out of Git.
- Treat the public GitHub repo as the source of truth for future sessions.
