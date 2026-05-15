# Deep Learning Engineering Onboarding

This repository tracks my professional onboarding into Deep Learning Engineering and Applied Research Engineering.

## Purpose

The goal of this repo is to build real engineering habits while learning PyTorch, ML project structure, testing, debugging, experiment tracking, computer vision, transformers, and paper reproduction.

This is a cumulative training repo, not a collection of isolated tutorials.

## Current Phase

Phase 1 — PyTorch Engineering Foundations.

Current accepted milestone:

- Tensor, device, and autograd basics
- Manual training loop
- Dataset and DataLoader foundations
- `nn.Module`
- Optimizer-based training
- Train/evaluation metrics
- Checkpointing basics
- TensorBoard scalar logging
- FashionMNIST MLP classification baseline
- FashionMNIST CNN baseline

## Machine

- OS: Windows 10 + WSL2 Ubuntu
- CPU: Intel i7-9750H
- RAM: 16 GB
- GPU: NVIDIA GTX 1650, 4 GB VRAM
- Development environment: WSL2 Ubuntu + VS Code

## Team Rules

- Work inside the Linux filesystem, not `/mnt/c`.
- Use Git for every meaningful change.
- Use small, reviewable commits.
- Verify work with commands and evidence.
- Prefer reproducible environments over global installs.
- Run small experiments locally; use cloud only when justified.
- Keep generated data, logs, checkpoints, and outputs out of Git.

## Current Status

The repo has completed its first realistic FashionMNIST classification baselines:

- `FashionMLP`
- `FashionCNN`

Next work continues the supervised computer vision training stack with better experiment comparison, logging, checkpointing, and evaluation discipline.
