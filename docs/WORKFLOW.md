# Team Workflow

## Purpose

This document defines how work is done in this onboarding repo.

## Core Loop

1. Receive a task.
2. Understand the objective and acceptance criteria.
3. Make a small change.
4. Verify with commands.
5. Inspect Git status and diff.
6. Commit with a clear message.
7. Push to GitHub.
8. Return evidence for review.

## Repo Rules

- Work from WSL Ubuntu.
- Keep the repo under `/home/****/ai-workspace`.
- Do not work from `/mnt/c`.
- Use small commits.
- Commit only meaningful files.
- Do not commit virtual environments, caches, checkpoints, logs, or secrets.
- Prefer source code over notebook-only work.
- Use notebooks for exploration only.
- Every task must have evidence.

## Local vs Cloud Rule

Use local WSL for:
- coding
- Git
- tests
- small scripts
- tiny experiments
- debugging
- smoke runs

Use local GPU for:
- small PyTorch CUDA checks
- tiny CNN/model experiments
- memory-aware debugging

Use cloud GPU for:
- larger training jobs
- transformer fine-tuning
- paper reproduction requiring meaningful compute
- sweeps
- experiments that exceed 4 GB VRAM

## Review Rule

A task is not complete because it “seems to work.”
A task is complete when evidence satisfies the acceptance criteria.

## AI-Assisted Development Rule

AI tools may be used for explanation, debugging, draft code, and review.

Before committing AI-assisted changes, always run:

```bash
git status
git diff
uv run ruff format .
uv run ruff check .
uv run python -m pytest -q
git status
```

The engineer must understand and explain every committed change.
