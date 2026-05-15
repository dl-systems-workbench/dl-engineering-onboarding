# Handoff State

## User

Junior engineer onboarding into Deep Learning Engineering / Applied Research Engineering.

The long-term goal is to become a strong Deep Learning Engineer / Applied Research Engineer capable of building, debugging, evaluating, reproducing, and explaining modern ML systems.

## Source of Truth

The repository is public and should be inspected directly before continuing work.

Repository:

    https://github.com/dl-systems-workbench/dl-engineering-onboarding

Future assistants should inspect:

1. `AGENTS.md`
2. `README.md`
3. `docs/TASKS.md`
4. `docs/HANDOFF.md`
5. `docs/DECISIONS.md`
6. `docs/TESTING.md`
7. `pyproject.toml`
8. `.gitignore`
9. `src/dl_onboarding/`
10. `tests/`
11. `scripts/quality_check.sh`

If docs and source disagree, repair repo state before assigning new feature work.

## Machine

- Windows 10 Pro
- WSL2 Ubuntu 24.04.3 LTS
- Intel i7-9750H
- 16 GB RAM
- NVIDIA GTX 1650, 4 GB VRAM
- VS Code with WSL workflow

## Repo

- Local path: `~/ai-workspace/dl-engineering-onboarding`
- Branch: `main`
- Python target: 3.12
- Environment manager: `uv`
- Package layout: `src/`
- Test runner: `pytest`
- Formatter/linter: `ruff`

## Current Phase

Phase 1 — PyTorch Engineering Foundations.

## Current Accepted Stack

The user has built:

1. tensor/device/autograd basics
2. manual training loop
3. testing strategy
4. Dataset/DataLoader
5. `nn.Module`
6. `torch.optim.SGD`
7. train/validation split
8. regression metrics
9. checkpoint save/load
10. TensorBoard scalar logging
11. FashionMNIST MLP classification baseline
12. FashionMNIST CNN baseline

## Important Source Files

- `src/dl_onboarding/tensor_lab.py`
- `src/dl_onboarding/manual_training.py`
- `src/dl_onboarding/data_loading.py`
- `src/dl_onboarding/module_training.py`
- `src/dl_onboarding/evaluation.py`
- `src/dl_onboarding/checkpointing.py`
- `src/dl_onboarding/experiment_logging.py`
- `src/dl_onboarding/fashion_mnist.py`
- `src/dl_onboarding/__init__.py`

## Current FashionMNIST Components

`src/dl_onboarding/fashion_mnist.py` includes:

- `fashion_class_names`
- `FashionMLP`
- `FashionCNN`
- `count_trainable_parameters`
- `make_fashion_mnist_dataloaders`
- `classification_accuracy`
- `evaluate_classifier`
- `make_tiny_classification_dataloader`
- `train_fashion_mlp`
- `train_fashion_cnn`

## Important Tests

- `tests/test_system_info.py`
- `tests/test_tensor_lab.py`
- `tests/test_manual_training.py`
- `tests/test_data_loading.py`
- `tests/test_module_training.py`
- `tests/test_optimizer_training.py`
- `tests/test_evaluation.py`
- `tests/test_checkpointing.py`
- `tests/test_experiment_logging.py`
- `tests/test_fashion_mnist.py`

## Important Scripts

- `scripts/quality_check.sh`
- `scripts/verify_torch.py`
- `scripts/tensor_autograd_lab.py`
- `scripts/manual_training_lab.py`
- `scripts/dataloader_lab.py`
- `scripts/module_training_lab.py`
- `scripts/optimizer_training_lab.py`
- `scripts/evaluation_lab.py`
- `scripts/checkpointing_lab.py`
- `scripts/tensorboard_lab.py`
- `scripts/fashion_mnist_lab.py`
- `scripts/fashion_cnn_lab.py`

## Current Quality Gate

Use:

    ./scripts/quality_check.sh

For PyTorch/GPU-related changes:

    ./scripts/quality_check.sh --torch

For safe Ruff auto-fix:

    ./scripts/quality_check.sh --fix

## Current Task

T1.10R — Public Repo State Alignment.

## Next Expected Work

After T1.10R is accepted, continue to:

    T1.11 — Controlled MLP vs CNN Experiment Comparison

Goal:

- compare MLP and CNN more fairly
- use consistent limited-run settings
- track runtime awareness
- prepare for TensorBoard classification logging and checkpointing

## Important Rules

- Do not restart the curriculum.
- Inspect the public repo before assigning new work.
- Work under `/home/lily/ai-workspace`, not `/mnt/c`.
- Use Git for every meaningful change.
- Keep commits small and reviewable.
- Do not commit `.venv`, caches, checkpoints, logs, outputs, generated datasets, secrets, or API keys.
- Use `uv` for Python dependency management.
- Run the quality gate before committing code.
- AI may propose; the engineer must inspect, understand, test, and commit.
- Use local machine for development and small experiments.
- Use cloud GPU only when justified.
- Tests should be planned before implementation.
- Default unit tests should run on CPU.
- GPU behavior should be checked through scripts or quality gate modes.
- Documentation tasks require content review, not only Ruff/pytest.
