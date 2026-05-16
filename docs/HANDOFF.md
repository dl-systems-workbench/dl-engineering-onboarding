# Handoff State

## Source of Truth

Public repo:

    https://github.com/dl-systems-workbench/dl-engineering-onboarding

Future assistants should inspect the repo directly, preferably using latest commit views or commit-specific URLs.

## Current Position

Current phase:

    Phase 1 — PyTorch Engineering Foundations

Current task:

    T1.15 — Phase 1 Classification Stack Recap and Handoff Refresh

Latest accepted technical task:

    T1.14 — Confusion Matrix and Per-Class Error Analysis

Next expected phase after T1.15 acceptance:

    Phase 2 — Computer Vision Engineering and Controlled Improvement

Likely next task:

    T2.1 — Inference Entry Point and Checkpoint Loading

## Machine

- Windows 10 Pro
- WSL2 Ubuntu
- Python 3.12 via `uv`
- Intel i7-9750H
- 16 GB RAM
- NVIDIA GTX 1650, 4 GB VRAM
- VS Code + WSL workflow

Local policy:

- use local CPU/GPU for tests, debugging, and tiny experiments
- use cloud only when larger training or modern foundation-model work justifies it

## Current Repo Stack

The repo has implemented:

- tensor/device/autograd basics
- manual training loop
- Dataset/DataLoader
- `nn.Module`
- optimizer-based training
- train/evaluation metrics
- checkpoint save/load
- TensorBoard logging
- FashionMNIST MLP baseline
- FashionMNIST CNN baseline
- controlled MLP vs CNN comparison
- classification experiment helper with checkpoint and TensorBoard logging
- train/validation/test dataloaders
- confusion matrix and per-class error analysis

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

## Important FashionMNIST Functions and Classes

- `FashionMLP`
- `FashionCNN`
- `count_trainable_parameters`
- `make_fashion_mnist_dataloaders`
- `split_dataset_train_val`
- `make_fashion_mnist_train_val_test_dataloaders`
- `classification_accuracy`
- `evaluate_classifier`
- `train_fashion_mlp`
- `train_fashion_cnn`
- `train_fashion_classifier_experiment`
- `confusion_matrix_from_logits`
- `per_class_accuracy_from_confusion_matrix`
- `evaluate_classifier_with_confusion_matrix`

## Important Scripts

- `scripts/quality_check.sh`
- `scripts/verify_torch.py`
- `scripts/fashion_mnist_lab.py`
- `scripts/fashion_cnn_lab.py`
- `scripts/fashion_model_comparison_lab.py`
- `scripts/fashion_experiment_lab.py`
- `scripts/fashion_train_val_test_lab.py`
- `scripts/fashion_error_analysis_lab.py`

## Important Tests

- `tests/test_fashion_mnist.py`
- `tests/test_checkpointing.py`
- `tests/test_experiment_logging.py`
- `tests/test_evaluation.py`
- `tests/test_module_training.py`

Current test suite size after T1.14:

    53 tests

## Quality Commands

Standard:

    ./scripts/quality_check.sh

With PyTorch verification:

    ./scripts/quality_check.sh --torch

## Runtime Artifacts

Do not commit:

- `data/`
- `runs/`
- `outputs/`
- checkpoints
- logs
- caches
- `.venv/`

## Known TensorBoard Behavior

TensorBoard may show both raw and smoothed scalar curves.

For very short runs, set smoothing to 0.

## Current Limitations

Not yet implemented:

- production inference path
- top-k prediction helper
- image-level misclassification inspection
- stronger CNN variants
- augmentation and normalization experiments
- optimizer comparisons
- best checkpoint selection
- resume training
- config-driven experiments
- CI
- Hugging Face publication
- transformers
- paper reproduction

## Next Direction

After T1.15 is accepted, begin Phase 2 with:

    T2.1 — Inference Entry Point and Checkpoint Loading

The goal is to move from training-only workflows to end-to-end use:

    train -> checkpoint -> load -> infer -> explain prediction
