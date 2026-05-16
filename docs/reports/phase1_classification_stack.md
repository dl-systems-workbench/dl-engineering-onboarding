# Phase 1 Recap — FashionMNIST Classification Stack

## Purpose

Phase 1 built the first complete supervised PyTorch training stack in this onboarding repo.

The goal was not to maximize FashionMNIST accuracy. The goal was to build professional habits around tensors, modules, data loading, training loops, evaluation, checkpointing, logging, model comparison, train/validation/test discipline, and error analysis.

## What Was Built

### Core PyTorch Foundations

The repo now includes foundations for:

- tensor metadata and device selection
- autograd basics
- manual gradient-based training
- `Dataset` and `DataLoader`
- `nn.Module`
- optimizer-based training with `torch.optim`
- train/evaluation loops
- metrics aggregation
- checkpoint save/load
- TensorBoard scalar logging

Important files:

- `src/dl_onboarding/tensor_lab.py`
- `src/dl_onboarding/manual_training.py`
- `src/dl_onboarding/data_loading.py`
- `src/dl_onboarding/module_training.py`
- `src/dl_onboarding/evaluation.py`
- `src/dl_onboarding/checkpointing.py`
- `src/dl_onboarding/experiment_logging.py`

### FashionMNIST Classification Stack

The repo now includes a realistic but small image-classification workflow around FashionMNIST.

Important source file:

- `src/dl_onboarding/fashion_mnist.py`

Important components:

- `fashion_class_names`
- `FashionMLP`
- `FashionCNN`
- `count_trainable_parameters`
- `make_fashion_mnist_dataloaders`
- `split_dataset_train_val`
- `make_fashion_mnist_train_val_test_dataloaders`
- `classification_accuracy`
- `evaluate_classifier`
- `evaluate_classifier_with_confusion_matrix`
- `confusion_matrix_from_logits`
- `per_class_accuracy_from_confusion_matrix`
- `make_tiny_classification_dataloader`
- `train_fashion_mlp`
- `train_fashion_cnn`
- `train_fashion_classifier_experiment`

## Model Baselines

### FashionMLP

`FashionMLP` is the first simple classification baseline.

It flattens each image:

    (batch, 1, 28, 28) -> (batch, 784)

Then applies linear layers to produce logits:

    (batch, 10)

Main limitation:

- it ignores spatial image structure.

### FashionCNN

`FashionCNN` is the first spatial image baseline.

Shape flow:

    input:       (batch, 1, 28, 28)
    conv1:       (batch, 16, 28, 28)
    pool1:       (batch, 16, 14, 14)
    conv2:       (batch, 32, 14, 14)
    pool2:       (batch, 32, 7, 7)
    flatten:     (batch, 1568)
    logits:      (batch, 10)

Trainable parameters:

    105,866

Why it matters:

- it preserves spatial structure before classification
- it learns local visual features
- it introduces feature extractor + classifier head architecture

## Classification Rules Learned

The model returns raw logits.

For `nn.CrossEntropyLoss`:

- input shape is `(batch, num_classes)`
- labels shape is `(batch,)`
- labels are integer class IDs
- do not apply softmax before the loss

Prediction rule:

    predicted_labels = logits.argmax(dim=1)

## Experiment Workflow Built

The repo now supports classification experiments with:

- train/eval loss
- train/eval accuracy
- TensorBoard scalar logging
- checkpoint writing
- train/validation/test split
- limited local GPU smoke runs
- controlled model comparison
- confusion matrix evaluation
- per-class accuracy
- top class-confusion analysis

Important scripts:

- `scripts/fashion_mnist_lab.py`
- `scripts/fashion_cnn_lab.py`
- `scripts/fashion_model_comparison_lab.py`
- `scripts/fashion_experiment_lab.py`
- `scripts/fashion_train_val_test_lab.py`
- `scripts/fashion_error_analysis_lab.py`

## TensorBoard Notes

TensorBoard may show both raw and smoothed curves.

For very short runs, such as 2 epochs:

- smoothing can be misleading
- set smoothing to 0 for direct inspection

Runtime logs are stored under:

    runs/

These files are generated artifacts and should not be committed.

## Checkpointing Notes

Classification experiment checkpoints store:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `metrics`
- `metadata`

Checkpoints are stored under:

    outputs/

These are generated artifacts and should not be committed.

Current checkpointing proves:

- model state can be saved
- checkpoint files are created
- required keys exist
- restored model predictions can match

Current checkpointing does not yet prove:

- exact deterministic resume
- best-checkpoint selection
- long-run training recovery

## Train / Validation / Test Discipline

Current split policy:

- train split: used for gradient updates
- validation split: used for model-development evaluation
- test split: used for final held-out evaluation

Important rule:

- do not tune repeatedly on the test set

The current split is deterministic with a seed for local workflow consistency, but full reproducibility across machines, hardware, and PyTorch versions is not guaranteed.

## Error Analysis

The confusion matrix convention is:

    rows = true labels
    columns = predicted labels

Diagonal entries are correct predictions.

Off-diagonal entries are mistakes.

Per-class accuracy is:

    correct examples for class / total true examples for class

T1.14 showed that overall accuracy alone is not enough. In the limited CNN run, the model was especially weak on `Shirt`, with many confusions into `Coat`, `Pullover`, and `T-shirt/top`.

This is the first real applied-research behavior in the repo:

    train -> evaluate -> inspect failures -> form next experiment hypothesis

## Testing Stack

Important tests:

- `tests/test_fashion_mnist.py`
- `tests/test_checkpointing.py`
- `tests/test_experiment_logging.py`
- `tests/test_evaluation.py`

The FashionMNIST tests now cover:

- class names
- MLP forward shape
- CNN forward shape
- CrossEntropyLoss compatibility
- parameter count
- training-step updates
- metric reporting
- checkpoint creation
- checkpoint required keys
- checkpoint restore predictions
- TensorBoard event creation
- deterministic train/validation split
- confusion matrix counting
- per-class accuracy behavior

Testing policy:

- unit tests should be CPU-friendly
- real dataset/GPU behavior belongs in scripts and quality-gate smoke checks
- generated data, logs, checkpoints, and outputs stay ignored

## Quality Gate

Standard command:

    ./scripts/quality_check.sh

PyTorch/GPU verification command:

    ./scripts/quality_check.sh --torch

Safe auto-fix command:

    ./scripts/quality_check.sh --fix

## Current Limitations

Phase 1 does not yet solve:

- production inference packaging
- top-k prediction utilities
- image-level inspection of misclassified examples
- better CNN architecture design
- data augmentation
- normalization experiments
- optimizer comparisons
- learning-rate schedules
- best-checkpoint selection
- resume training
- config-driven experiments
- CI
- Hugging Face publication
- transformer workflows
- paper reproduction

## Phase 1 Promotion Summary

Phase 1 proves that the repo can now support a small but professional supervised PyTorch classification workflow.

The learner can now explain and work with:

- tensors and devices
- autograd
- DataLoaders
- `nn.Module`
- training loops
- optimizers
- evaluation mode
- `torch.no_grad`
- logits and CrossEntropyLoss
- MLP vs CNN baselines
- checkpointing
- TensorBoard logging
- train/validation/test split
- confusion matrices
- per-class error analysis

Next phase should begin with inference and checkpoint loading before moving into architecture improvements.
