"""Utilities for the Deep Learning Engineering onboarding repo."""

from dl_onboarding.checkpointing import (
    create_checkpoint,
    load_checkpoint,
    restore_model_and_optimizer,
    save_checkpoint,
    train_and_save_checkpoint,
)
from dl_onboarding.data_loading import (
    make_dataloader,
    make_tensor_dataset,
    train_with_dataloader,
)
from dl_onboarding.evaluation import (
    evaluate_regression,
    make_train_val_dataloaders,
    regression_metrics,
    split_tensor_dataset,
    train_with_validation,
)
from dl_onboarding.experiment_logging import (
    find_tensorboard_event_files,
    train_with_tensorboard_logging,
)
from dl_onboarding.fashion_mnist import (
    FashionCNN,
    FashionMLP,
    classification_accuracy,
    count_trainable_parameters,
    evaluate_classifier,
    fashion_class_names,
    make_fashion_mnist_dataloaders,
    make_tiny_classification_dataloader,
    train_fashion_classifier_experiment,
    train_fashion_cnn,
    train_fashion_mlp,
)
from dl_onboarding.manual_training import (
    make_linear_regression_data,
    mse_loss,
    predict_linear,
    train_manual_linear_regression,
)
from dl_onboarding.module_training import (
    LinearRegressionModel,
    train_module_linear_regression,
    train_module_with_optimizer,
)
from dl_onboarding.system_info import get_runtime_summary
from dl_onboarding.tensor_lab import (
    matrix_smoke,
    quadratic_autograd,
    select_device,
    tensor_summary,
)

__all__ = [
    "LinearRegressionModel",
    "get_runtime_summary",
    "make_dataloader",
    "make_linear_regression_data",
    "make_tensor_dataset",
    "matrix_smoke",
    "mse_loss",
    "predict_linear",
    "quadratic_autograd",
    "select_device",
    "tensor_summary",
    "train_manual_linear_regression",
    "train_module_linear_regression",
    "train_module_with_optimizer",
    "train_with_dataloader",
    "train_with_validation",
    "train_fashion_mlp",
    "make_tiny_classification_dataloader",
    "make_fashion_mnist_dataloaders",
    "fashion_class_names",
    "evaluate_classifier",
    "classification_accuracy",
    "FashionMLP",
    "FashionCNN",
    "train_fashion_cnn",
    "train_fashion_classifier_experiment",
    "count_trainable_parameters",
    "train_with_tensorboard_logging",
    "find_tensorboard_event_files",
    "train_and_save_checkpoint",
    "save_checkpoint",
    "restore_model_and_optimizer",
    "load_checkpoint",
    "create_checkpoint",
    "split_tensor_dataset",
    "regression_metrics",
    "make_train_val_dataloaders",
    "evaluate_regression",
]
