from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def load_model_from_state_dict_checkpoint(
    *,
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model weights from a state-dict checkpoint and prepare eval mode."""
    active_device = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=active_device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint must contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(active_device)
    model.eval()

    return model, checkpoint


def top_k_predictions_from_logits(
    logits: torch.Tensor,
    *,
    class_names: Sequence[str],
    k: int = 3,
) -> list[list[dict[str, int | str | float]]]:
    """Convert logits into top-k class predictions for each example."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes).")

    if not class_names:
        raise ValueError("class_names must be non-empty.")

    if logits.shape[1] != len(class_names):
        raise ValueError("logits second dimension must match class_names length.")

    if not 1 <= k <= len(class_names):
        raise ValueError("k must be between 1 and the number of classes.")

    probabilities = F.softmax(logits, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=1)

    predictions: list[list[dict[str, int | str | float]]] = []

    for example_probabilities, example_indices in zip(
        top_probabilities.cpu(),
        top_indices.cpu(),
        strict=True,
    ):
        example_predictions: list[dict[str, int | str | float]] = []

        for probability, class_index in zip(
            example_probabilities,
            example_indices,
            strict=True,
        ):
            class_index_int = int(class_index.item())
            example_predictions.append(
                {
                    "class_index": class_index_int,
                    "class_name": class_names[class_index_int],
                    "probability": float(probability.item()),
                },
            )

        predictions.append(example_predictions)

    return predictions


def predict_top_k_for_batch(
    *,
    model: nn.Module,
    images: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    k: int = 3,
) -> list[list[dict[str, int | str | float]]]:
    """Run model inference on an image batch and return top-k predictions."""
    was_training = model.training
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))

    if was_training:
        model.train()

    return top_k_predictions_from_logits(
        logits,
        class_names=class_names,
        k=k,
    )
