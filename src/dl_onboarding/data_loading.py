from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_onboarding.manual_training import mse_loss, predict_linear


def make_tensor_dataset(device: torch.device) -> TensorDataset:
    """Create a TensorDataset for y = 2x - 1."""
    x = torch.linspace(-1.0, 1.0, steps=64, device=device).reshape(-1, 1)
    y = 2.0 * x - 1.0

    return TensorDataset(x, y)


def make_dataloader(
    *,
    device: torch.device,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the tiny linear regression dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dataset = make_tensor_dataset(device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_with_dataloader(
    *,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train y = xw + b manually using mini-batches from a DataLoader."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    dataloader = make_dataloader(
        device=active_device,
        batch_size=batch_size,
        shuffle=True,
    )

    weight = torch.zeros((1, 1), device=active_device, requires_grad=True)
    bias = torch.zeros((1,), device=active_device, requires_grad=True)

    epoch_losses: list[float] = []

    for _ in range(num_epochs):
        batch_losses: list[float] = []

        for x_batch, y_batch in dataloader:
            prediction = predict_linear(x_batch, weight, bias)
            loss = mse_loss(prediction, y_batch)
            loss.backward()

            if weight.grad is None or bias.grad is None:
                raise RuntimeError("Expected gradients after backward().")

            with torch.no_grad():
                weight -= learning_rate * weight.grad
                bias -= learning_rate * bias.grad

            weight.grad = None
            bias.grad = None

            batch_losses.append(float(loss.item()))

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)

    return {
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "learned_weight": float(weight.item()),
        "learned_bias": float(bias.item()),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
        "epoch_losses": epoch_losses,
    }
