from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dl_onboarding.data_loading import make_dataloader
from dl_onboarding.manual_training import mse_loss


class LinearRegressionModel(nn.Module):
    """A one-feature linear regression model: y = xw + b."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, 1)))
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute model predictions."""
        return x @ self.weight + self.bias


def train_module_linear_regression(
    *,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train LinearRegressionModel manually using PyTorch autograd."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    model = LinearRegressionModel().to(active_device)

    dataloader = make_dataloader(
        device=torch.device("cpu"),
        batch_size=batch_size,
        shuffle=True,
    )

    epoch_losses: list[float] = []

    for _ in range(num_epochs):
        batch_losses: list[float] = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(active_device)
            y_batch = y_batch.to(active_device)

            prediction = model(x_batch)
            loss = mse_loss(prediction, y_batch)
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is None:
                        raise RuntimeError("Expected all model parameters to have gradients.")

                    parameter.add_(parameter.grad, alpha=-learning_rate)

            for parameter in model.parameters():
                parameter.grad = None

            batch_losses.append(float(loss.item()))

        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    state_dict_keys = list(model.state_dict().keys())

    return {
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "learned_weight": float(model.weight.item()),
        "learned_bias": float(model.bias.item()),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
        "epoch_losses": epoch_losses,
        "state_dict_keys": state_dict_keys,
    }


def train_module_with_optimizer(
    *,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train LinearRegressionModel using torch.optim.SGD."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    model = LinearRegressionModel().to(active_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    dataloader = make_dataloader(
        device=torch.device("cpu"),
        batch_size=batch_size,
        shuffle=True,
    )

    epoch_losses: list[float] = []

    for _ in range(num_epochs):
        batch_losses: list[float] = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(active_device)
            y_batch = y_batch.to(active_device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = mse_loss(prediction, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return {
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "learned_weight": float(model.weight.item()),
        "learned_bias": float(model.bias.item()),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
        "optimizer_name": optimizer.__class__.__name__,
        "epoch_losses": epoch_losses,
        "state_dict_keys": list(model.state_dict().keys()),
    }
