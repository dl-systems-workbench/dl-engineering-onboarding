from __future__ import annotations

from typing import Any

import torch


def make_linear_regression_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a tiny deterministic dataset for y = 2x - 1."""
    x = torch.linspace(-1.0, 1.0, steps=64, device=device).reshape(-1, 1)
    y = 2.0 * x - 1.0

    return x, y


def predict_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute predictions for a one-feature linear model."""
    return x @ weight + bias


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error loss."""
    return ((prediction - target) ** 2).mean()


def train_manual_linear_regression(
    *,
    num_steps: int = 200,
    learning_rate: float = 0.1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train y = xw + b manually using PyTorch autograd."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    x, y = make_linear_regression_data(active_device)

    weight = torch.zeros((1, 1), device=active_device, requires_grad=True)
    bias = torch.zeros((1,), device=active_device, requires_grad=True)

    loss_history: list[float] = []

    for _ in range(num_steps):
        prediction = predict_linear(x, weight, bias)
        loss = mse_loss(prediction, y)
        loss.backward()

        if weight.grad is None or bias.grad is None:
            raise RuntimeError("Expected gradients to be populated after backward().")

        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        weight.grad = None
        bias.grad = None

        loss_history.append(float(loss.item()))

    return {
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "learned_weight": float(weight.item()),
        "learned_bias": float(bias.item()),
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "device": str(active_device),
        "loss_history": loss_history,
    }
