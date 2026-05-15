from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_onboarding.data_loading import make_tensor_dataset
from dl_onboarding.manual_training import mse_loss
from dl_onboarding.module_training import LinearRegressionModel


def split_tensor_dataset(
    *,
    device: torch.device,
    train_size: int = 48,
) -> tuple[TensorDataset, TensorDataset]:
    """Split the tiny regression TensorDataset into train and validation datasets."""
    full_dataset = make_tensor_dataset(device)
    x, y = full_dataset.tensors

    if train_size <= 0:
        raise ValueError("train_size must be positive.")

    if train_size >= len(full_dataset):
        raise ValueError("train_size must be smaller than the full dataset length.")

    train_dataset = TensorDataset(x[:train_size], y[:train_size])
    val_dataset = TensorDataset(x[train_size:], y[train_size:])

    return train_dataset, val_dataset


def make_train_val_dataloaders(
    *,
    device: torch.device,
    batch_size: int = 16,
    train_size: int = 48,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    train_dataset, val_dataset = split_tensor_dataset(
        device=device,
        train_size=train_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute regression metrics for prediction and target tensors."""
    prediction_flat = prediction.detach().reshape(-1)
    target_flat = target.detach().reshape(-1)

    error = prediction_flat - target_flat
    mse = torch.mean(error**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(error))

    ss_res = torch.sum((target_flat - prediction_flat) ** 2)
    ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2)

    if math.isclose(float(ss_tot.item()), 0.0):
        r2 = float("nan")
    else:
        r2 = 1.0 - float((ss_res / ss_tot).item())

    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "r2": r2,
    }


def evaluate_regression(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a regression model without computing gradients."""
    was_training = model.training
    model.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            prediction = model(x_batch)

            predictions.append(prediction.detach().cpu())
            targets.append(y_batch.detach().cpu())

    if was_training:
        model.train()

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    metrics = regression_metrics(prediction_tensor, target_tensor)

    return {
        **metrics,
        "num_examples": int(target_tensor.shape[0]),
    }


def train_with_validation(
    *,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train LinearRegressionModel and evaluate on a validation split each epoch."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    model = LinearRegressionModel().to(active_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader, val_loader = make_train_val_dataloaders(
        device=torch.device("cpu"),
        batch_size=batch_size,
    )

    train_losses: list[float] = []
    val_mse_history: list[float] = []
    val_r2_history: list[float] = []

    for _ in range(num_epochs):
        model.train()
        batch_losses: list[float] = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(active_device)
            y_batch = y_batch.to(active_device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = mse_loss(prediction, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_losses.append(sum(batch_losses) / len(batch_losses))

        val_metrics = evaluate_regression(
            model=model,
            dataloader=val_loader,
            device=active_device,
        )
        val_mse_history.append(float(val_metrics["mse"]))
        val_r2_history.append(float(val_metrics["r2"]))

    final_val_metrics = evaluate_regression(
        model=model,
        dataloader=val_loader,
        device=active_device,
    )

    return {
        "initial_train_loss": train_losses[0],
        "final_train_loss": train_losses[-1],
        "initial_val_mse": val_mse_history[0],
        "final_val_mse": final_val_metrics["mse"],
        "final_val_rmse": final_val_metrics["rmse"],
        "final_val_mae": final_val_metrics["mae"],
        "final_val_r2": final_val_metrics["r2"],
        "learned_weight": float(model.weight.item()),
        "learned_bias": float(model.bias.item()),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
        "train_losses": train_losses,
        "val_mse_history": val_mse_history,
        "val_r2_history": val_r2_history,
    }
