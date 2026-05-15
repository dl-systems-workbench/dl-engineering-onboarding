from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dl_onboarding.evaluation import evaluate_regression, make_train_val_dataloaders
from dl_onboarding.manual_training import mse_loss
from dl_onboarding.module_training import LinearRegressionModel


def create_checkpoint(
    *,
    model: LinearRegressionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int],
    metadata: dict[str, str | int | float],
) -> dict[str, Any]:
    """Create a serializable training checkpoint dictionary."""
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "metadata": metadata,
    }


def save_checkpoint(
    *,
    path: str | Path,
    model: LinearRegressionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int],
    metadata: dict[str, str | int | float],
) -> Path:
    """Save a model, optimizer, metrics, and metadata checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = create_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        metadata=metadata,
    )

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    *,
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint dictionary onto the requested device."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to be a dictionary.")

    return checkpoint


def restore_model_and_optimizer(
    *,
    path: str | Path,
    device: torch.device,
    learning_rate: float,
) -> tuple[LinearRegressionModel, torch.optim.Optimizer, dict[str, Any]]:
    """Restore a LinearRegressionModel and SGD optimizer from a checkpoint."""
    checkpoint = load_checkpoint(path=path, device=device)

    model = LinearRegressionModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()

    return model, optimizer, checkpoint


def train_and_save_checkpoint(
    *,
    path: str | Path,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train LinearRegressionModel with validation and save a checkpoint."""
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

    final_metrics = evaluate_regression(
        model=model,
        dataloader=val_loader,
        device=active_device,
    )

    metadata = {
        "model_class": "LinearRegressionModel",
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
    }

    checkpoint_path = save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        metrics=final_metrics,
        metadata=metadata,
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "epoch": num_epochs,
        "final_train_loss": train_losses[-1],
        "final_val_mse": final_metrics["mse"],
        "final_val_rmse": final_metrics["rmse"],
        "final_val_mae": final_metrics["mae"],
        "final_val_r2": final_metrics["r2"],
        "learned_weight": float(model.weight.item()),
        "learned_bias": float(model.bias.item()),
        "metadata": metadata,
    }
