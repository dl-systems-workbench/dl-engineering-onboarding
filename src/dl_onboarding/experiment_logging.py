from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from dl_onboarding.evaluation import evaluate_regression, make_train_val_dataloaders
from dl_onboarding.manual_training import mse_loss
from dl_onboarding.module_training import LinearRegressionModel


def find_tensorboard_event_files(log_dir: str | Path) -> list[Path]:
    """Return TensorBoard event files under a log directory."""
    path = Path(log_dir)

    if not path.exists():
        return []

    return sorted(path.rglob("events.out.tfevents.*"))


def train_with_tensorboard_logging(
    *,
    log_dir: str | Path,
    num_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train LinearRegressionModel and log scalar metrics to TensorBoard."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    active_device = device if device is not None else torch.device("cpu")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    model = LinearRegressionModel().to(active_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader, val_loader = make_train_val_dataloaders(
        device=torch.device("cpu"),
        batch_size=batch_size,
    )

    train_losses: list[float] = []
    val_mse_history: list[float] = []
    val_r2_history: list[float] = []

    writer = SummaryWriter(log_dir=str(log_path))

    try:
        for epoch in range(num_epochs):
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

            train_loss = sum(batch_losses) / len(batch_losses)
            val_metrics = evaluate_regression(
                model=model,
                dataloader=val_loader,
                device=active_device,
            )

            train_losses.append(train_loss)
            val_mse_history.append(float(val_metrics["mse"]))
            val_r2_history.append(float(val_metrics["r2"]))

            step = epoch + 1
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Metrics/val_mse", val_metrics["mse"], step)
            writer.add_scalar("Metrics/val_rmse", val_metrics["rmse"], step)
            writer.add_scalar("Metrics/val_mae", val_metrics["mae"], step)
            writer.add_scalar("Metrics/val_r2", val_metrics["r2"], step)
            writer.add_scalar("Parameters/weight", float(model.weight.item()), step)
            writer.add_scalar("Parameters/bias", float(model.bias.item()), step)
    finally:
        writer.flush()
        writer.close()

    event_files = find_tensorboard_event_files(log_path)

    return {
        "log_dir": str(log_path),
        "event_file_count": len(event_files),
        "initial_train_loss": train_losses[0],
        "final_train_loss": train_losses[-1],
        "initial_val_mse": val_mse_history[0],
        "final_val_mse": val_mse_history[-1],
        "final_val_r2": val_r2_history[-1],
        "learned_weight": float(model.weight.item()),
        "learned_bias": float(model.bias.item()),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(active_device),
    }
