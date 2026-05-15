from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from dl_onboarding import (
    FashionCNN,
    FashionMLP,
    count_trainable_parameters,
    make_fashion_mnist_dataloaders,
    train_fashion_cnn,
    train_fashion_mlp,
)

TrainingFunction = Callable[..., dict[str, Any]]


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work before timing if the active device is CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_timed_training(
    *,
    model_name: str,
    train_fn: TrainingFunction,
    parameter_count: int,
    device: torch.device,
    seed: int,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    max_train_batches: int,
    max_eval_batches: int,
) -> dict[str, Any]:
    """Run one model training function with simple wall-clock timing."""
    torch.manual_seed(seed)

    synchronize_if_cuda(device)
    start_time = time.perf_counter()

    metrics = train_fn(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
    )

    synchronize_if_cuda(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "model": model_name,
        "parameters": parameter_count,
        "elapsed_seconds": elapsed_seconds,
        **metrics,
    }


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a compact model comparison table."""
    header = (
        f"{'Model':<12}"
        f"{'Params':>12}"
        f"{'Eval Acc':>12}"
        f"{'Eval Loss':>12}"
        f"{'Train Acc':>12}"
        f"{'Runtime':>12}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['model']:<12}"
            f"{result['parameters']:>12,}"
            f"{result['final_eval_accuracy']:>12.4f}"
            f"{result['final_eval_loss']:>12.4f}"
            f"{result['final_train_accuracy']:>12.4f}"
            f"{result['elapsed_seconds']:>11.2f}s"
        )


def main() -> None:
    """Compare FashionMLP and FashionCNN under the same limited run budget."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    num_epochs = 2
    learning_rate = 0.1
    max_train_batches = 100
    max_eval_batches = 50
    seed = 0

    print("Controlled FashionMNIST model comparison")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max train batches: {max_train_batches}")
    print(f"Max eval batches: {max_eval_batches}")
    print(f"Seed before each model run: {seed}")
    print()

    train_loader, eval_loader = make_fashion_mnist_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        download=True,
    )

    mlp_parameter_count = count_trainable_parameters(FashionMLP())
    cnn_parameter_count = count_trainable_parameters(FashionCNN())

    results = [
        run_timed_training(
            model_name="FashionMLP",
            train_fn=train_fashion_mlp,
            parameter_count=mlp_parameter_count,
            device=device,
            seed=seed,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        run_timed_training(
            model_name="FashionCNN",
            train_fn=train_fashion_cnn,
            parameter_count=cnn_parameter_count,
            device=device,
            seed=seed,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
    ]

    print_comparison_table(results)
    print()
    print("Notes:")
    print("- This is a controlled local smoke comparison, not a final benchmark.")
    print("- Small accuracy differences can be noise in short limited runs.")
    print("- Runtime includes Python, DataLoader, training, and evaluation overhead.")


if __name__ == "__main__":
    main()
