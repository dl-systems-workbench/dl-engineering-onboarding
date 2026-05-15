from __future__ import annotations

import torch

from dl_onboarding import make_dataloader, select_device, train_with_dataloader


def print_training_summary(label: str, result: dict[str, object]) -> None:
    """Print a compact summary for a DataLoader training run."""
    print(f"==> {label}")
    print(f"device: {result['device']}")
    print(f"initial loss: {result['initial_loss']:.6f}")
    print(f"final loss: {result['final_loss']:.6f}")
    print(f"learned weight: {result['learned_weight']:.6f}")
    print(f"learned bias: {result['learned_bias']:.6f}")
    print(f"batch size: {result['batch_size']}")


def main() -> None:
    device = select_device()
    print(f"Selected device: {device}")

    dataloader = make_dataloader(
        device=device,
        batch_size=16,
        shuffle=False,
    )

    x_batch, y_batch = next(iter(dataloader))
    print(f"x batch shape: {tuple(x_batch.shape)}, device: {x_batch.device}")
    print(f"y batch shape: {tuple(y_batch.shape)}, device: {y_batch.device}")

    cpu_result = train_with_dataloader(device=torch.device("cpu"))
    print_training_summary("CPU DataLoader training", cpu_result)

    if device.type == "cuda":
        cuda_result = train_with_dataloader(device=device)
        print_training_summary("CUDA DataLoader training", cuda_result)


if __name__ == "__main__":
    main()
