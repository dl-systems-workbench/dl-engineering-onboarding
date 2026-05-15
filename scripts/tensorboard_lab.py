from __future__ import annotations

from pathlib import Path

import torch

from dl_onboarding import select_device, train_with_tensorboard_logging


def print_tensorboard_summary(label: str, result: dict[str, object]) -> None:
    """Print a compact summary for a TensorBoard-logged run."""
    print(f"==> {label}")
    print(f"device: {result['device']}")
    print(f"log dir: {result['log_dir']}")
    print(f"event files: {result['event_file_count']}")
    print(f"initial train loss: {result['initial_train_loss']:.6f}")
    print(f"final train loss: {result['final_train_loss']:.6f}")
    print(f"initial val MSE: {result['initial_val_mse']:.6f}")
    print(f"final val MSE: {result['final_val_mse']:.6f}")
    print(f"final val R2: {result['final_val_r2']:.6f}")
    print(f"learned weight: {result['learned_weight']:.6f}")
    print(f"learned bias: {result['learned_bias']:.6f}")


def main() -> None:
    cpu_result = train_with_tensorboard_logging(
        log_dir=Path("runs/tensorboard/linear_regression_cpu"),
        device=torch.device("cpu"),
    )
    print_tensorboard_summary("CPU TensorBoard run", cpu_result)

    device = select_device()
    if device.type == "cuda":
        cuda_result = train_with_tensorboard_logging(
            log_dir=Path("runs/tensorboard/linear_regression_cuda"),
            device=device,
        )
        print_tensorboard_summary("CUDA TensorBoard run", cuda_result)
    else:
        print("CUDA TensorBoard run skipped because CUDA is not available.")

    print("")
    print("To view locally, run:")
    print("uv run tensorboard --logdir runs/tensorboard --port 6006")


if __name__ == "__main__":
    main()
