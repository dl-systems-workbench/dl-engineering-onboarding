from __future__ import annotations

import torch

from dl_onboarding import select_device, train_with_validation


def print_evaluation_summary(label: str, result: dict[str, object]) -> None:
    """Print a compact summary for train/validation training."""
    print(f"==> {label}")
    print(f"device: {result['device']}")
    print(f"initial train loss: {result['initial_train_loss']:.6f}")
    print(f"final train loss: {result['final_train_loss']:.6f}")
    print(f"initial val MSE: {result['initial_val_mse']:.6f}")
    print(f"final val MSE: {result['final_val_mse']:.6f}")
    print(f"final val RMSE: {result['final_val_rmse']:.6f}")
    print(f"final val MAE: {result['final_val_mae']:.6f}")
    print(f"final val R2: {result['final_val_r2']:.6f}")
    print(f"learned weight: {result['learned_weight']:.6f}")
    print(f"learned bias: {result['learned_bias']:.6f}")


def main() -> None:
    cpu_result = train_with_validation(device=torch.device("cpu"))
    print_evaluation_summary("CPU train/validation run", cpu_result)

    device = select_device()
    if device.type == "cuda":
        cuda_result = train_with_validation(device=device)
        print_evaluation_summary("CUDA train/validation run", cuda_result)
    else:
        print("CUDA train/validation run skipped because CUDA is not available.")


if __name__ == "__main__":
    main()
