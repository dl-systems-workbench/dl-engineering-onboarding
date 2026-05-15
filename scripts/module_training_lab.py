from __future__ import annotations

import torch

from dl_onboarding import select_device, train_module_linear_regression


def print_training_summary(label: str, result: dict[str, object]) -> None:
    """Print a compact summary for an nn.Module training run."""
    print(f"==> {label}")
    print(f"device: {result['device']}")
    print(f"initial loss: {result['initial_loss']:.6f}")
    print(f"final loss: {result['final_loss']:.6f}")
    print(f"learned weight: {result['learned_weight']:.6f}")
    print(f"learned bias: {result['learned_bias']:.6f}")
    print(f"state_dict keys: {result['state_dict_keys']}")


def main() -> None:
    cpu_result = train_module_linear_regression(device=torch.device("cpu"))
    print_training_summary("CPU nn.Module training", cpu_result)

    device = select_device()
    if device.type == "cuda":
        cuda_result = train_module_linear_regression(device=device)
        print_training_summary("CUDA nn.Module training", cuda_result)
    else:
        print("CUDA nn.Module training skipped because CUDA is not available.")


if __name__ == "__main__":
    main()
