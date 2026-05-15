from __future__ import annotations

import torch

from dl_onboarding import matrix_smoke, quadratic_autograd, select_device, tensor_summary


def main() -> None:
    device = select_device()
    print(f"Selected device: {device}")

    tensor = torch.randn(2, 3, device=device)
    print(f"Tensor summary: {tensor_summary(tensor)}")

    autograd_result = quadratic_autograd(2.0)
    print(f"Autograd result: {autograd_result}")

    matrix_result = matrix_smoke(device)
    print(f"Matrix smoke result: {matrix_result}")


if __name__ == "__main__":
    main()
