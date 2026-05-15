from __future__ import annotations

import torch


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Return cuda when requested and available, otherwise return cpu."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def tensor_summary(tensor: torch.Tensor) -> dict[str, str]:
    """Return key metadata about a tensor."""
    return {
        "shape": str(tuple(tensor.shape)),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": str(tensor.requires_grad),
    }


def quadratic_autograd(x_value: float) -> dict[str, float]:
    """Compute y = x^2 + 3x + 2 and dy/dx using PyTorch autograd."""
    x = torch.tensor(float(x_value), requires_grad=True)
    y = x**2 + 3 * x + 2

    y.backward()

    if x.grad is None:
        raise RuntimeError("Expected x.grad to be populated after backward().")

    return {
        "x": float(x.item()),
        "y": float(y.item()),
        "dy_dx": float(x.grad.item()),
    }


def matrix_smoke(device: torch.device) -> dict[str, object]:
    """Run a tiny matrix multiplication on the requested device."""
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    b = torch.ones((3, 2), dtype=torch.float32, device=device)
    result = a @ b

    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "shape": tuple(result.shape),
        "device": str(result.device),
        "sum": float(result.sum().item()),
    }
