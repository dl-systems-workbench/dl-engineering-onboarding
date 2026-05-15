from __future__ import annotations

import torch
import torchvision


def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    cpu_tensor = torch.randn(2, 3)
    print(f"CPU tensor: shape={tuple(cpu_tensor.shape)}, device={cpu_tensor.device}")

    if not torch.cuda.is_available():
        print("CUDA smoke test skipped because CUDA is not available.")
        return

    device = torch.device("cuda")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    gpu_tensor = torch.randn(512, 512, device=device)
    result = gpu_tensor @ gpu_tensor.T
    torch.cuda.synchronize()

    print(
        "GPU tensor result: "
        f"shape={tuple(result.shape)}, "
        f"device={result.device}, "
        f"dtype={result.dtype}"
    )
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
