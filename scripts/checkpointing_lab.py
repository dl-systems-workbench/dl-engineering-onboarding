from __future__ import annotations

from pathlib import Path

import torch

from dl_onboarding import (
    restore_model_and_optimizer,
    select_device,
    train_and_save_checkpoint,
)


def main() -> None:
    checkpoint_path = Path("outputs/checkpoints/linear_regression_checkpoint.pt")

    result = train_and_save_checkpoint(
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    print("==> Saved checkpoint")
    print(f"path: {result['checkpoint_path']}")
    print(f"epoch: {result['epoch']}")
    print(f"final val MSE: {result['final_val_mse']:.6f}")
    print(f"final val R2: {result['final_val_r2']:.6f}")
    print(f"learned weight: {result['learned_weight']:.6f}")
    print(f"learned bias: {result['learned_bias']:.6f}")

    device = select_device()
    model, _, checkpoint = restore_model_and_optimizer(
        path=checkpoint_path,
        device=device,
        learning_rate=0.1,
    )

    x = torch.tensor([[-1.0], [0.0], [1.0]], device=device)

    with torch.no_grad():
        prediction = model(x).detach().cpu().reshape(-1).tolist()

    print("==> Restored checkpoint")
    print(f"loaded device: {device}")
    print(f"loaded epoch: {checkpoint['epoch']}")
    print(f"metadata: {checkpoint['metadata']}")
    print(f"predictions for x=[-1, 0, 1]: {prediction}")


if __name__ == "__main__":
    main()
