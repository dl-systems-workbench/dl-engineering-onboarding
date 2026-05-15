from __future__ import annotations

import torch

from dl_onboarding import (
    FashionCNN,
    count_trainable_parameters,
    make_fashion_mnist_dataloaders,
    train_fashion_classifier_experiment,
)


def main() -> None:
    """Run a limited FashionMNIST CNN experiment with logs and checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN()

    print("FashionMNIST CNN experiment")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, eval_loader = make_fashion_mnist_dataloaders(
        data_dir="data",
        batch_size=64,
        download=True,
    )

    metrics = train_fashion_classifier_experiment(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_dir="outputs/fashion_experiments/cnn_smoke",
        log_dir="runs/fashion_experiments/cnn_smoke",
        checkpoint_name="fashion_cnn_checkpoint.pt",
        num_epochs=2,
        learning_rate=0.1,
        device=device,
        max_train_batches=100,
        max_eval_batches=50,
    )

    print("Experiment metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print()
    print("To inspect TensorBoard locally:")
    print("  uv run tensorboard --logdir runs/fashion_experiments")


if __name__ == "__main__":
    main()
