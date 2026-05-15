from __future__ import annotations

import torch

from dl_onboarding import (
    FashionCNN,
    count_trainable_parameters,
    evaluate_classifier,
    make_fashion_mnist_train_val_test_dataloaders,
    train_fashion_classifier_experiment,
)


def main() -> None:
    """Run a FashionMNIST experiment with separate train, val, and test loaders."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN()

    print("FashionMNIST train/validation/test experiment")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, val_loader, test_loader = make_fashion_mnist_train_val_test_dataloaders(
        data_dir="data",
        batch_size=64,
        val_fraction=0.1,
        seed=0,
        download=True,
    )

    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(val_loader.dataset)}")
    print(f"Test examples: {len(test_loader.dataset)}")

    metrics = train_fashion_classifier_experiment(
        model=model,
        train_loader=train_loader,
        eval_loader=val_loader,
        run_dir="outputs/fashion_experiments/cnn_train_val_test",
        log_dir="runs/fashion_experiments/cnn_train_val_test",
        checkpoint_name="fashion_cnn_train_val_checkpoint.pt",
        num_epochs=2,
        learning_rate=0.1,
        device=device,
        max_train_batches=100,
        max_eval_batches=50,
        model_name="FashionCNN",
    )

    test_metrics = evaluate_classifier(
        model=model,
        dataloader=test_loader,
        device=device,
        max_batches=50,
    )

    print("Validation experiment metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("Final limited test metrics:")
    for key, value in test_metrics.items():
        print(f"  test_{key}: {value}")

    print()
    print("To inspect TensorBoard locally:")
    print("  uv run tensorboard --logdir runs/fashion_experiments")


if __name__ == "__main__":
    main()
