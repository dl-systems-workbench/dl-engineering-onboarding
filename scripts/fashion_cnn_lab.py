from __future__ import annotations

import torch

from dl_onboarding import (
    FashionCNN,
    count_trainable_parameters,
    fashion_class_names,
    make_fashion_mnist_dataloaders,
    train_fashion_cnn,
)


def main() -> None:
    """Run a limited FashionMNIST CNN smoke experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Classes: {fashion_class_names()}")

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        data_dir="data",
        batch_size=64,
        download=True,
    )

    images, labels = next(iter(train_loader))
    model = FashionCNN()

    with torch.no_grad():
        logits = model(images[:4])

    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"CNN logits shape for 4 examples: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    metrics = train_fashion_cnn(
        train_loader=train_loader,
        eval_loader=test_loader,
        num_epochs=2,
        learning_rate=0.1,
        device=device,
        max_train_batches=100,
        max_eval_batches=50,
    )

    print("CNN limited training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
