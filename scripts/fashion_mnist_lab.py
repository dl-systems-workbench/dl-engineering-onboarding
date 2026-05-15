from __future__ import annotations

from dl_onboarding import (
    fashion_class_names,
    make_fashion_mnist_dataloaders,
    select_device,
    train_fashion_mlp,
)


def print_classification_summary(label: str, result: dict[str, object]) -> None:
    """Print a compact summary for a FashionMNIST classification run."""
    print(f"==> {label}")
    print(f"device: {result['device']}")
    print(f"initial train loss: {result['initial_train_loss']:.6f}")
    print(f"final train loss: {result['final_train_loss']:.6f}")
    print(f"initial train accuracy: {result['initial_train_accuracy']:.4f}")
    print(f"final train accuracy: {result['final_train_accuracy']:.4f}")
    print(f"initial eval accuracy: {result['initial_eval_accuracy']:.4f}")
    print(f"final eval accuracy: {result['final_eval_accuracy']:.4f}")


def main() -> None:
    print(f"FashionMNIST classes: {fashion_class_names()}")

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        data_dir="data",
        batch_size=64,
        download=True,
    )

    images, labels = next(iter(train_loader))
    print(f"image batch shape: {tuple(images.shape)}")
    print(f"label batch shape: {tuple(labels.shape)}")
    print(f"first label: {int(labels[0].item())}")

    device = select_device()

    result = train_fashion_mlp(
        train_loader=train_loader,
        eval_loader=test_loader,
        num_epochs=2,
        learning_rate=0.1,
        device=device,
        max_train_batches=200,
        max_eval_batches=80,
    )
    print_classification_summary("FashionMNIST limited MLP run", result)

    if device.type != "cuda":
        print("CUDA was not available, so this run used CPU.")


if __name__ == "__main__":
    main()
