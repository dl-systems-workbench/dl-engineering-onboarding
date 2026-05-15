from __future__ import annotations

import torch

from dl_onboarding import (
    FashionCNN,
    count_trainable_parameters,
    evaluate_classifier_with_confusion_matrix,
    fashion_class_names,
    make_fashion_mnist_train_val_test_dataloaders,
    train_fashion_classifier_experiment,
)


def print_per_class_accuracy(
    *,
    class_names: tuple[str, ...],
    confusion_matrix: torch.Tensor,
    per_class_accuracy: torch.Tensor,
) -> None:
    """Print support, correct count, and accuracy for each class."""
    print("Per-class accuracy:")
    print(f"{'Class':<14}{'Support':>10}{'Correct':>10}{'Accuracy':>12}")
    print("-" * 46)

    for class_index, class_name in enumerate(class_names):
        support = int(confusion_matrix[class_index].sum().item())
        correct = int(confusion_matrix[class_index, class_index].item())
        accuracy = float(per_class_accuracy[class_index].item())

        print(f"{class_name:<14}{support:>10}{correct:>10}{accuracy:>11.4f}")


def print_top_confusions(
    *,
    class_names: tuple[str, ...],
    confusion_matrix: torch.Tensor,
    top_k: int = 8,
) -> None:
    """Print the largest off-diagonal confusion counts."""
    matrix = confusion_matrix.clone()
    matrix.fill_diagonal_(0)

    flat_counts = matrix.flatten()
    top_values, top_indices = torch.topk(
        flat_counts,
        k=min(top_k, flat_counts.numel()),
    )

    print("Top confusions:")
    printed = 0
    for value, flat_index in zip(top_values, top_indices, strict=True):
        count = int(value.item())
        if count == 0:
            continue

        true_index = int(flat_index.item() // len(class_names))
        predicted_index = int(flat_index.item() % len(class_names))

        print(
            f"  true={class_names[true_index]!r} "
            f"predicted={class_names[predicted_index]!r} "
            f"count={count}"
        )
        printed += 1

    if printed == 0:
        print("  No off-diagonal confusions found.")


def main() -> None:
    """Run FashionMNIST CNN error analysis with a confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = fashion_class_names()
    model = FashionCNN()

    print("FashionMNIST CNN error analysis")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    train_loader, val_loader, test_loader = make_fashion_mnist_train_val_test_dataloaders(
        data_dir="data",
        batch_size=64,
        val_fraction=0.1,
        seed=0,
        download=True,
    )

    train_fashion_classifier_experiment(
        model=model,
        train_loader=train_loader,
        eval_loader=val_loader,
        run_dir="outputs/fashion_experiments/cnn_error_analysis",
        log_dir="runs/fashion_experiments/cnn_error_analysis",
        checkpoint_name="fashion_cnn_error_analysis_checkpoint.pt",
        num_epochs=2,
        learning_rate=0.1,
        device=device,
        max_train_batches=100,
        max_eval_batches=50,
        model_name="FashionCNN",
    )

    test_metrics = evaluate_classifier_with_confusion_matrix(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=len(class_names),
        max_batches=50,
    )

    confusion_matrix = test_metrics["confusion_matrix"]
    per_class_accuracy = test_metrics["per_class_accuracy"]

    print("Final limited test metrics:")
    print(f"  test_loss: {test_metrics['loss']}")
    print(f"  test_accuracy: {test_metrics['accuracy']}")
    print(f"  test_examples: {test_metrics['num_examples']}")
    print(f"  test_batches: {test_metrics['num_batches']}")
    print()

    print_per_class_accuracy(
        class_names=class_names,
        confusion_matrix=confusion_matrix,
        per_class_accuracy=per_class_accuracy,
    )
    print()
    print_top_confusions(
        class_names=class_names,
        confusion_matrix=confusion_matrix,
    )


if __name__ == "__main__":
    main()
