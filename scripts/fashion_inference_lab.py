from __future__ import annotations

from pathlib import Path

import torch

from dl_onboarding import (
    FashionCNN,
    fashion_class_names,
    load_model_from_state_dict_checkpoint,
    make_fashion_mnist_train_val_test_dataloaders,
    predict_top_k_for_batch,
    train_fashion_classifier_experiment,
)


def main() -> None:
    """Train a small checkpoint if needed, then load it for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(
        "outputs/fashion_experiments/cnn_inference/fashion_cnn_checkpoint.pt",
    )

    print("FashionMNIST checkpoint inference lab")
    print(f"Device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")

    train_loader, val_loader, test_loader = make_fashion_mnist_train_val_test_dataloaders(
        data_dir="data",
        batch_size=64,
        val_fraction=0.1,
        seed=0,
        download=True,
    )

    if not checkpoint_path.exists():
        print("Checkpoint not found. Creating a small training checkpoint first.")
        train_fashion_classifier_experiment(
            model=FashionCNN(),
            train_loader=train_loader,
            eval_loader=val_loader,
            run_dir=checkpoint_path.parent,
            log_dir="runs/fashion_experiments/cnn_inference",
            checkpoint_name=checkpoint_path.name,
            num_epochs=2,
            learning_rate=0.1,
            device=device,
            max_train_batches=100,
            max_eval_batches=50,
            model_name="FashionCNN",
        )
    else:
        print("Checkpoint found. Loading existing checkpoint.")

    model, checkpoint = load_model_from_state_dict_checkpoint(
        model=FashionCNN(),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    metadata = checkpoint.get("metadata", {})
    print("Loaded checkpoint metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    images, labels = next(iter(test_loader))
    predictions = predict_top_k_for_batch(
        model=model,
        images=images[:5],
        class_names=fashion_class_names(),
        device=device,
        k=3,
    )

    class_names = fashion_class_names()

    print()
    print("Top-3 predictions for first 5 test examples:")
    for example_index, example_predictions in enumerate(predictions):
        true_label = int(labels[example_index].item())
        print(f"Example {example_index}: true={class_names[true_label]!r}")

        for rank, prediction in enumerate(example_predictions, start=1):
            print(f"  top{rank}: {prediction['class_name']!r} prob={prediction['probability']:.4f}")


if __name__ == "__main__":
    main()
