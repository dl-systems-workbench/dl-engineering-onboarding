from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor


def fashion_class_names() -> tuple[str, ...]:
    """Return FashionMNIST class names in label-index order."""
    return (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    )


class FashionMLP(nn.Module):
    """A small MLP baseline for FashionMNIST classification."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(fashion_class_names())),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of FashionMNIST images."""
        x = self.flatten(x)
        return self.classifier(x)


class FashionCNN(nn.Module):
    """A small CNN baseline for FashionMNIST classification."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(fashion_class_names())),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of FashionMNIST images."""
        features = self.feature_extractor(x)
        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_dataset_train_val(
    *,
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Split a dataset into deterministic train and validation subsets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("dataset must contain at least two examples.")

    val_size = round(dataset_size * val_fraction)
    val_size = max(1, min(dataset_size - 1, val_size))
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_subset, val_subset


def make_fashion_mnist_train_val_test_dataloaders(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create FashionMNIST train, validation, and test DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    root = Path(data_dir)

    full_train_dataset = datasets.FashionMNIST(
        root=str(root),
        train=True,
        download=download,
        transform=ToTensor(),
    )
    test_dataset = datasets.FashionMNIST(
        root=str(root),
        train=False,
        download=download,
        transform=ToTensor(),
    )

    train_dataset, val_dataset = split_dataset_train_val(
        dataset=full_train_dataset,
        val_fraction=val_fraction,
        seed=seed,
    )

    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def make_fashion_mnist_dataloaders(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 64,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create FashionMNIST train and test DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    root = Path(data_dir)

    train_dataset = datasets.FashionMNIST(
        root=str(root),
        train=True,
        download=download,
        transform=ToTensor(),
    )
    test_dataset = datasets.FashionMNIST(
        root=str(root),
        train=False,
        download=download,
        transform=ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def classification_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute classification accuracy from logits and integer labels."""
    predicted_labels = logits.argmax(dim=1)
    correct = (predicted_labels == labels).sum().item()

    return float(correct / labels.numel())


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Create a confusion matrix with rows=true labels and columns=predictions."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes).")

    if labels.ndim != 1:
        raise ValueError("labels must have shape (batch_size,).")

    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same batch size.")

    if logits.shape[1] != num_classes:
        raise ValueError("logits second dimension must equal num_classes.")

    predicted_labels = logits.argmax(dim=1)
    encoded_pairs = labels.to(torch.long) * num_classes + predicted_labels.to(torch.long)

    return torch.bincount(
        encoded_pairs,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def per_class_accuracy_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute per-class accuracy from a confusion matrix."""
    if confusion_matrix.ndim != 2:
        raise ValueError("confusion_matrix must be 2-dimensional.")

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix must be square.")

    matrix = confusion_matrix.to(dtype=torch.float32)
    correct_per_class = matrix.diag()
    support_per_class = matrix.sum(dim=1)

    return torch.where(
        support_per_class > 0,
        correct_per_class / support_per_class,
        torch.full_like(support_per_class, torch.nan),
    )


def evaluate_classifier_with_confusion_matrix(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    loss_fn: nn.Module | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a classifier and return aggregate metrics plus confusion matrix."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    active_loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches_seen = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = active_loss_fn(logits, labels)
            batch_confusion_matrix = confusion_matrix_from_logits(
                logits,
                labels,
                num_classes=num_classes,
            ).to(device)

            batch_size = labels.shape[0]
            predictions = logits.argmax(dim=1)

            total_correct += int((predictions == labels).sum().item())
            total_examples += int(batch_size)
            total_loss += float(loss.item()) * batch_size
            batches_seen += 1
            confusion_matrix += batch_confusion_matrix

            if max_batches is not None and batches_seen >= max_batches:
                break

    if was_training:
        model.train()

    if total_examples == 0:
        raise RuntimeError("No examples were evaluated.")

    confusion_matrix_cpu = confusion_matrix.cpu()
    per_class_accuracy = per_class_accuracy_from_confusion_matrix(
        confusion_matrix_cpu,
    )

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
        "num_batches": batches_seen,
        "confusion_matrix": confusion_matrix_cpu,
        "per_class_accuracy": per_class_accuracy,
    }


def evaluate_classifier(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate a classifier without gradient tracking."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    active_loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches_seen = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = active_loss_fn(logits, labels)

            batch_size = labels.shape[0]
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(batch_size)
            total_loss += float(loss.item()) * batch_size
            batches_seen += 1

            if max_batches is not None and batches_seen >= max_batches:
                break

    if was_training:
        model.train()

    if total_examples == 0:
        raise RuntimeError("No examples were evaluated.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": total_examples,
        "num_batches": batches_seen,
    }


def make_tiny_classification_dataloader(
    *,
    batch_size: int = 8,
) -> DataLoader:
    """Create a tiny deterministic fake image classification dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    images = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
    labels = torch.arange(20) % len(fashion_class_names())

    for index, label in enumerate(labels):
        images[index, 0, label, label] = 1.0

    dataset = TensorDataset(images, labels.long())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def train_fashion_mlp(
    *,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: torch.device | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Train FashionMLP and report classification metrics."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided.")

    active_device = device if device is not None else torch.device("cpu")
    model = FashionMLP().to(active_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    eval_loss_history: list[float] = []
    eval_accuracy_history: list[float] = []

    for _ in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(active_device)
            labels = labels.to(active_device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            predictions = logits.argmax(dim=1)

            total_train_loss += float(loss.item()) * batch_size
            total_train_correct += int((predictions == labels).sum().item())
            total_train_examples += int(batch_size)
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

        if total_train_examples == 0:
            raise RuntimeError("No training examples were processed.")

        train_loss_history.append(total_train_loss / total_train_examples)
        train_accuracy_history.append(total_train_correct / total_train_examples)

        eval_metrics = evaluate_classifier(
            model=model,
            dataloader=eval_loader,
            device=active_device,
            loss_fn=loss_fn,
            max_batches=max_eval_batches,
        )
        eval_loss_history.append(float(eval_metrics["loss"]))
        eval_accuracy_history.append(float(eval_metrics["accuracy"]))

    return {
        "initial_train_loss": train_loss_history[0],
        "final_train_loss": train_loss_history[-1],
        "initial_train_accuracy": train_accuracy_history[0],
        "final_train_accuracy": train_accuracy_history[-1],
        "initial_eval_loss": eval_loss_history[0],
        "final_eval_loss": eval_loss_history[-1],
        "initial_eval_accuracy": eval_accuracy_history[0],
        "final_eval_accuracy": eval_accuracy_history[-1],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "device": str(active_device),
    }


def train_fashion_cnn(
    *,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    hidden_size: int = 64,
    device: torch.device | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Train FashionCNN and report classification metrics."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")

    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided.")

    active_device = device if device is not None else torch.device("cpu")
    model = FashionCNN(hidden_size=hidden_size).to(active_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    eval_loss_history: list[float] = []
    eval_accuracy_history: list[float] = []

    for _ in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(active_device)
            labels = labels.to(active_device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            predictions = logits.argmax(dim=1)

            total_train_loss += float(loss.item()) * batch_size
            total_train_correct += int((predictions == labels).sum().item())
            total_train_examples += int(batch_size)

            if max_train_batches is not None and batch_index >= max_train_batches:
                break

        if total_train_examples == 0:
            raise RuntimeError("No training examples were processed.")

        train_loss_history.append(total_train_loss / total_train_examples)
        train_accuracy_history.append(total_train_correct / total_train_examples)

        eval_metrics = evaluate_classifier(
            model=model,
            dataloader=eval_loader,
            device=active_device,
            loss_fn=loss_fn,
            max_batches=max_eval_batches,
        )
        eval_loss_history.append(float(eval_metrics["loss"]))
        eval_accuracy_history.append(float(eval_metrics["accuracy"]))

    return {
        "initial_train_loss": train_loss_history[0],
        "final_train_loss": train_loss_history[-1],
        "initial_train_accuracy": train_accuracy_history[0],
        "final_train_accuracy": train_accuracy_history[-1],
        "initial_eval_loss": eval_loss_history[0],
        "final_eval_loss": eval_loss_history[-1],
        "initial_eval_accuracy": eval_accuracy_history[0],
        "final_eval_accuracy": eval_accuracy_history[-1],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "device": str(active_device),
    }


def train_fashion_classifier_experiment(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    run_dir: str | Path,
    log_dir: str | Path,
    checkpoint_name: str = "checkpoint.pt",
    num_epochs: int = 2,
    learning_rate: float = 0.1,
    device: torch.device | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Train a FashionMNIST classifier with TensorBoard logging and checkpointing."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided.")

    if max_eval_batches is not None and max_eval_batches <= 0:
        raise ValueError("max_eval_batches must be positive when provided.")

    if not checkpoint_name:
        raise ValueError("checkpoint_name must be non-empty.")

    active_device = device if device is not None else torch.device("cpu")
    active_model_name = model_name if model_name is not None else type(model).__name__

    run_path = Path(run_dir)
    log_path = Path(log_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    model = model.to(active_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=str(log_path))

    train_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    eval_loss_history: list[float] = []
    eval_accuracy_history: list[float] = []

    try:
        for epoch_index in range(1, num_epochs + 1):
            model.train()

            total_train_loss = 0.0
            total_train_correct = 0
            total_train_examples = 0

            for batch_index, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(active_device)
                labels = labels.to(active_device)

                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.shape[0]
                predictions = logits.argmax(dim=1)

                total_train_loss += float(loss.item()) * batch_size
                total_train_correct += int((predictions == labels).sum().item())
                total_train_examples += int(batch_size)

                if max_train_batches is not None and batch_index >= max_train_batches:
                    break

            if total_train_examples == 0:
                raise RuntimeError("No training examples were processed.")

            train_loss = total_train_loss / total_train_examples
            train_accuracy = total_train_correct / total_train_examples

            eval_metrics = evaluate_classifier(
                model=model,
                dataloader=eval_loader,
                device=active_device,
                loss_fn=loss_fn,
                max_batches=max_eval_batches,
            )

            eval_loss = float(eval_metrics["loss"])
            eval_accuracy = float(eval_metrics["accuracy"])

            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            eval_loss_history.append(eval_loss)
            eval_accuracy_history.append(eval_accuracy)

            writer.add_scalar("Loss/train", train_loss, epoch_index)
            writer.add_scalar("Loss/eval", eval_loss, epoch_index)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch_index)
            writer.add_scalar("Accuracy/eval", eval_accuracy, epoch_index)

        writer.flush()
    finally:
        writer.close()

    checkpoint_path = run_path / checkpoint_name

    result: dict[str, Any] = {
        "initial_train_loss": train_loss_history[0],
        "final_train_loss": train_loss_history[-1],
        "initial_train_accuracy": train_accuracy_history[0],
        "final_train_accuracy": train_accuracy_history[-1],
        "initial_eval_loss": eval_loss_history[0],
        "final_eval_loss": eval_loss_history[-1],
        "initial_eval_accuracy": eval_accuracy_history[0],
        "final_eval_accuracy": eval_accuracy_history[-1],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "device": str(active_device),
        "model_name": active_model_name,
        "checkpoint_path": str(checkpoint_path),
        "log_dir": str(log_path),
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "metrics": result,
        "metadata": {
            "model_name": active_model_name,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
            "device": str(active_device),
        },
    }

    torch.save(checkpoint, checkpoint_path)

    return result
