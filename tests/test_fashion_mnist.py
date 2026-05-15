import math

import torch
from torch import nn

from dl_onboarding import (
    FashionMLP,
    classification_accuracy,
    fashion_class_names,
    make_tiny_classification_dataloader,
    train_fashion_mlp,
)


def test_fashion_class_names_has_ten_classes() -> None:
    class_names = fashion_class_names()

    assert len(class_names) == 10
    assert class_names[0] == "T-shirt/top"
    assert class_names[-1] == "Ankle boot"


def test_fashion_mlp_forward_shape() -> None:
    model = FashionMLP(hidden_size=128)
    images = torch.zeros((4, 1, 28, 28))

    logits = model(images)

    assert logits.shape == (4, 10)


def test_classification_accuracy_from_logits() -> None:
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 1, 1])

    accuracy = classification_accuracy(logits, labels)

    assert math.isclose(accuracy, 2 / 3)


def test_cross_entropy_loss_accepts_logits_and_class_indices() -> None:
    model = FashionMLP(hidden_size=32)
    images = torch.zeros((4, 1, 28, 28))
    labels = torch.tensor([0, 1, 2, 3])
    loss_fn = nn.CrossEntropyLoss()

    logits = model(images)
    loss = loss_fn(logits, labels)

    assert logits.shape == (4, 10)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_tiny_classification_training_learns() -> None:
    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)

    result = train_fashion_mlp(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epochs=30,
        learning_rate=0.5,
        device=torch.device("cpu"),
    )

    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["final_eval_accuracy"] >= result["initial_eval_accuracy"]
    assert result["final_eval_accuracy"] > 0.8


def test_fashion_cnn_forward_shape() -> None:
    import torch

    from dl_onboarding import FashionCNN, fashion_class_names

    model = FashionCNN()
    images = torch.randn((4, 1, 28, 28))

    logits = model(images)

    assert logits.shape == (4, len(fashion_class_names()))


def test_fashion_cnn_cross_entropy_loss_compatible() -> None:
    import torch
    from torch import nn

    from dl_onboarding import FashionCNN

    model = FashionCNN()
    images = torch.randn((4, 1, 28, 28))
    labels = torch.tensor([0, 1, 2, 3])

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))


def test_fashion_cnn_parameter_count_matches_architecture() -> None:
    from dl_onboarding import FashionCNN, count_trainable_parameters

    model = FashionCNN()

    assert count_trainable_parameters(model) == 105_866


def test_fashion_cnn_training_step_updates_parameters() -> None:
    import torch
    from torch import nn

    from dl_onboarding import FashionCNN

    torch.manual_seed(0)

    model = FashionCNN()
    images = torch.zeros((10, 1, 28, 28), dtype=torch.float32)
    labels = torch.arange(10, dtype=torch.long)

    for index, label in enumerate(labels):
        images[index, 0, label, label] = 1.0

    before_step = [parameter.detach().clone() for parameter in model.parameters()]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    after_step = [parameter.detach().clone() for parameter in model.parameters()]

    assert any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before_step,
            after_step,
            strict=True,
        )
    )


def test_train_fashion_cnn_reports_metrics() -> None:
    import torch

    from dl_onboarding import make_tiny_classification_dataloader, train_fashion_cnn

    torch.manual_seed(0)

    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)

    metrics = train_fashion_cnn(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    assert metrics["num_epochs"] == 1
    assert metrics["learning_rate"] == 0.1
    assert metrics["hidden_size"] == 64
    assert metrics["device"] == "cpu"
    assert metrics["final_train_loss"] > 0.0
    assert metrics["final_eval_loss"] > 0.0
    assert 0.0 <= metrics["final_train_accuracy"] <= 1.0
    assert 0.0 <= metrics["final_eval_accuracy"] <= 1.0


def test_fashion_classifier_experiment_writes_checkpoint_and_tensorboard(
    tmp_path,
) -> None:
    from pathlib import Path

    import torch

    from dl_onboarding import (
        FashionCNN,
        find_tensorboard_event_files,
        make_tiny_classification_dataloader,
        train_fashion_classifier_experiment,
    )

    torch.manual_seed(0)

    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)

    metrics = train_fashion_classifier_experiment(
        model=FashionCNN(),
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_dir=tmp_path / "outputs",
        log_dir=tmp_path / "runs",
        num_epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    checkpoint_path = Path(str(metrics["checkpoint_path"]))
    log_dir = Path(str(metrics["log_dir"]))

    assert checkpoint_path.exists()
    assert log_dir.exists()
    assert find_tensorboard_event_files(log_dir)
    assert metrics["model_name"] == "FashionCNN"
    assert metrics["num_epochs"] == 1
    assert metrics["device"] == "cpu"


def test_fashion_classifier_experiment_checkpoint_has_required_keys(
    tmp_path,
) -> None:
    import torch

    from dl_onboarding import (
        FashionCNN,
        make_tiny_classification_dataloader,
        train_fashion_classifier_experiment,
    )

    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)

    metrics = train_fashion_classifier_experiment(
        model=FashionCNN(),
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_dir=tmp_path / "outputs",
        log_dir=tmp_path / "runs",
        num_epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    checkpoint = torch.load(
        metrics["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )

    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "metrics",
        "metadata",
    }.issubset(checkpoint)

    assert checkpoint["epoch"] == 1
    assert checkpoint["metadata"]["model_name"] == "FashionCNN"
    assert checkpoint["metadata"]["learning_rate"] == 0.1


def test_fashion_classifier_experiment_checkpoint_restores_predictions(
    tmp_path,
) -> None:
    import torch

    from dl_onboarding import (
        FashionCNN,
        make_tiny_classification_dataloader,
        train_fashion_classifier_experiment,
    )

    torch.manual_seed(0)

    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)
    model = FashionCNN()

    metrics = train_fashion_classifier_experiment(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_dir=tmp_path / "outputs",
        log_dir=tmp_path / "runs",
        num_epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    images, _ = next(iter(eval_loader))

    model.eval()
    with torch.no_grad():
        trained_logits = model(images)

    checkpoint = torch.load(
        metrics["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )

    restored_model = FashionCNN()
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()

    with torch.no_grad():
        restored_logits = restored_model(images)

    assert torch.allclose(trained_logits, restored_logits)
