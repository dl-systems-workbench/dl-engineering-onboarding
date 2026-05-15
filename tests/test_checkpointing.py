from pathlib import Path

import torch

from dl_onboarding import (
    load_checkpoint,
    restore_model_and_optimizer,
    train_and_save_checkpoint,
)


def test_train_and_save_checkpoint_creates_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "linear_regression_checkpoint.pt"

    result = train_and_save_checkpoint(
        path=checkpoint_path,
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert checkpoint_path.exists()
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["final_val_mse"] < 1e-4
    assert result["final_val_r2"] > 0.999


def test_checkpoint_contains_required_keys(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "linear_regression_checkpoint.pt"

    train_and_save_checkpoint(
        path=checkpoint_path,
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert set(checkpoint.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "metrics",
        "metadata",
    }
    assert checkpoint["epoch"] == 100


def test_restore_model_and_optimizer_recovers_predictions(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "linear_regression_checkpoint.pt"

    train_and_save_checkpoint(
        path=checkpoint_path,
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    model, optimizer, checkpoint = restore_model_and_optimizer(
        path=checkpoint_path,
        device=torch.device("cpu"),
        learning_rate=0.1,
    )

    x = torch.tensor([[-1.0], [0.0], [1.0]])
    expected = 2.0 * x - 1.0

    with torch.no_grad():
        prediction = model(x)

    torch.testing.assert_close(prediction, expected, rtol=0.0, atol=1e-2)
    assert optimizer.__class__.__name__ == "SGD"
    assert checkpoint["metadata"]["optimizer"] == "SGD"


def test_checkpoint_metadata_is_preserved(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "linear_regression_checkpoint.pt"

    train_and_save_checkpoint(
        path=checkpoint_path,
        num_epochs=100,
        learning_rate=0.05,
        batch_size=16,
        device=torch.device("cpu"),
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    metadata = checkpoint["metadata"]

    assert metadata["model_class"] == "LinearRegressionModel"
    assert metadata["optimizer"] == "SGD"
    assert metadata["learning_rate"] == 0.05
    assert metadata["batch_size"] == 16
    assert metadata["device"] == "cpu"
