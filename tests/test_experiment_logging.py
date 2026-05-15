from pathlib import Path

import torch

from dl_onboarding import find_tensorboard_event_files, train_with_tensorboard_logging


def test_tensorboard_training_creates_event_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard_run"

    result = train_with_tensorboard_logging(
        log_dir=log_dir,
        num_epochs=10,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    event_files = find_tensorboard_event_files(log_dir)

    assert result["log_dir"] == str(log_dir)
    assert result["event_file_count"] >= 1
    assert len(event_files) >= 1


def test_tensorboard_training_still_learns(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard_run"

    result = train_with_tensorboard_logging(
        log_dir=log_dir,
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["final_val_mse"] < result["initial_val_mse"]
    assert result["final_val_mse"] < 1e-4
    assert result["final_val_r2"] > 0.999
    assert abs(result["learned_weight"] - 2.0) < 1e-2
    assert abs(result["learned_bias"] + 1.0) < 1e-2
