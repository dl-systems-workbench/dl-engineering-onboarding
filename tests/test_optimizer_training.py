import math

import torch

from dl_onboarding import train_module_with_optimizer


def test_optimizer_training_learns_parameters() -> None:
    result = train_module_with_optimizer(
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] < 1e-4
    assert math.isclose(result["learned_weight"], 2.0, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(result["learned_bias"], -1.0, rel_tol=0.0, abs_tol=1e-2)


def test_optimizer_training_reports_optimizer_name_and_learning_rate() -> None:
    result = train_module_with_optimizer(
        num_epochs=5,
        learning_rate=0.05,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["optimizer_name"] == "SGD"
    assert math.isclose(result["learning_rate"], 0.05)
    assert result["batch_size"] == 16
    assert result["device"] == "cpu"
