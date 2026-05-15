import math

import torch

from dl_onboarding import (
    LinearRegressionModel,
    evaluate_regression,
    make_train_val_dataloaders,
    regression_metrics,
    split_tensor_dataset,
    train_with_validation,
)


def test_split_tensor_dataset_sizes() -> None:
    train_dataset, val_dataset = split_tensor_dataset(
        device=torch.device("cpu"),
        train_size=48,
    )

    assert len(train_dataset) == 48
    assert len(val_dataset) == 16
    assert len(train_dataset) + len(val_dataset) == 64


def test_train_val_dataloader_batch_shapes() -> None:
    train_loader, val_loader = make_train_val_dataloaders(
        device=torch.device("cpu"),
        batch_size=16,
    )

    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))

    assert train_x.shape == (16, 1)
    assert train_y.shape == (16, 1)
    assert val_x.shape == (16, 1)
    assert val_y.shape == (16, 1)


def test_regression_metrics_for_perfect_prediction() -> None:
    target = torch.tensor([[1.0], [2.0], [3.0]])
    prediction = target.clone()

    metrics = regression_metrics(prediction, target)

    assert math.isclose(metrics["mse"], 0.0)
    assert math.isclose(metrics["rmse"], 0.0)
    assert math.isclose(metrics["mae"], 0.0)
    assert math.isclose(metrics["r2"], 1.0)


def test_evaluate_regression_does_not_create_gradients() -> None:
    model = LinearRegressionModel()
    _, val_loader = make_train_val_dataloaders(
        device=torch.device("cpu"),
        batch_size=16,
    )

    assert all(parameter.grad is None for parameter in model.parameters())

    metrics = evaluate_regression(
        model=model,
        dataloader=val_loader,
        device=torch.device("cpu"),
    )

    assert metrics["num_examples"] == 16
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_with_validation_learns_and_reports_metrics() -> None:
    result = train_with_validation(
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["final_val_mse"] < result["initial_val_mse"]
    assert result["final_val_mse"] < 1e-4
    assert result["final_val_rmse"] < 1e-2
    assert result["final_val_mae"] < 1e-2
    assert result["final_val_r2"] > 0.999
    assert math.isclose(result["learned_weight"], 2.0, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(result["learned_bias"], -1.0, rel_tol=0.0, abs_tol=1e-2)
