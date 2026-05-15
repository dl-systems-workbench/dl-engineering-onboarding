import math

import torch

from dl_onboarding import (
    make_linear_regression_data,
    mse_loss,
    predict_linear,
    train_manual_linear_regression,
)


def test_make_linear_regression_data_shapes_and_values() -> None:
    x, y = make_linear_regression_data(torch.device("cpu"))

    assert x.shape == (64, 1)
    assert y.shape == (64, 1)
    assert math.isclose(float(y[0].item()), -3.0)
    assert math.isclose(float(y[-1].item()), 1.0)


def test_predict_linear_returns_expected_shape() -> None:
    x = torch.ones((4, 1))
    weight = torch.tensor([[2.0]])
    bias = torch.tensor([-1.0])

    prediction = predict_linear(x, weight, bias)

    assert prediction.shape == (4, 1)
    assert torch.allclose(prediction, torch.ones((4, 1)))


def test_mse_loss_is_zero_for_perfect_prediction() -> None:
    target = torch.tensor([[1.0], [2.0], [3.0]])
    prediction = target.clone()

    loss = mse_loss(prediction, target)

    assert math.isclose(float(loss.item()), 0.0)


def test_manual_training_reduces_loss_and_learns_parameters() -> None:
    result = train_manual_linear_regression(
        num_steps=200,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] < 1e-4
    assert math.isclose(result["learned_weight"], 2.0, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(result["learned_bias"], -1.0, rel_tol=0.0, abs_tol=1e-2)
    assert result["device"] == "cpu"
