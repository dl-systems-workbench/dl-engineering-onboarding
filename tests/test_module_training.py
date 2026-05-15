import math

import torch

from dl_onboarding import LinearRegressionModel, train_module_linear_regression


def test_module_forward_output_shape_and_initial_values() -> None:
    model = LinearRegressionModel()
    x = torch.ones((4, 1))

    prediction = model(x)

    assert prediction.shape == (4, 1)
    torch.testing.assert_close(prediction, torch.zeros((4, 1)))


def test_module_exposes_two_trainable_parameters() -> None:
    model = LinearRegressionModel()
    parameters = list(model.parameters())

    assert len(parameters) == 2
    assert parameters[0].shape == (1, 1)
    assert parameters[1].shape == (1,)
    assert all(parameter.requires_grad for parameter in parameters)


def test_module_to_cpu_moves_parameters_to_cpu() -> None:
    model = LinearRegressionModel()
    model.to(torch.device("cpu"))

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_module_state_dict_contains_weight_and_bias() -> None:
    model = LinearRegressionModel()

    assert set(model.state_dict().keys()) == {"weight", "bias"}


def test_module_training_learns_parameters() -> None:
    result = train_module_linear_regression(
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] < 1e-4
    assert math.isclose(result["learned_weight"], 2.0, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(result["learned_bias"], -1.0, rel_tol=0.0, abs_tol=1e-2)
    assert set(result["state_dict_keys"]) == {"weight", "bias"}
