import math

import torch

from dl_onboarding import make_dataloader, make_tensor_dataset, train_with_dataloader


def test_tensor_dataset_length() -> None:
    dataset = make_tensor_dataset(torch.device("cpu"))

    assert len(dataset) == 64


def test_tensor_dataset_single_sample_shape_and_value() -> None:
    dataset = make_tensor_dataset(torch.device("cpu"))
    x, y = dataset[0]

    assert x.shape == (1,)
    assert y.shape == (1,)
    assert math.isclose(float(x.item()), -1.0)
    assert math.isclose(float(y.item()), -3.0)


def test_dataloader_batch_shapes_without_shuffle() -> None:
    dataloader = make_dataloader(
        device=torch.device("cpu"),
        batch_size=16,
        shuffle=False,
    )

    x_batch, y_batch = next(iter(dataloader))

    assert x_batch.shape == (16, 1)
    assert y_batch.shape == (16, 1)


def test_dataloader_number_of_batches() -> None:
    dataloader = make_dataloader(
        device=torch.device("cpu"),
        batch_size=16,
        shuffle=False,
    )

    batches = list(dataloader)

    assert len(batches) == 4


def test_dataloader_training_learns_parameters() -> None:
    result = train_with_dataloader(
        num_epochs=100,
        learning_rate=0.1,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] < 1e-4
    assert math.isclose(result["learned_weight"], 2.0, rel_tol=0.0, abs_tol=1e-2)
    assert math.isclose(result["learned_bias"], -1.0, rel_tol=0.0, abs_tol=1e-2)
    assert result["batch_size"] == 16
    assert result["device"] == "cpu"
