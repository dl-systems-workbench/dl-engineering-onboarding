from __future__ import annotations


def test_top_k_predictions_from_logits_returns_readable_predictions() -> None:
    import torch

    from dl_onboarding import top_k_predictions_from_logits

    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [5.0, 1.0, 0.0],
        ],
    )
    class_names = ("zero", "one", "two")

    predictions = top_k_predictions_from_logits(
        logits,
        class_names=class_names,
        k=2,
    )

    assert len(predictions) == 2
    assert predictions[0][0]["class_name"] == "one"
    assert predictions[0][1]["class_name"] == "two"
    assert predictions[1][0]["class_name"] == "zero"
    assert 0.0 <= predictions[0][0]["probability"] <= 1.0


def test_load_model_from_state_dict_checkpoint_restores_eval_model(
    tmp_path,
) -> None:
    import torch

    from dl_onboarding import (
        FashionCNN,
        load_model_from_state_dict_checkpoint,
        make_tiny_classification_dataloader,
        train_fashion_classifier_experiment,
    )

    torch.manual_seed(0)

    train_loader = make_tiny_classification_dataloader(batch_size=10)
    eval_loader = make_tiny_classification_dataloader(batch_size=10)

    trained_model = FashionCNN()
    metrics = train_fashion_classifier_experiment(
        model=trained_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_dir=tmp_path / "outputs",
        log_dir=tmp_path / "runs",
        num_epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    fresh_model = FashionCNN()
    loaded_model, checkpoint = load_model_from_state_dict_checkpoint(
        model=fresh_model,
        checkpoint_path=metrics["checkpoint_path"],
        device=torch.device("cpu"),
    )

    assert loaded_model.training is False
    assert checkpoint["metadata"]["model_name"] == "FashionCNN"

    images, _ = next(iter(eval_loader))

    trained_model.eval()
    with torch.no_grad():
        trained_logits = trained_model(images)
        loaded_logits = loaded_model(images)

    assert torch.allclose(trained_logits, loaded_logits)


def test_predict_top_k_for_batch_returns_one_prediction_list_per_example() -> None:
    import torch

    from dl_onboarding import (
        FashionCNN,
        fashion_class_names,
        predict_top_k_for_batch,
    )

    model = FashionCNN()
    images = torch.randn((4, 1, 28, 28))

    predictions = predict_top_k_for_batch(
        model=model,
        images=images,
        class_names=fashion_class_names(),
        device=torch.device("cpu"),
        k=3,
    )

    assert len(predictions) == 4
    assert all(len(example_predictions) == 3 for example_predictions in predictions)
    assert all(
        "class_name" in prediction and "class_index" in prediction and "probability" in prediction
        for example_predictions in predictions
        for prediction in example_predictions
    )
