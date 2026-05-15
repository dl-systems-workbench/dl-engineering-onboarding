import math

import torch

from dl_onboarding import matrix_smoke, quadratic_autograd, select_device, tensor_summary


def test_select_device_can_force_cpu() -> None:
    device = select_device(prefer_cuda=False)

    assert device.type == "cpu"


def test_tensor_summary_reports_basic_metadata() -> None:
    tensor = torch.ones((2, 3), dtype=torch.float32)
    summary = tensor_summary(tensor)

    assert summary["shape"] == "(2, 3)"
    assert summary["dtype"] == "torch.float32"
    assert summary["device"] == "cpu"
    assert summary["requires_grad"] == "False"


def test_quadratic_autograd_matches_analytic_gradient() -> None:
    summary = quadratic_autograd(2.0)

    assert math.isclose(summary["x"], 2.0)
    assert math.isclose(summary["y"], 12.0)
    assert math.isclose(summary["dy_dx"], 7.0)


def test_matrix_smoke_cpu_result() -> None:
    result = matrix_smoke(torch.device("cpu"))

    assert result["shape"] == (2, 2)
    assert result["device"] == "cpu"
    assert math.isclose(result["sum"], 30.0)
