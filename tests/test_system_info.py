from dl_onboarding import get_runtime_summary


def test_runtime_summary_contains_expected_keys() -> None:
    summary = get_runtime_summary()

    assert "python_version" in summary
    assert "python_executable" in summary
    assert "platform" in summary


def test_runtime_summary_uses_project_virtual_environment() -> None:
    summary = get_runtime_summary()

    assert ".venv" in summary["python_executable"]
