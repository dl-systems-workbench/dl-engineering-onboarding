# Handoff State

## User

Junior engineer onboarding into Deep Learning Engineering / Applied Research Engineering.

The long-term goal is to become a strong Deep Learning Engineer / Applied Research Engineer capable of building, debugging, evaluating, reproducing, and explaining modern ML systems.

## Machine

- Windows 10 Pro
- WSL2 Ubuntu 24.04.3 LTS
- Intel i7-9750H
- 16 GB RAM
- NVIDIA GTX 1650, 4 GB VRAM
- VS Code with WSL workflow

## Repo

- Local path: ~/ai-workspace/dl-engineering-onboarding
- GitHub remote: <public-repo-url>
- Branch: main
- Latest accepted feature commit before this handoff refresh: 8ed25f8 Add checkpointing basics

## Current Phase

Phase 1 — PyTorch Engineering Foundations

## Current Milestone

M1 — Minimal professional supervised PyTorch training stack.

The user has built the progression:

1. tensor/device/autograd basics
2. manual training loop
3. testing strategy
4. Dataset/DataLoader
5. nn.Module
6. torch.optim.SGD
7. train/validation split
8. regression metrics
9. checkpoint save/load

## Completed Work

### Environment and workflow

- Machine inspected.
- WSL2 verified.
- NVIDIA GPU visible in WSL through nvidia-smi.
- Git configured.
- Local repo initialized.
- GitHub SSH configured.
- GitHub remote created and pushed.
- VS Code WSL workflow verified.
- Repo structure created.
- uv Python environment created.
- pytest installed and verified.
- Ruff formatter and linter configured.
- Quality gate script added.
- AI tool policy documented.
- GitHub Copilot CLI verified.
- OpenAI Codex CLI installed and verified.
- Bubblewrap installed for Codex sandbox support.
- PyTorch and torchvision installed with CUDA 11.8 wheels.
- PyTorch verified on CPU and NVIDIA GTX 1650 GPU.

### PyTorch engineering

- Tensor metadata, device selection, and autograd utilities created.
- Manual linear regression training loop created.
- Dataset and DataLoader training loop created.
- LinearRegressionModel using nn.Module created.
- Optimizer-based training using torch.optim.SGD created.
- Train/validation split and regression metrics created.
- Checkpointing implemented with model_state_dict, optimizer_state_dict, epoch, metrics, and metadata.
- Checkpoint restore verified with predictions close to y = 2x - 1.

## Important Source Files

- src/dl_onboarding/tensor_lab.py
- src/dl_onboarding/manual_training.py
- src/dl_onboarding/data_loading.py
- src/dl_onboarding/module_training.py
- src/dl_onboarding/evaluation.py
- src/dl_onboarding/checkpointing.py
- src/dl_onboarding/__init__.py

## Important Test Files

- tests/test_system_info.py
- tests/test_tensor_lab.py
- tests/test_manual_training.py
- tests/test_data_loading.py
- tests/test_module_training.py
- tests/test_optimizer_training.py
- tests/test_evaluation.py
- tests/test_checkpointing.py

## Important Scripts

- scripts/quality_check.sh
- scripts/verify_torch.py
- scripts/tensor_autograd_lab.py
- scripts/manual_training_lab.py
- scripts/dataloader_lab.py
- scripts/module_training_lab.py
- scripts/optimizer_training_lab.py
- scripts/evaluation_lab.py
- scripts/checkpointing_lab.py

## Important Docs

- docs/TASKS.md
- docs/HANDOFF.md
- docs/DECISIONS.md
- docs/WORKFLOW.md
- docs/AI_TOOLS.md
- docs/PYTHON_ENV.md
- docs/TESTING.md

## Important Tool Versions / Signals

- Python: 3.12.3 through project .venv
- uv: installed and managing pyproject.toml plus uv.lock
- torch: 2.7.1+cu118
- torchvision: 0.22.1+cu118
- CUDA visible to PyTorch: True
- GPU: NVIDIA GeForce GTX 1650
- Codex CLI: 0.129.0
- bubblewrap: 0.9.0

## Current Quality Gate

Use:

    ./scripts/quality_check.sh

For PyTorch/GPU-related changes:

    ./scripts/quality_check.sh --torch

For safe Ruff auto-fix:

    ./scripts/quality_check.sh --fix

## Current Accepted Training Stack Pattern

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)
        loss.backward()
        optimizer.step()

Validation pattern:

    model.eval()

    with torch.no_grad():
        prediction = model(x_val)
        metrics = ...

Checkpoint pattern:

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "metadata": metadata,
    }, path)

Restore pattern:

    model = LinearRegressionModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

## Current Task

T1.8 — TensorBoard / Experiment Logging

## Next Expected Work

Add local experiment logging so train/validation curves can be inspected visually.

Likely files:

- src/dl_onboarding/experiment_logging.py
- tests/test_experiment_logging.py
- scripts/tensorboard_lab.py

## Important Rules

- Work under /home/****/ai-workspace, not /mnt/c.
- Use Git for every meaningful change.
- Verify before claiming success.
- Keep commits small and reviewable.
- Do not commit .venv, caches, checkpoints, logs, outputs, secrets, or API keys.
- Use uv for Python dependency management.
- Run the quality gate before committing code.
- AI may propose; the engineer must inspect, understand, test, and commit.
- Use local machine for development and small experiments.
- Use cloud GPU only when justified.
- Tests should be planned before implementation.
- Default unit tests should run on CPU.
- GPU behavior should be checked explicitly through scripts or quality gate modes.
- Documentation tasks require content review, not only Ruff/pytest.
