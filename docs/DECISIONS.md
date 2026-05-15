# Decisions Log

## D0.1 — Use WSL2 Ubuntu as primary development environment

Decision:
Use WSL2 Ubuntu for ML development.

Reason:
Linux tooling is standard in ML engineering, and WSL2 gives a Linux-like workflow on Windows.

## D0.2 — Keep repos under Linux home

Decision:
Store active repos under `/home/****/ai-workspace`.

Reason:
Avoid WSL performance/path/tooling problems caused by working under `/mnt/c`.

## D0.3 — Use GitHub SSH remote

Decision:
Use SSH for GitHub remote operations.

Reason:
SSH is secure, standard, and avoids repeated password/token prompts.

## D0.4 — Keep repository private during onboarding

Decision:
Use a private GitHub repo initially.

Reason:
The repo is a learning/workspace repo. Public portfolio repos will be created later once artifacts are polished.

## D0.5 — Local machine is for development and small experiments

Decision:
Use the local GTX 1650 for small experiments only.

Reason:
4 GB VRAM is useful for smoke tests and tiny models, but not for serious modern model training.

## D0.6 — Add a reusable local quality gate script

Decision:
Add scripts/quality_check.sh as the standard local quality gate.

Reason:
The PyTorch verification task exposed a real workflow risk: formatting can pass while linting fails, and tests can pass while code quality fails. A reusable script reduces the chance of forgetting one required check.

Policy:
Before committing code changes, run ./scripts/quality_check.sh.

Use ./scripts/quality_check.sh --fix when you want Ruff to apply safe automatic fixes for fixable lint issues.

Use ./scripts/quality_check.sh --torch when a change touches PyTorch, CUDA, or environment behavior.

Future:
Add pre-commit after the user understands the manual quality loop.

## D1.1 — Define testing strategy before expanding PyTorch code

Decision:
Add docs/TESTING.md and require a test plan before implementation for future coding tasks.

Reason:
The project started writing tests before the testing strategy was explicitly explained. This created a knowledge gap around what tests are for, how to judge whether they are good, and what they do not prove.

Policy:
Future coding tasks should include a test plan before implementation.

The engineer must be able to explain:

- what each test protects
- why the assertion is meaningful
- what bug the test would catch
- whether the test belongs in pytest or a script
- what the test does not prove

Future:
Add torch.testing.assert_close, pytest-cov, CI, and optional Sonar-style quality reporting when appropriate.

## D1.2 — Use state_dict-based checkpointing

Decision:
Use state_dict-based checkpointing instead of saving full Python model objects.

Reason:
state_dict-based checkpointing separates model code from learned parameter values. This is the standard PyTorch pattern for portable model save/load workflows.

Checkpoint contents:
- model_state_dict
- optimizer_state_dict
- epoch
- metrics
- metadata

Policy:
Generated checkpoint files belong under outputs/ and must not be committed to Git.

Current checkpoint output path:
outputs/checkpoints/linear_regression_checkpoint.pt

Limitations:
The current checkpoint proves save/load and restored predictions. It does not yet prove exact resume determinism, random-number-generator restoration, scheduler state, best-checkpoint selection, distributed checkpointing, or fault-tolerant cloud checkpointing.

Future:
Add checkpoint roundtrip tests for resume training, best-validation checkpoint selection, and config-aware checkpoint metadata later.

## D0.7 — Public repo is now the source of truth

Decision: The onboarding repository is now public under the `dl-systems-workbench` organization and should be treated as the source of truth for future assistant sessions.

Reason: Public availability makes the repo directly inspectable and reduces dependence on long pasted handoffs. It also makes the repo a stronger professional artifact.

Policy:
- Future assistants should inspect the public repo before assigning new work.
- If source code and docs disagree, repair repo state before adding new feature work.
- `AGENTS.md`, `README.md`, `docs/TASKS.md`, `docs/HANDOFF.md`, and `docs/DECISIONS.md` must stay aligned after meaningful milestones.

Limitations:
- The repo is still an onboarding repo, not yet a polished portfolio project.
- Public visibility increases the importance of keeping docs accurate and avoiding secrets or generated artifacts.

Future:
- Consider GitHub Actions CI after the user understands the local quality loop.
- Consider GitHub Issues or milestone tags when task tracking outgrows Markdown.
