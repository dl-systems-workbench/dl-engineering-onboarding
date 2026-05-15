# Testing Strategy

## Purpose

Tests are executable evidence that important behavior still works.

In this repo, tests are not decoration. They are part of the engineering contract.

## Core Rule

Before implementing code, define what behavior must be protected.

Every coding task should include a test plan before implementation.

## Test Plan Before Implementation

For each task, answer:

- What behavior are we trying to protect?
- Which tests will we write?
- Why are these tests meaningful?
- What bugs would these tests catch?
- What do these tests not prove?
- Should the tests run on CPU, GPU, or both?

## Pytest Discovery Conventions

Pytest discovers tests by naming convention.

Use:

- files named test_*.py
- test functions named test_*

Example:

- source file: src/dl_onboarding/manual_training.py
- test file: tests/test_manual_training.py

## What Makes a Good Test

A good test should usually be:

- focused on one behavior
- small and fast
- deterministic
- meaningful
- easy to understand when it fails
- based on a real expected contract
- independent of local hardware unless explicitly testing hardware behavior

## What Makes a Weak Test

Weak tests often:

- only check that code runs without crashing
- contain no meaningful assertions
- test many unrelated behaviors at once
- depend on random behavior without controlling it
- require GPU when CPU would be enough
- pass even if the important behavior is wrong

## CPU vs GPU Policy

Default unit tests should run on CPU.

Reason:

- CPU tests are portable.
- CPU tests are cheaper.
- CPU tests can run in CI.
- CPU tests avoid driver and hardware dependency.

GPU behavior should be checked through explicit smoke scripts or optional quality-gate modes.

Current GPU verification command:

    ./scripts/quality_check.sh --torch

## Scripts vs Tests

Use tests for behavior that should always be verified automatically.

Use scripts for:

- human-readable inspection
- local hardware checks
- demos
- smoke runs
- debugging
- exploratory verification

Tests answer:

    Is the behavior correct?

Scripts answer:

    What does the behavior look like when I run it?

## Current Test Categories

Current examples:

- tests/test_system_info.py
  - verifies runtime/environment helpers

- tests/test_tensor_lab.py
  - verifies tensor metadata
  - verifies autograd result
  - verifies matrix smoke behavior on CPU

- tests/test_manual_training.py
  - verifies synthetic dataset shape and values
  - verifies forward prediction formula
  - verifies MSE loss behavior
  - verifies manual training reduces loss and learns parameters

## Manual Training Test Plan

For the manual training loop, we test:

1. Dataset contract
   - x and y have shape (64, 1)
   - y follows y = 2x - 1 at known endpoints

2. Prediction contract
   - prediction has expected shape
   - x @ weight + bias gives expected values

3. Loss contract
   - perfect prediction gives zero MSE loss

4. Learning contract
   - final loss is smaller than initial loss
   - final loss is very small
   - learned weight is close to 2
   - learned bias is close to -1

## What Current Tests Do Not Prove

Current tests do not yet prove:

- real-world generalization
- DataLoader behavior
- nn.Module usage
- optimizer usage
- batching
- checkpointing
- robustness to noisy data
- model performance on real datasets
- distributed training
- mixed precision
- production deployment readiness

## Floating Point Assertions

Avoid exact equality for floating point values when calculations are involved.

Use:

- math.isclose for Python floats
- torch.allclose for simple tensor comparisons
- torch.testing.assert_close for richer PyTorch tensor tests later

## Test Count Policy

There is no fixed number of tests per function.

Instead, test:

- the main contract
- important edge cases
- likely failure modes
- shape and device expectations when relevant
- learning behavior for training code

For small utility functions, one or two tests may be enough.

For ML training components, tests should usually cover:

- data shape
- forward output shape
- loss sanity
- gradient or learning behavior
- error handling where relevant

## Review Checklist

Before accepting a coding task, check:

- Do tests pass?
- Does Ruff pass?
- Do tests contain meaningful assertions?
- Do tests protect the intended behavior?
- Are CPU and GPU responsibilities separated correctly?
- Is there a script if human inspection is useful?
- Does the task evidence explain what the tests prove?
- Does the task evidence explain what the tests do not prove?

## Future Improvements

Planned improvements:

- Add torch.testing.assert_close for tensor-specific tests.
- Add pytest-cov when the codebase is large enough for coverage to be meaningful.
- Add GitHub Actions CI.
- Consider SonarCloud or similar tools later if the repo becomes a larger portfolio project.

## Team Rule

A task is not accepted only because the code runs.

A task is accepted when:

- the behavior is implemented
- the tests protect the intended contract
- quality gates pass
- the engineer can explain what is tested and what is not tested
