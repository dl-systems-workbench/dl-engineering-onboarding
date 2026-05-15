# AI Tool Operating Rules

## Purpose

This document defines how AI coding tools may be used in this repo.

AI tools are allowed, but they are not a replacement for engineering judgment.

## Approved AI Tool Uses

AI tools may be used for:

- explaining unfamiliar commands
- explaining unfamiliar code
- suggesting implementation options
- generating small draft snippets
- debugging error messages
- reviewing diffs
- proposing tests
- improving documentation
- summarizing repo state
- suggesting refactors

## Prohibited or Risky Uses

Do not use AI tools to:

- blindly apply large changes
- commit code you do not understand
- bypass tests
- hide errors
- paste secrets, tokens, private keys, API keys, or credentials
- modify many unrelated files in one change
- generate large ML systems without incremental review
- accept code without running quality gates
- run commands with broad permissions unless you understand the risk

## Required Safety Loop

Before committing AI-assisted changes, always run:

~~~bash
git status
git diff
uv run ruff format .
uv run ruff check .
uv run python -m pytest -q
git status
~~~

## Human Review Rule

The engineer must be able to explain:

- what changed
- why it changed
- which files changed
- how it was verified
- what risks remain

If you cannot explain the change, do not commit it.

## Prompt Template

Use this template when asking an AI tool for help:

~~~text
Context:
I am working in a Python ML engineering repo using uv, pytest, ruff, and a src/ layout.

Goal:
<what I want to accomplish>

Relevant files:
<file paths>

Constraints:
- Keep the change small.
- Do not modify unrelated files.
- Do not add dependencies unless explicitly needed.
- Explain the code and imports.
- Include verification commands.

Expected output:
<patch, explanation, test, or debugging plan>
~~~

## Approved Local AI Tools

Current:

- ChatGPT conversation for team-lead guidance and review
- GitHub Copilot in VS Code
- GitHub Copilot CLI, if installed and authenticated

Planned:

- OpenAI Codex CLI using ChatGPT sign-in
- Optional OpenAI Codex IDE extension if useful
- Optional ChatGPT GitHub connection for read-only repository analysis, if available in the current ChatGPT experience

## Codex Usage Rule

Use Codex for small, reviewable tasks only.

Good first Codex tasks:

- explain a file
- suggest a test
- review a diff
- propose a small refactor
- identify likely bugs

Avoid early Codex tasks like:

- build the whole training system
- rewrite the entire repo
- add multiple libraries at once
- run broad autonomous changes
- use unrestricted permissions

## Team Rule

AI may propose.
The engineer must inspect, understand, test, and commit.
