#!/usr/bin/env bash
set -euo pipefail

RUN_FIX=false
RUN_TORCH=false

for arg in "$@"; do
    case "$arg" in
        --fix)
            RUN_FIX=true
            ;;
        --torch)
            RUN_TORCH=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./scripts/quality_check.sh [--fix] [--torch]"
            exit 2
            ;;
    esac
done

echo "==> Formatting with Ruff"
uv run ruff format .

if [ "$RUN_FIX" = true ]; then
    echo "==> Running Ruff auto-fix for fixable lint issues"
    uv run ruff check . --fix
fi

echo "==> Running Ruff lint checks"
uv run ruff check .

echo "==> Running pytest"
uv run python -m pytest -q

if [ "$RUN_TORCH" = true ]; then
    echo "==> Running PyTorch verification"
    uv run python scripts/verify_torch.py
fi

echo "==> Quality checks passed"
