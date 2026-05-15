"""Small runtime-inspection helpers used during onboarding."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def get_runtime_summary() -> dict[str, str]:
    """Return a small summary of the active Python runtime.

    This intentionally avoids ML dependencies. It is a tiny first function
    that proves our package can be imported and tested.
    """
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable)),
        "platform": platform.platform(),
    }
