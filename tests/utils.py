"""Public-package test helpers that do not depend on Tactical Edge private libraries."""

from pathlib import Path

import pytest


def pytest_this_file(path: str | Path) -> int:
    """Run one test module when it is executed directly."""
    return pytest.main([str(path)])
