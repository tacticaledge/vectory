"""Build helpers for packaging the Vectory Streamlit app."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


RUNTIME_PATHS = [
    "app.py",
    ".streamlit",
    "pages",
    "components",
    "data",
    "docs",
    "scripts/run_vectory_benchmark.py",
]


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        source_root = Path(__file__).resolve().parent
        target_root = Path(self.build_lib) / "vectory_cli" / "streamlit_app"
        target_root.mkdir(parents=True, exist_ok=True)

        for relative in RUNTIME_PATHS:
            source = source_root / relative
            target = target_root / relative
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


setup(cmdclass={"build_py": build_py})
