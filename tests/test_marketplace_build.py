"""The Marketplace build must reject changed source before touching ECR."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "infra/marketplace/build_image.py"


def load_build_module():
    spec = importlib.util.spec_from_file_location("marketplace_build", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_tree_hash_is_independent_of_archive_manifest(tmp_path, monkeypatch):
    build = load_build_module()
    monkeypatch.chdir(tmp_path)
    source = Path("app.py")
    source.write_text("released = True\n")
    digest = hashlib.sha256(b"app.py\0released = True\n\0").hexdigest()
    Path("marketplace-source.json").write_text(json.dumps({
        "commit": build.SOURCE_COMMIT, "version": "1.1.1", "files": ["app.py"]}))
    config = SimpleNamespace(vectory_expected_source_tree_sha256=digest)
    assert build.verify_source(config) == digest

    source.write_text("released = False\n")
    with pytest.raises(RuntimeError, match="independently approved tree hash"):
        build.verify_source(config)

    source.write_text("released = True\n")
    Path("other.py").write_text("unreviewed = True\n")
    with pytest.raises(RuntimeError, match="file inventory changed"):
        build.verify_source(config)
